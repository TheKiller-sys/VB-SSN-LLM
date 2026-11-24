"""SNN-LM: transformer estilo GPT-2 con FFN de spikes y capa de razonamiento opcional."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .rope import precompute_rope


class SNNLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.head.weight = self.tok_emb.weight

        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope(head_dim, max_len=config.seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(
                    p, mean=0.0, std=config.init_std / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------ fwd
    def _forward(self, idx, kv_caches=None, start_pos: int = 0):
        x = self.drop(self.tok_emb(idx))
        new_kv_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, self.rope_cos, self.rope_sin, kv, start_pos)
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)
        logits = self.head(self.ln_f(x))
        return logits, new_kv_caches

    def forward(self, idx, targets=None, kv_caches=None, start_pos: int = 0):
        logits, new_kv_caches = self._forward(idx, kv_caches, start_pos)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        if kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss

    # ------------------------------------------------------------- sampling
    @staticmethod
    def _apply_sampling(logits, temperature, top_k, top_p):
        logits = logits.float() / max(float(temperature), 1e-8)
        logits = torch.clamp(logits, -1e4, 1e4)

        if top_k is not None and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits = torch.where(
                logits < v[:, [-1]],
                torch.full_like(logits, float("-inf")),
                logits,
            )

        if top_p is not None and 0 < float(top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            mask = (cum - probs) > float(top_p)
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_idx, sorted_logits)

        return F.softmax(logits, dim=-1)

    def _sample_next(self, logits, temperature, top_k, top_p):
        if temperature is not None and float(temperature) <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = self._apply_sampling(logits, temperature, top_k, top_p)
        return torch.multinomial(probs, num_samples=1)

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()
        if max_new_tokens <= 0:
            return idx

        seq_len = self.config.seq_len
        if idx.size(1) > seq_len:
            idx = idx[:, -seq_len:]

        # temporal mode no soporta KV-cache
        if (
            use_cache
            and self.config.use_spiking
            and self.config.snn_mode == "temporal"
        ):
            use_cache = False

        if use_cache and idx.size(1) < seq_len:
            kv_caches = [None] * len(self.blocks)
            logits, _, kv_caches = self(idx, kv_caches=kv_caches, start_pos=0)
        else:
            use_cache = False
            kv_caches = None
            logits, _ = self(idx)

        pos = idx.size(1)
        logits = logits[:, -1, :]

        for step in range(max_new_tokens):
            idx_next = self._sample_next(logits, temperature, top_k, top_p)
            idx = torch.cat([idx, idx_next], dim=1)
            if step == max_new_tokens - 1:
                break

            if use_cache and pos + 1 <= seq_len:
                logits, _, kv_caches = self(
                    idx_next, kv_caches=kv_caches, start_pos=pos
                )
                pos += 1
                logits = logits[:, -1, :]
            else:
                use_cache = False
                kv_caches = None
                logits, _ = self(idx[:, -seq_len:])
                logits = logits[:, -1, :]

        return idx
