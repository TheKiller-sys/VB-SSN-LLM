"""SNN-LM v4.0: transformer estilo GPT-2 con FFN de spikes, memoria neural
(kNN + fast weights), abstracción geométrica (VQ) y autoconocimiento."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .rope import precompute_rope
from .memory import (
    NeuralMemory,
    AbstractionHead,
    SelfKnowledgeHead,
    LatentReasoner,
)


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

        # ---- memoria neural (v4.0) ----
        self.neural_memory: Optional[NeuralMemory] = None
        if config.neural_memory_enabled:
            self.neural_memory = NeuralMemory(
                dim=config.n_embd,
                max_size=config.neural_memory_max_size,
                k=config.neural_memory_k,
                hebbian_lr=config.neural_memory_hebbian_lr,
                hebbian_decay=config.neural_memory_hebbian_decay,
                norm_clamp=config.neural_memory_norm_clamp,
                use_fast_weights=config.neural_memory_use_fast_weights,
                use_knn=config.neural_memory_use_knn,
                append_batch_size=config.neural_memory_append_batch_size,
            )

        # ---- abstracción geométrica ----
        self.abstraction_head: Optional[AbstractionHead] = None
        if config.abstraction_enabled:
            self.abstraction_head = AbstractionHead(
                dim=config.n_embd,
                n_codes=config.abstraction_n_codes,
                ema_decay=config.abstraction_ema_decay,
                min_count_for_update=config.abstraction_min_count_for_update,
            )

        # ---- autoconocimiento ----
        self.self_knowledge_head: Optional[SelfKnowledgeHead] = None
        if config.self_knowledge_enabled:
            hidden = config.self_knowledge_hidden or None
            self.self_knowledge_head = SelfKnowledgeHead(
                dim=config.n_embd, hidden=hidden,
            )

        # ---- razonador latente ----
        self.latent_reasoner: Optional[LatentReasoner] = None
        if config.latent_reasoner_enabled:
            self.latent_reasoner = LatentReasoner(
                dim=config.n_embd,
                n_heads=config.latent_reasoner_heads,
                n_layers=config.latent_reasoner_layers,
            )

        # Último estado de auto-conocimiento (accesible sin cambiar signature)
        self._last_self_knowledge: Optional[torch.Tensor] = None

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
    def _forward(self, idx, kv_caches=None, start_pos: int = 0,
                 update_memory: Optional[bool] = None):
        x = self.drop(self.tok_emb(idx))
        new_kv_caches = [] if kv_caches is not None else None
        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(x, self.rope_cos, self.rope_sin, kv, start_pos)
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)

        # ---- memoria neural ----
        if self.neural_memory is not None:
            x = self.neural_memory(x, update=update_memory)

        # ---- razonador latente (opcional) ----
        if self.latent_reasoner is not None and self.neural_memory is not None:
            mem_keys = self.neural_memory.get_keys()
            x = self.latent_reasoner(x, mem_keys)

        # ---- abstracción geométrica ----
        vq_loss = None
        if self.abstraction_head is not None:
            quantized, _ = self.abstraction_head(x, update=update_memory)
            x = x + self.config.abstraction_residual_scale * quantized
            if self.training:
                z = self.abstraction_head.proj(x)
                vq_loss = F.mse_loss(z, quantized.detach())

        hidden = self.ln_f(x)
        logits = self.head(hidden)
        return logits, new_kv_caches, vq_loss, hidden

    def forward(
        self,
        idx,
        targets=None,
        kv_caches=None,
        start_pos: int = 0,
        update_memory: Optional[bool] = None,
    ):
        logits, new_kv_caches, vq_loss, hidden = self._forward(
            idx, kv_caches, start_pos, update_memory,
        )

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            if vq_loss is not None and self.config.abstraction_loss_weight > 0:
                loss = loss + self.config.abstraction_loss_weight * vq_loss

            # Self-knowledge loss (usa `hidden` que acabamos de calcular)
            if (
                self.self_knowledge_head is not None
                and self.config.self_knowledge_loss_weight > 0
            ):
                concept_emb = self.tok_emb(idx)
                sk = self.self_knowledge_head(hidden, concept_emb)
                self._last_self_knowledge = sk.detach()
                sk_loss = SelfKnowledgeHead.compute_loss(sk, logits, targets)
                loss = loss + self.config.self_knowledge_loss_weight * sk_loss

        if kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss

    def get_last_self_knowledge(self) -> Optional[torch.Tensor]:
        return self._last_self_knowledge

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
        update_memory: bool = True,
    ) -> torch.Tensor:
        self.eval()
        if max_new_tokens <= 0:
            return idx

        seq_len = self.config.seq_len
        if idx.size(1) > seq_len:
            idx = idx[:, -seq_len:]

        if (
            use_cache
            and self.config.use_spiking
            and self.config.snn_mode == "temporal"
        ):
            use_cache = False

        if use_cache and idx.size(1) < seq_len:
            kv_caches = [None] * len(self.blocks)
            logits, _, _, _ = self(
                idx, kv_caches=kv_caches, start_pos=0,
                update_memory=update_memory,
            )
        else:
            use_cache = False
            kv_caches = None
            logits, _ = self(idx, update_memory=update_memory)

        pos = idx.size(1)
        logits = logits[:, -1, :]

        for step in range(max_new_tokens):
            idx_next = self._sample_next(logits, temperature, top_k, top_p)
            idx = torch.cat([idx, idx_next], dim=1)
            if step == max_new_tokens - 1:
                break

            if use_cache and pos + 1 <= seq_len:
                logits, _, _, _ = self(
                    idx_next, kv_caches=kv_caches, start_pos=pos,
                    update_memory=update_memory,
                )
                pos += 1
                logits = logits[:, -1, :]
            else:
                use_cache = False
                kv_caches = None
                logits, _ = self(
                    idx[:, -seq_len:], update_memory=update_memory,
                )
                logits = logits[:, -1, :]

        return idx

    # ------------------------------------------------------------- utilities
    def consolidate(self) -> dict:
        """Consolidación manual (llamada por ContinuousAnalyzer)."""
        result: dict = {}
        if self.neural_memory is not None:
            result["memory_size"] = int(self.neural_memory.size.item())
        if self.abstraction_head is not None:
            result["dead_codes"] = self.abstraction_head.reinit_dead_codes()
        return result

    def reset_memory(self) -> None:
        if self.neural_memory is not None:
            self.neural_memory.reset()
        if self.abstraction_head is not None:
            self.abstraction_head.reset()

    def memory_state_for_checkpoint(self) -> dict:
        """Estado de memoria para guardar en checkpoint."""
        out: dict = {}
        if self.neural_memory is not None:
            out["neural_memory"] = self.neural_memory.state_for_checkpoint()
        if self.abstraction_head is not None:
            out["abstraction"] = {
                "codebook": self.abstraction_head.codebook.cpu().clone(),
                "ema_counts": self.abstraction_head.ema_counts.cpu().clone(),
                "ema_sums": self.abstraction_head.ema_sums.cpu().clone(),
            }
        return out

    def load_memory_state_from_checkpoint(self, data: dict, device) -> None:
        if not data:
            return
        if self.neural_memory is not None and "neural_memory" in data:
            self.neural_memory.load_state_from_checkpoint(
                data["neural_memory"], device,
            )
        if self.abstraction_head is not None and "abstraction" in data:
            abs_data = data["abstraction"]
            if "codebook" in abs_data:
                self.abstraction_head.codebook.copy_(
                    abs_data["codebook"].to(device),
                )
            if "ema_counts" in abs_data:
                self.abstraction_head.ema_counts.copy_(
                    abs_data["ema_counts"].to(device),
                )
            if "ema_sums" in abs_data:
                self.abstraction_head.ema_sums.copy_(
                    abs_data["ema_sums"].to(device),
                )

    def get_knowledge_summary(self) -> dict:
        """Resumen del autoconocimiento del modelo."""
        if self.self_knowledge_head is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "memory_size": int(self.neural_memory.size.item())
            if self.neural_memory is not None else 0,
            "codebook_used": int(
                (self.abstraction_head.ema_counts > 0.01).sum().item()
            ) if self.abstraction_head is not None else 0,
            "codebook_total": self.abstraction_head.n_codes
            if self.abstraction_head is not None else 0,
        }
