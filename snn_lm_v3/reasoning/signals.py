"""Extracción de señales internas del modelo.

v4.3:
  - Soporta core_type="attention" y "temporal".
  - `attention_entropy` y `kv_norm` solo tienen sentido con attention; son
    0.0 con temporal (mantenidos por compatibilidad con `ConsistencyVerifier`).
  - Nuevo: `state_norm` para temporal (mean L2 norm del estado final).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import torch
import torch.nn.functional as F

from ..rope import apply_rope


def _unwrap(model):
    seen: set[int] = set()
    while hasattr(model, "_orig_mod") and id(model) not in seen:
        seen.add(id(model))
        model = model._orig_mod
    return model


def _is_rt(model) -> bool:
    model = _unwrap(model)
    if model.__class__.__name__ == "RTModelWrapper":
        return True
    return (
        hasattr(model, "rv")
        and hasattr(model, "snnlm")
        and hasattr(model.rv, "encode")
    )


@dataclass
class InternalSignals:
    mean_spike_rate: float = 0.0
    output_entropy: float = 0.0
    attention_entropy: float = 0.0  # 0.0 si core_type != "attention"
    perplexity: float = 1.0
    max_prob: float = 0.0
    kv_norm: float = 0.0            # 0.0 si core_type != "attention"
    state_norm: float = 0.0         # 0.0 si core_type != "temporal"
    spike_variance: float = 0.0
    n_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InternalSignals":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


class SignalProbe:
    """Corre un forward de inspección y extrae InternalSignals."""

    def __init__(self, model, device=None, enc_name: str = "gpt2"):
        self.raw_model = model
        self.model = _unwrap(model)
        self._is_rt = _is_rt(self.model)
        self._inner = self.model.snnlm if self._is_rt else self.model
        self.device = device or next(self._inner.parameters()).device
        self.enc_name = enc_name
        self._inner.eval()
        self._enc = None
        self._core_type = getattr(
            self._inner.config, "core_type", "attention",
        )

    def _encoding(self):
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding(self.enc_name)
        return self._enc

    @torch.no_grad()
    def probe(self, text: str, enc=None) -> InternalSignals:
        n_blocks = len(self._inner.blocks)
        if self._is_rt:
            ids_list = self.model.rv.encode(text)[: self._inner.config.seq_len]
            if len(ids_list) < 2:
                ids_list = [0, 0]
            idx = torch.tensor([ids_list], dtype=torch.long, device=self.device)
            q_emb, _, _ = self.model.rv.embed_and_quantize(idx, update=False)
            logits, new_states, _, _ = self._inner._forward(
                idx, state=[None] * n_blocks,
                update_memory=False, precomputed_embeddings=q_emb,
            )
        else:
            if enc is None:
                enc = self._encoding()
            ids = enc.encode_ordinary(text)[: self._inner.config.seq_len]
            if len(ids) < 2:
                ids = [enc.eot_token, enc.eot_token]
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            logits, new_states, _, _ = self._inner._forward(
                idx, state=[None] * n_blocks, update_memory=False,
            )

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        output_entropy = -(probs * log_probs).sum(-1).mean().item()
        max_prob = probs.max(-1).values.mean().item()

        if logits.size(1) >= 2:
            shifted_logits = logits[:, :-1].contiguous()
            targets = idx[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.size(-1)),
                targets.reshape(-1),
            )
            perplexity = float(torch.exp(loss.clamp(max=20.0)).item())
        else:
            perplexity = float("inf")

        # Firing rates
        rates = []
        for block in self._inner.blocks:
            lr = getattr(block.ffn, "last_firing_rate", None)
            if lr is not None:
                rates.append(float(lr.item()))
        mean_rate = sum(rates) / len(rates) if rates else 0.0
        variance = (
            sum((r - mean_rate) ** 2 for r in rates) / len(rates) if rates else 0.0
        )

        # Señales dependientes del core
        if self._core_type == "attention":
            kv_norm = self._kv_norm_from_states(new_states)
            attn_entropy = self._attention_entropy(idx)
            state_norm = 0.0
        else:  # temporal
            kv_norm = 0.0
            attn_entropy = 0.0
            state_norm = self._state_norm_from_states(new_states)

        return InternalSignals(
            mean_spike_rate=round(mean_rate, 6),
            output_entropy=round(output_entropy, 4),
            attention_entropy=round(attn_entropy, 4),
            perplexity=round(perplexity, 3),
            max_prob=round(max_prob, 4),
            kv_norm=round(kv_norm, 3),
            state_norm=round(state_norm, 3),
            spike_variance=round(variance, 6),
            n_tokens=int(idx.size(1)),
        )

    # --------------------------------------------------------- signals
    def _kv_norm_from_states(self, new_states) -> float:
        """new_states[0] = (k, v) del bloque 0. Norm media de k."""
        try:
            if not new_states or new_states[0] is None:
                return 0.0
            k, _ = new_states[0]
            return float(k.norm(dim=-1).mean().item())
        except Exception:
            return 0.0

    def _state_norm_from_states(self, new_states) -> float:
        """new_states[i] = (B, inner_dim) del bloque i. Norm media."""
        try:
            if not new_states:
                return 0.0
            norms = []
            for s in new_states:
                if s is None:
                    continue
                norms.append(float(s.norm(dim=-1).mean().item()))
            return sum(norms) / len(norms) if norms else 0.0
        except Exception:
            return 0.0

    def _attention_entropy(self, idx) -> float:
        """Entropía de atención de la capa 0 (solo para core_type="attention")."""
        if self._core_type != "attention":
            return 0.0
        try:
            block = self._inner.blocks[0]
            x = self._inner.drop(self._inner.tok_emb(idx))
            # En v4.3 el bloque expone `.core`; en v4.2.1 era `.attn`.
            attn_mod = getattr(block, "core", None) or getattr(block, "attn", None)
            if attn_mod is None or not hasattr(attn_mod, "qkv"):
                return 0.0
            qkv = attn_mod.qkv(block.ln1(x))
            q, k, _ = qkv.chunk(3, dim=-1)
            B, T, C = q.shape
            H = attn_mod.n_head
            D = attn_mod.head_dim
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            cos = self._inner.rope_cos[:T].unsqueeze(0).unsqueeze(0)
            sin = self._inner.rope_sin[:T].unsqueeze(0).unsqueeze(0)
            q, k = apply_rope(q, k, cos, sin)
            att = (q @ k.transpose(-2, -1)) / (D ** 0.5)
            mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1
            )
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            ent = -(att * att.clamp_min(1e-9).log()).sum(-1).mean().item()
            return float(ent)
        except Exception:
            return 0.0
