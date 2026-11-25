"""Extracción de señales internas del modelo SNN-LM.

Las señales son el equivalente LM de `neuralState` del sistema cognitivo JS:
  - mean_spike_rate    ~ actividad del sustrato
  - output_entropy     ~ incertidumbre
  - attention_entropy  ~ foco
  - perplexity         ~ calidad
  - max_prob           ~ confianza
  - kv_norm            ~ carga de contexto
  - spike_variance     ~ coherencia entre capas
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F

from ..rope import apply_rope


@dataclass
class InternalSignals:
    mean_spike_rate: float = 0.0
    output_entropy: float = 0.0
    attention_entropy: float = 0.0
    perplexity: float = 1.0
    max_prob: float = 0.0
    kv_norm: float = 0.0
    spike_variance: float = 0.0
    n_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InternalSignals":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in fields})


class SignalProbe:
    """Corre un forward de inspección y extrae InternalSignals.

    Uso:
        probe = SignalProbe(model)
        sig = probe.probe("The quick brown fox")
        # sig.mean_spike_rate, sig.output_entropy, ...

    Nota: para que `mean_spike_rate` sea significativo, el modelo debe haberse
    construido con `snn_track_stats=True` en su GPTConfig.
    """

    def __init__(self, model, device=None, enc_name: str = "gpt2"):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.enc_name = enc_name
        self.model.eval()
        self._enc = None

    def _encoding(self):
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding(self.enc_name)
        return self._enc

    @torch.no_grad()
    def probe(self, text: str, enc=None) -> InternalSignals:
        if enc is None:
            enc = self._encoding()
        ids = enc.encode_ordinary(text)[: self.model.config.seq_len]
        if len(ids) < 2:
            ids = [enc.eot_token, enc.eot_token]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Un único forward: suficiente para entropy, max_prob y perplexity
        # shifted. Además, NO contamina la memoria del modelo.
        logits, _ = self.model(idx, update_memory=False)

        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        output_entropy = -(probs * log_probs).sum(-1).mean().item()
        max_prob = probs.max(-1).values.mean().item()

        # Perplejidad shifted sobre el mismo forward
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
        for block in self.model.blocks:
            lr = getattr(block.ffn, "last_firing_rate", None)
            if lr is not None:
                rates.append(float(lr.item()))
        mean_rate = sum(rates) / len(rates) if rates else 0.0
        variance = (
            sum((r - mean_rate) ** 2 for r in rates) / len(rates) if rates else 0.0
        )

        kv_norm = self._kv_norm(idx)
        attn_entropy = self._attention_entropy(idx)

        return InternalSignals(
            mean_spike_rate=round(mean_rate, 6),
            output_entropy=round(output_entropy, 4),
            attention_entropy=round(attn_entropy, 4),
            perplexity=round(perplexity, 3),
            max_prob=round(max_prob, 4),
            kv_norm=round(kv_norm, 3),
            spike_variance=round(variance, 6),
            n_tokens=len(ids),
        )

    def _kv_norm(self, idx) -> float:
        try:
            block = self.model.blocks[0]
            x = self.model.drop(self.model.tok_emb(idx))
            qkv = block.attn.qkv(block.ln1(x))
            _, k, _ = qkv.chunk(3, dim=-1)
            return float(k.norm(dim=-1).mean().item())
        except Exception:
            return 0.0

    def _attention_entropy(self, idx) -> float:
        """Entropía de atención de la capa 0 recomputada con RoPE aplicado,
        de modo que coincida con la atención real del modelo."""
        try:
            block = self.model.blocks[0]
            x = self.model.drop(self.model.tok_emb(idx))
            qkv = block.attn.qkv(block.ln1(x))
            q, k, _ = qkv.chunk(3, dim=-1)
            B, T, C = q.shape
            H = block.attn.n_head
            D = block.attn.head_dim
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            # RoPE igual que en MultiHeadAttention
            cos = self.model.rope_cos[:T].unsqueeze(0).unsqueeze(0)
            sin = self.model.rope_sin[:T].unsqueeze(0).unsqueeze(0)
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
