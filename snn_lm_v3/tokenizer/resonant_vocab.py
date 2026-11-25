"""Tokenizer Resonante: BPE (via `tokenizers`) + codebook VQ acoplado.

- `encode(text) -> list[int]` (BPE reversible)
- `decode(ids) -> str` (BPE reversible)
- `embed(ids) -> [B,T,D]` (embedding aprendido)
- `quantize(emb) -> (quantized_emb, concept_ids, aux_loss)` (VQ con straight-through)
- `embed_and_quantize(ids, update=True) -> (quantized_emb, concept_ids, aux_loss)`

El codebook se actualiza con EMA vectorizado (sin bucles Python por nodo).
El mapa `concept_to_tokens` (top-K tokens BPE por concepto) permite
decodificación conceptual e introspección.

Todo es determinista. Sin `torch.rand` en updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RVConfig:
    bpe_vocab_size: int = 40_000
    n_concepts: int = 12_092
    concept_dim: int = 132
    embedding_dim: int = 256
    ema_decay: float = 0.99
    commitment_beta: float = 0.25
    min_count_for_update: float = 0.1
    concept_top_k_tokens: int = 8


class ResonantVocab(nn.Module):
    """Tokenizer BPE + codebook VQ. Ver docstring del módulo."""

    def __init__(self, cfg: RVConfig, bpe_tokenizer):
        super().__init__()
        self.cfg = cfg
        self.bpe = bpe_tokenizer  # `tokenizers.Tokenizer`

        # Embedding aprendido (reemplaza al tok_emb clásico)
        self.tok_emb = nn.Embedding(cfg.bpe_vocab_size, cfg.embedding_dim)

        # Proyección al espacio de conceptos
        self.projector = nn.Linear(cfg.embedding_dim, cfg.concept_dim, bias=False)
        # Proyección de vuelta al espacio de embedding
        self.expander = nn.Linear(cfg.concept_dim, cfg.embedding_dim, bias=False)

        # Codebook VQ (buffers, no params; actualizado vía EMA)
        self.register_buffer(
            "codebook", torch.randn(cfg.n_concepts, cfg.concept_dim) * 0.02
        )
        self.register_buffer("ema_counts", torch.zeros(cfg.n_concepts))
        self.register_buffer(
            "ema_sums", torch.zeros(cfg.n_concepts, cfg.concept_dim)
        )
        # Mapa concepto -> top-K tokens BPE más frecuentes
        self.register_buffer(
            "concept_to_tokens",
            torch.full(
                (cfg.n_concepts, cfg.concept_top_k_tokens), -1, dtype=torch.long
            ),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.projector.weight, std=0.02)
        # Expander cercano a la inversa del projector para arrancar cerca de identidad
        with torch.no_grad():
            self.expander.weight.copy_(self.projector.weight.T)

    # ---------------------------------------------------------------- BPE
    def encode(self, text: str) -> list[int]:
        return self.bpe.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.bpe.decode([int(i) for i in ids])

    @property
    def vocab_size(self) -> int:
        return self.cfg.bpe_vocab_size

    # ---------------------------------------------------------------- VQ
    def quantize(
        self, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """emb: [B,T,D] -> (quantized_emb [B,T,D], concept_ids [B,T], aux_loss)."""
        B, T, D = emb.shape
        z = self.projector(emb.float())  # [B,T,C]
        C = z.size(-1)
        z_flat = z.reshape(-1, C)

        # Distancias en chunks (n_concepts grande → cdist O(B*T*K) memoria)
        n = z_flat.size(0)
        chunk = 4096
        idx = torch.empty(n, dtype=torch.long, device=z.device)
        for i in range(0, n, chunk):
            d = torch.cdist(z_flat[i:i + chunk], self.codebook)
            idx[i:i + chunk] = d.argmin(dim=-1)

        quantized_c = self.codebook[idx].reshape(B, T, C)  # [B,T,C]

        # VQ losses (en espacio de concepto)
        commitment = F.mse_loss(z, quantized_c.detach())
        codebook_loss = F.mse_loss(quantized_c, z.detach())
        aux_loss = codebook_loss + self.cfg.commitment_beta * commitment

        # Straight-through
        quantized_c_st = z + (quantized_c - z).detach()
        quantized_emb = self.expander(quantized_c_st)  # [B,T,D]

        return quantized_emb, idx.reshape(B, T), aux_loss

    @torch.no_grad()
    def update_codebook(self, emb: torch.Tensor, concept_ids: torch.Tensor):
        """EMA vectorizado. emb: [B,T,D], concept_ids: [B,T]."""
        z = self.projector(emb.float()).reshape(-1, self.cfg.concept_dim)
        idx = concept_ids.reshape(-1).long()

        one_hot = F.one_hot(idx, self.cfg.n_concepts).float()
        counts = one_hot.sum(dim=0)
        sums = one_hot.T @ z

        d = self.cfg.ema_decay
        self.ema_counts.mul_(d).add_(counts, alpha=1 - d)
        self.ema_sums.mul_(d).add_(sums, alpha=1 - d)

        n_eff = self.ema_counts.clamp(min=1e-6).unsqueeze(-1)
        new_codes = self.ema_sums / n_eff
        mask = (self.ema_counts > self.cfg.min_count_for_update).unsqueeze(-1)
        self.codebook.copy_(torch.where(mask, new_codes, self.codebook))

    @torch.no_grad()
    def update_concept_to_tokens(
        self, ids: torch.Tensor, concept_ids: torch.Tensor
    ):
        """Actualiza mapa concepto -> top-K tokens BPE (solo conceptos del batch)."""
        ids_flat = ids.reshape(-1).long()
        idx_flat = concept_ids.reshape(-1).long()
        K_top = self.cfg.concept_top_k_tokens

        unique_concepts = torch.unique(idx_flat)
        for k in unique_concepts.tolist():
            mask = idx_flat == k
            if not mask.any():
                continue
            tokens, counts = torch.unique(ids_flat[mask], return_counts=True)
            top = tokens[counts.argsort(descending=True)[:K_top]]
            self.concept_to_tokens[k, :top.numel()] = top

    @torch.no_grad()
    def reinit_dead_codes(self, min_count: float = 0.01) -> int:
        """Reinicializa conceptos muertos con perturbaciones de conceptos vivos."""
        dead = (self.ema_counts < min_count).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            return 0
        live = (self.ema_counts >= min_count).nonzero(as_tuple=True)[0]
        if live.numel() == 0:
            self.codebook[dead] = torch.randn_like(self.codebook[dead]) * 0.02
        else:
            src = live[torch.randint(0, live.numel(), (dead.numel(),), device=live.device)]
            noise = torch.randn_like(self.codebook[dead]) * 0.01
            self.codebook[dead] = self.codebook[src] + noise
        self.ema_counts[dead] = 0.0
        self.ema_sums[dead] = 0.0
        return int(dead.numel())

    # ---------------------------------------------------------------- API
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B,T] long -> emb: [B,T,D]."""
        return self.tok_emb(ids)

    def embed_and_quantize(
        self, ids: torch.Tensor, update: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ids: [B,T] -> (quantized_emb [B,T,D], concept_ids [B,T], aux_loss)."""
        emb = self.embed(ids)
        quantized, concept_ids, aux_loss = self.quantize(emb)
        if update and not torch.is_grad_enabled():
            self.update_codebook(emb.detach(), concept_ids)
            self.update_concept_to_tokens(ids.detach(), concept_ids)
        return quantized, concept_ids, aux_loss

    # ---------------------------------------------------------------- load/save
    @classmethod
    def from_bpe(
        cls, bpe_path: str, cfg: Optional[RVConfig] = None
    ) -> "ResonantVocab":
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(bpe_path)
        if cfg is None:
            cfg = RVConfig(bpe_vocab_size=tokenizer.get_vocab_size())
        else:
            cfg.bpe_vocab_size = tokenizer.get_vocab_size()
        return cls(cfg, tokenizer)

    def state_for_checkpoint(self) -> dict:
        return {
            "config": dict(self.cfg.__dict__),
            "state_dict": {k: v.cpu().clone() for k, v in self.state_dict().items()},
        }

    def load_state_from_checkpoint(self, data: dict, device):
        if "state_dict" in data:
            self.load_state_dict(
                {k: v.to(device) for k, v in data["state_dict"].items()},
                strict=False,
            )

    def extra_repr(self) -> str:
        used = int((self.ema_counts > self.cfg.min_count_for_update).sum().item())
        return (
            f"bpe_vocab={self.cfg.bpe_vocab_size}, "
            f"n_concepts={self.cfg.n_concepts} (used={used}), "
            f"concept_dim={self.cfg.concept_dim}"
        )
