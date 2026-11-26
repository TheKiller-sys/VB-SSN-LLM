"""Tokenizer Resonante: BPE + codebook VQ acoplado.

- `encode/decode`: BPE reversible.
- `embed_and_quantize`: BPE embed → VQ → straight-through → expander.
- `update_codebook`: EMA vectorizado (O(K) memoria, sin loops por nodo).
- `update_concept_to_tokens`: top-K tokens por concepto (O(N log N), sin loops
  O(K·N)).
- `reinit_dead_codes`: no determinista (usa randint/randn); solo lo llama el
  `ContinuousAnalyzer`, nunca el training loop.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
        self.bpe = bpe_tokenizer

        self.tok_emb = nn.Embedding(cfg.bpe_vocab_size, cfg.embedding_dim)
        self.projector = nn.Linear(cfg.embedding_dim, cfg.concept_dim, bias=False)
        self.expander = nn.Linear(cfg.concept_dim, cfg.embedding_dim, bias=False)

        self.register_buffer(
            "codebook", torch.randn(cfg.n_concepts, cfg.concept_dim) * 0.02
        )
        self.register_buffer("ema_counts", torch.zeros(cfg.n_concepts))
        self.register_buffer(
            "ema_sums", torch.zeros(cfg.n_concepts, cfg.concept_dim)
        )
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
        z = self.projector(emb.float())
        C = z.size(-1)
        z_flat = z.reshape(-1, C)

        n = z_flat.size(0)
        chunk = 4096
        idx = torch.empty(n, dtype=torch.long, device=z.device)
        for i in range(0, n, chunk):
            d = torch.cdist(z_flat[i:i + chunk], self.codebook)
            idx[i:i + chunk] = d.argmin(dim=-1)

        quantized_c = self.codebook[idx].reshape(B, T, C)

        commitment = F.mse_loss(z, quantized_c.detach())
        codebook_loss = F.mse_loss(quantized_c, z.detach())
        aux_loss = codebook_loss + self.cfg.commitment_beta * commitment

        quantized_c_st = z + (quantized_c - z).detach()
        quantized_emb = self.expander(quantized_c_st)

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
        """Actualiza mapa concepto -> top-K tokens. O(N log N) sin loops O(K·N).

        Estrategia:
          1. Ordenar (concepto, token) por concepto.
          2. Iterar solo por los conceptos únicos (K << N típicamente),
             procesando cada grupo sin boolean masking repetido.
        """
        ids_flat = ids.reshape(-1).long()
        idx_flat = concept_ids.reshape(-1).long()
        K_top = self.cfg.concept_top_k_tokens

        if ids_flat.numel() == 0:
            return

        # Ordenar por concepto (estable) para agrupar
        order = idx_flat.argsort(stable=True)
        sorted_concepts = idx_flat[order]
        sorted_tokens = ids_flat[order]

        # Fronteras de grupos
        unique_concepts, counts = torch.unique_consecutive(
            sorted_concepts, return_counts=True
        )
        ends = counts.cumsum(0).tolist()
        starts = [0] + ends[:-1]

        for k, start, end in zip(unique_concepts.tolist(), starts, ends):
            toks = sorted_tokens[start:end]
            if toks.numel() == 0:
                continue
            tokens, tok_counts = torch.unique(toks, return_counts=True)
            k_top = min(K_top, tokens.numel())
            top_idx = tok_counts.argsort(descending=True)[:k_top]
            self.concept_to_tokens[k, :k_top] = tokens[top_idx]
            if k_top < K_top:
                self.concept_to_tokens[k, k_top:] = -1

    @torch.no_grad()
    def reinit_dead_codes(self, min_count: float = 0.01) -> int:
        """Reinicializa conceptos muertos con perturbaciones de vivos.

        No determinista (usa randint/randn). Solo lo llama el analyzer.
        """
        dead = (self.ema_counts < min_count).nonzero(as_tuple=True)[0]
        if dead.numel() == 0:
            return 0
        live = (self.ema_counts >= min_count).nonzero(as_tuple=True)[0]
        if live.numel() == 0:
            self.codebook[dead] = torch.randn_like(self.codebook[dead]) * 0.02
        else:
            src = live[
                torch.randint(0, live.numel(), (dead.numel(),), device=live.device)
            ]
            noise = torch.randn_like(self.codebook[dead]) * 0.01
            self.codebook[dead] = self.codebook[src] + noise
        self.ema_counts[dead] = 0.0
        self.ema_sums[dead] = 0.0
        return int(dead.numel())

    # ---------------------------------------------------------------- API
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(ids)

    def embed_and_quantize(
        self, ids: torch.Tensor, update: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        actual_vocab = tokenizer.get_vocab_size()

        if cfg is None:
            cfg = RVConfig(bpe_vocab_size=actual_vocab)
        elif cfg.bpe_vocab_size != actual_vocab:
            # No mutar el cfg del caller
            cfg = replace(cfg, bpe_vocab_size=actual_vocab)

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
