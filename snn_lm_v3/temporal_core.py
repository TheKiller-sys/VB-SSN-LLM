"""TemporalCore: SSM selectivo con decaimiento modulado por LIF.

Reemplaza MultiHeadAttention cuando `GPTConfig.core_type == "temporal"`.

Dinámica por paso:
    h_t = d_t * h_{t-1} + x_t
    d_t = base + spike_t * (target - base)
    spike_t = LIF(x_t)  (binario, STE)

- Estado: (B, inner_dim). O(1) memoria, sin KV-cache.
- Scan chunked-parallel con cumprod/cumsum para estabilidad numérica.
- `state=None` -> no track state (training).
- `state=List[Tensor]` -> track state (generation); la lista tiene un tensor
  por bloque, shape (B, inner_dim), o None en el primer paso.

Diseño validado contra:
  - Mamba (Gu & Dao, 2023): linear scan parallelizable.
  - RWKV v6: recurrence con gate.
  - S5 (Smith et al., 2023): parallel scan stability.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .snn import LIFNeuron, spike_fn


class TemporalCore(nn.Module):
    """SSM selectivo con decaimiento LIF-driven."""

    def __init__(self, config):
        super().__init__()
        if config.n_embd <= 0:
            raise ValueError("n_embd must be > 0")
        if config.temporal_expand < 1:
            raise ValueError("temporal_expand must be >= 1")

        self.dim = config.n_embd
        self.inner_dim = config.n_embd * config.temporal_expand
        self.chunk_size = max(1, int(config.temporal_chunk_size))
        self.use_spike_decay = bool(config.temporal_use_spike_decay)

        self.norm = nn.LayerNorm(config.n_embd)
        # in_proj: (dim) -> (2 * inner_dim) = x_inner || gate
        self.in_proj = nn.Linear(config.n_embd, 2 * self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # LIF para la modulación del decaimiento (binario, STE)
        self.decay_lif = LIFNeuron(
            threshold=config.temporal_lif_threshold,
            decay=config.temporal_lif_decay,
            alpha=config.temporal_lif_alpha,
            reset_mode="soft",
            learnable_threshold=False,
            learnable_decay=False,
        )

        # Decaimiento base (continuo, aprendible, en logit-space)
        base_init = config.temporal_decay_base
        base_init = min(max(base_init, 1e-3), 1.0 - 1e-3)
        base_logit = math.log(base_init / (1.0 - base_init))
        self.base_decay_logit = nn.Parameter(torch.full((self.inner_dim,), base_logit))

        # Target de decaimiento cuando spike=1 (aprendible, en logit-space)
        tgt_init = config.temporal_decay_target
        tgt_init = min(max(tgt_init, 1e-3), 1.0 - 1e-3)
        tgt_logit = math.log(tgt_init / (1.0 - tgt_init))
        self.target_decay_logit = nn.Parameter(torch.full((self.inner_dim,), tgt_logit))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    # --------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (B, T, C). state: (B, inner_dim) or None.

        Returns:
            (out, new_state) donde new_state es (B, inner_dim) si state!=None,
            o None si state==None (training path).
        """
        B, T, C = x.shape
        if C != self.dim:
            raise ValueError(f"TemporalCore expected dim={self.dim}, got {C}")

        h = self.norm(x)
        proj = self.in_proj(h)                # (B, T, 2*inner)
        x_inner, gate = proj.chunk(2, dim=-1)  # (B, T, inner) each

        if self.use_spike_decay:
            # LIF produce spike binario {0,1} con STE
            spike = self.decay_lif(x_inner, num_steps=1)  # (B, T, inner)
        else:
            spike = torch.zeros_like(x_inner)

        # decay_t = base + spike * (target - base)
        base = torch.sigmoid(self.base_decay_logit)    # (inner,)
        target = torch.sigmoid(self.target_decay_logit)  # (inner,)
        decay = base.unsqueeze(0).unsqueeze(0) + spike * (
            target - base
        ).unsqueeze(0).unsqueeze(0)                    # (B, T, inner)
        decay = decay.clamp(0.0, 1.0 - 1e-4)

        track_state = state is not None
        if track_state and state.dim() != 2:
            raise ValueError(
                f"state must be (B, inner_dim), got shape {tuple(state.shape)}"
            )

        y, new_state = self._chunked_scan(
            x_inner, decay, state if track_state else None,
        )

        y = y * torch.sigmoid(gate)
        out = self.out_proj(self.dropout(y))

        return out, (new_state if track_state else None)

    # --------------------------------------------------------- scan
    def _chunked_scan(
        self,
        x: torch.Tensor,
        decay: torch.Tensor,
        state0: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Scan paralelo por chunks. x,decay: (B, T, D). state0: (B, D) or None.

        Dinámica: h_t = d_t * h_{t-1} + x_t.
        Dentro de cada chunk, se usa la formulación asociativa:
            A_t = prod_{s<=t} d_s
            B_t = sum_{s<=t} (x_s / A_s) * A_t
            h_t = A_t * h_0 + B_t
        con clamp para evitar division-by-zero.
        """
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        if state0 is None:
            state0 = torch.zeros(B, D, device=device, dtype=dtype)

        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        outs: list[torch.Tensor] = []
        state = state0

        eps = 1e-8
        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, T)
            x_c = x[:, start:end]         # (B, L, D)
            d_c = decay[:, start:end]     # (B, L, D)
            L = x_c.size(1)

            # Cumprod estable: A_t = cumprod(d_1..d_t). d_t puede ser ~1.
            # Para evitar underflow en L grande, chunk_size <= 64 y d>=0.
            A = torch.cumprod(d_c, dim=1)          # (B, L, D)
            A_safe = A.clamp(min=eps)

            # B_t = A_t * cumsum(x_t / A_t)
            contrib = x_c / A_safe                  # (B, L, D)
            B_t = A * torch.cumsum(contrib, dim=1)  # (B, L, D)

            # h_t = A_t * state_in + B_t
            chunk_out = A * state.unsqueeze(1) + B_t  # (B, L, D)
            outs.append(chunk_out)
            state = chunk_out[:, -1]                # (B, D)

        return torch.cat(outs, dim=1), state

    # --------------------------------------------------------- utils
    def reset_state(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.inner_dim, device=device, dtype=dtype,
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, inner_dim={self.inner_dim}, "
            f"expand={self.inner_dim // self.dim}, "
            f"chunk={self.chunk_size}, spike_decay={self.use_spike_decay}"
        )
