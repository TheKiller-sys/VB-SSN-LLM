"""Bloque Transformer con core configurable (attention|temporal) y FFN
configurable (dense|spiking|hybrid).

v4.3: `core_type="temporal"` usa TemporalCore (SSM con LIF decay).
       `ffn_mode="hybrid"` usa SpikingFFN híbrido.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .snn import SpikingFFN
from .temporal_core import TemporalCore


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core_type = config.core_type

        self.ln1 = nn.LayerNorm(config.n_embd)
        if config.core_type == "attention":
            self.core = MultiHeadAttention(config)
        elif config.core_type == "temporal":
            self.core = TemporalCore(config)
        else:
            raise ValueError(f"invalid core_type: {config.core_type}")

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = SpikingFFN(
            config.n_embd,
            mult=config.spiking_mult,
            threshold=config.spiking_threshold,
            alpha=config.spiking_alpha,
            dropout=config.dropout,
            use_spiking=config.use_spiking,
            residual=config.spiking_residual,
            mode=config.snn_mode,
            decay=config.snn_decay,
            reset_mode=config.snn_reset_mode,
            num_steps=config.snn_num_steps,
            learnable_threshold=config.snn_learnable_threshold,
            learnable_decay=config.snn_learnable_decay,
            track_stats=config.snn_track_stats,
            ffn_mode=config.ffn_mode,
            hybrid_gate_init=config.ffn_hybrid_spike_gate_init,
        )

    def forward(
        self,
        x: torch.Tensor,
        state=None,
        cos=None,
        sin=None,
        start_pos: int = 0,
    ):
        """Devuelve (x_out, new_state).

        - core_type="temporal": `state` es (B, inner_dim) o None; `cos`/`sin`/`start_pos`
          son ignorados.
        - core_type="attention": `state` es (k, v) o None; usa `cos`/`sin`/`start_pos`.
        """
        if self.core_type == "temporal":
            core_out, new_state = self.core(self.ln1(x), state)
        else:
            core_out, new_state = self.core(
                self.ln1(x), cos, sin, state, start_pos,
            )
        x = x + core_out
        x = x + self.ffn(self.ln2(x))
        return x, new_state
