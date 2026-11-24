"""Bloque Transformer pre-LN con FFN de spikes."""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .snn import SpikingFFN


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = SpikingFFN(
            config.n_embd,
            mult=config.spiking_mult,
            threshold=config.spiking_threshold,
            alpha=config.spiking_alpha,
            dropout=config.dropout,
            use_spiking=config.use_spiking,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        start_pos: int = 0,
    ):
        attn_out, new_kv = self.attn(self.ln1(x), cos, sin, kv_cache, start_pos)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv
