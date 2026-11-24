import torch.nn as nn

from .attention import MultiHeadAttention
from .snn import SpikingFFN


class TransformerBlock(nn.Module):
    """Pre-LN transformer con FFN de spikes y residual connections."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = SpikingFFN(
            config.n_embd,
            mult=4,
            threshold=config.spiking_threshold,
            alpha=config.spiking_alpha,
            dropout=config.dropout,
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x
