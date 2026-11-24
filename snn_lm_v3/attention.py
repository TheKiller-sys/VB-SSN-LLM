import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.seq_len = config.seq_len

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.triu(
            torch.ones(config.seq_len, config.seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos_t = cos[:T].unsqueeze(0).unsqueeze(0)
        sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos_t, sin_t)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self.causal_mask[:T, :T]
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
