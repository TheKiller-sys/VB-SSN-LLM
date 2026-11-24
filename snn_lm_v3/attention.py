"""Atención causal multi-cabeza con RoPE y KV-cache."""
from __future__ import annotations

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

    def forward(self, x, cos, sin, kv_cache=None, start_pos: int = 0):
        """x: (B, T, C). Devuelve (y, new_kv_cache)."""
        if kv_cache is None:
            start_pos = 0

        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if start_pos + T > cos.size(0):
            raise ValueError(
                f"RoPE out of range: start_pos={start_pos}, T={T}, max={cos.size(0)}"
            )
        cos_t = cos[start_pos : start_pos + T].unsqueeze(0).unsqueeze(0)
        sin_t = sin[start_pos : start_pos + T].unsqueeze(0).unsqueeze(0)
        q, k = apply_rope(q, k, cos_t, sin_t)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            if k_past is not None:
                k = torch.cat([k_past, k], dim=2)
                v = torch.cat([v_past, v], dim=2)
            new_kv = (k, v)
        else:
            new_kv = None

        T_k = k.size(2)
        dropout_p = self.dropout.p if self.training else 0.0

        if start_pos == 0 and T == T_k:
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True
            )
        elif T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device).unsqueeze(1)
            k_pos = torch.arange(0, T_k, device=x.device).unsqueeze(0)
            attend = k_pos <= q_pos
            attn_mask = torch.zeros(T, T_k, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~attend, float("-inf"))
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_kv
