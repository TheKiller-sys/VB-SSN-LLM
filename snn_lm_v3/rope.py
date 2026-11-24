import torch


def precompute_rope(dim, max_len=8192, base=10000.0, device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (max_len, dim//2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, H, T, D). cos, sin: (1, 1, T, D)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out
