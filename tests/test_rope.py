"""Tests de RoPE."""
import pytest
import torch

from snn_lm_v3.rope import precompute_rope, apply_rope, rotate_half


def test_rope_shapes():
    cos, sin = precompute_rope(64, max_len=128)
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)


def test_rope_dtype():
    cos, sin = precompute_rope(64, max_len=8)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_rotate_half():
    x = torch.arange(4, dtype=torch.float).view(1, 1, 1, 4)
    out = rotate_half(x)
    expected = torch.tensor([-2.0, -3.0, 0.0, 1.0]).view(1, 1, 1, 4)
    assert torch.allclose(out, expected)


def test_apply_rope_shape():
    cos, sin = precompute_rope(64, max_len=128)
    q = torch.randn(2, 4, 16, 64)
    k = torch.randn(2, 4, 16, 64)
    c = cos[:16].unsqueeze(0).unsqueeze(0)
    s = sin[:16].unsqueeze(0).unsqueeze(0)
    q2, k2 = apply_rope(q, k, c, s)
    assert q2.shape == q.shape
    assert k2.shape == k.shape


def test_rope_identity_at_pos0():
    cos, sin = precompute_rope(64, max_len=4)
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)
    c = cos[0:1].unsqueeze(0).unsqueeze(0)
    s = sin[0:1].unsqueeze(0).unsqueeze(0)
    q2, _ = apply_rope(q, k, c, s)
    assert torch.allclose(q2, q, atol=1e-6)


def test_rope_odd_dim_raises():
    with pytest.raises(ValueError, match="even dim"):
        precompute_rope(63, max_len=16)


def test_rope_norm_preserved():
    """RoPE es una rotación: preserva la norma de cada vector."""
    cos, sin = precompute_rope(32, max_len=8)
    q = torch.randn(1, 1, 8, 32)
    k = torch.randn(1, 1, 8, 32)
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    q2, k2 = apply_rope(q, k, c, s)
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k2.norm(dim=-1), atol=1e-5)
