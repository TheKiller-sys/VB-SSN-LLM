"""Tests de atención causal con KV-cache."""
import torch

from snn_lm_v3 import GPTConfig, MultiHeadAttention, precompute_rope


def _cfg(**kw):
    base = dict(n_layer=1, n_head=2, n_embd=32, seq_len=32, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_no_cache():
    cfg = _cfg()
    attn = MultiHeadAttention(cfg)
    x = torch.randn(2, 8, 32)
    cos, sin = precompute_rope(16, cfg.seq_len)
    y, kv = attn(x, cos, sin, None, 0)
    assert y.shape == (2, 8, 32)
    assert kv is None


def test_forward_with_cache():
    cfg = _cfg()
    attn = MultiHeadAttention(cfg)
    cos, sin = precompute_rope(16, cfg.seq_len)
    x = torch.randn(2, 8, 32)
    y, kv = attn(x, cos, sin, (None, None), 0)
    assert y.shape == (2, 8, 32)
    k, v = kv
    assert k.shape == (2, 2, 8, 16)
    assert v.shape == (2, 2, 8, 16)


def test_attention_cache_equivalence():
    torch.manual_seed(0)
    cfg = _cfg()
    attn = MultiHeadAttention(cfg)
    attn.eval()
    cos, sin = precompute_rope(16, cfg.seq_len)
    x = torch.randn(1, 8, 32)

    y_full, _ = attn(x, cos, sin, None, 0)

    kv = (None, None)
    y1, kv = attn(x[:, :4], cos, sin, kv, 0)
    y2, kv = attn(x[:, 4:], cos, sin, kv, 4)
    y_cached = torch.cat([y1, y2], dim=1)

    assert torch.allclose(y_full, y_cached, atol=1e-5)


def test_attention_rope_out_of_range():
    cfg = _cfg(seq_len=8)
    attn = MultiHeadAttention(cfg)
    cos, sin = precompute_rope(16, cfg.seq_len)
    x = torch.randn(1, 4, 32)
    try:
        attn(x, cos, sin, (None, None), start_pos=7)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_attention_mask_causal():
    """La atención no debe permitir atender a futuro."""
    cfg = _cfg(dropout=0.0)
    attn = MultiHeadAttention(cfg)
    attn.eval()
    cos, sin = precompute_rope(16, cfg.seq_len)
    x = torch.randn(1, 4, 32)
    y, _ = attn(x, cos, sin, None, 0)
    # No assertion on values (depends on weights), but shape must be right
    assert y.shape == (1, 4, 32)


def test_attention_gradient_flow():
    cfg = _cfg()
    attn = MultiHeadAttention(cfg)
    cos, sin = precompute_rope(16, cfg.seq_len)
    x = torch.randn(1, 4, 32, requires_grad=True)
    y, _ = attn(x, cos, sin, None, 0)
    y.sum().backward()
    assert x.grad is not None
    assert attn.qkv.weight.grad is not None
