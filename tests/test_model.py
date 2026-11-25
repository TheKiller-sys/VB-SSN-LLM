"""Tests del modelo completo: forward, loss, generate, modos SNN."""
import pytest
import torch

from snn_lm_v3 import GPTConfig, SNNLM


def _cfg(**kw):
    base = dict(
        n_layer=2, n_head=2, n_embd=32, seq_len=16, vocab_size=100, dropout=0.0
    )
    base.update(kw)
    return GPTConfig(**base)


def test_forward_shapes():
    model = SNNLM(_cfg())
    x = torch.randint(0, 100, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, 100)
    assert loss is None


def test_forward_loss():
    model = SNNLM(_cfg())
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 100)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_num_params():
    model = SNNLM(_cfg())
    n = model.num_params()
    assert n > 0
    assert n == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_generate_shape():
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 5))
    out = model.generate(idx, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 15)


def test_generate_zero_tokens():
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 5))
    out = model.generate(idx, max_new_tokens=0)
    assert out.shape == (1, 5)


def test_generate_greedy():
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 4))
    out = model.generate(idx, 3, temperature=0.0)
    assert out.shape == (1, 7)


def test_generate_cache_equivalence():
    """Cache y no-cache DEBEN producir la misma secuencia.

    Se desactiva `update_memory` para que el estado de memoria no varíe
    entre las dos llamadas y enmascare la diferencia de cache.
    """
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 5))

    torch.manual_seed(123)
    out_cached = model.generate(
        idx.clone(), 10, temperature=1.0, use_cache=True, update_memory=False,
    )

    torch.manual_seed(123)
    out_no_cache = model.generate(
        idx.clone(), 10, temperature=1.0, use_cache=False, update_memory=False,
    )

    assert torch.equal(out_cached, out_no_cache)


def test_top_p_extreme_does_not_crash():
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 5))
    out = model.generate(idx, 5, temperature=1.0, top_p=1e-6)
    assert out.shape == (1, 10)


def test_top_k_extreme_does_not_crash():
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 5))
    out = model.generate(idx, 5, temperature=1.0, top_k=1)
    assert out.shape == (1, 10)


def test_generate_exceeds_seq_len():
    model = SNNLM(_cfg(seq_len=8))
    model.eval()
    idx = torch.randint(0, 100, (1, 4))
    out = model.generate(idx, 10, temperature=1.0, use_cache=True)
    assert out.shape == (1, 14)


def test_spiking_toggle():
    model = SNNLM(_cfg(use_spiking=False))
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    _, loss = model(x, y)
    assert loss.item() > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_temporal_mode_forward():
    model = SNNLM(_cfg(snn_mode="temporal", snn_num_steps=1))
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    _, loss = model(x, y)
    assert loss.item() > 0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_temporal_mode_generate_disables_cache():
    model = SNNLM(_cfg(snn_mode="temporal", snn_num_steps=1))
    model.eval()
    idx = torch.randint(0, 100, (1, 5))
    out = model.generate(idx, 4, temperature=1.0, use_cache=True)
    assert out.shape == (1, 9)


def test_static_mode_forward():
    model = SNNLM(_cfg(snn_mode="static"))
    x = torch.randint(0, 100, (2, 16))
    _, loss = model(x, torch.randint(0, 100, (2, 16)))
    assert loss.item() > 0


def test_gelu_mode_forward():
    model = SNNLM(_cfg(snn_mode="gelu"))
    x = torch.randint(0, 100, (2, 16))
    _, loss = model(x, torch.randint(0, 100, (2, 16)))
    assert loss.item() > 0


def test_track_stats_requires_snn_mode():
    cfg = _cfg(snn_track_stats=True)
    model = SNNLM(cfg)
    x = torch.randint(0, 100, (1, 8))
    model(x)
    for block in model.blocks:
        lr = getattr(block.ffn, "last_firing_rate", None)
        assert lr is not None
        assert 0.0 <= lr.item() <= 1.0


def test_tie_weights():
    cfg = _cfg(tie_weights=True)
    model = SNNLM(cfg)
    assert model.head.weight is model.tok_emb.weight


def test_no_tie_weights():
    cfg = _cfg(tie_weights=False)
    model = SNNLM(cfg)
    assert model.head.weight is not model.tok_emb.weight
