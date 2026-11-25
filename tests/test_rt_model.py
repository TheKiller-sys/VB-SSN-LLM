"""Tests del RTModelWrapper."""
import pytest
import torch

pytest.importorskip("tokenizers")

from snn_lm_v3 import GPTConfig, SNNLM
from snn_lm_v3.tokenizer import RTModelWrapper


def _cfg(tiny_rv, **kw):
    base = dict(
        n_layer=1, n_head=1, n_embd=16, seq_len=8,
        vocab_size=tiny_rv.vocab_size, dropout=0.0,
        use_spiking=True, snn_mode="multi_step", snn_num_steps=2,
    )
    base.update(kw)
    return GPTConfig(**base)


def test_rt_forward_shapes(tiny_rv):
    cfg = _cfg(tiny_rv)
    snn = SNNLM(cfg)
    model = RTModelWrapper(snn, tiny_rv, aux_loss_weight=0.1)
    ids = torch.randint(0, tiny_rv.vocab_size, (2, 8))
    logits, _ = model(ids)
    assert logits.shape == (2, 8, tiny_rv.vocab_size)


def test_rt_forward_loss(tiny_rv):
    cfg = _cfg(tiny_rv)
    snn = SNNLM(cfg)
    model = RTModelWrapper(snn, tiny_rv)
    ids = torch.randint(0, tiny_rv.vocab_size, (2, 8))
    targets = torch.randint(0, tiny_rv.vocab_size, (2, 8))
    logits, loss = model(ids, targets)
    assert loss.dim() == 0 and loss.item() > 0


def test_rt_vocab_mismatch_raises(tiny_rv):
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=8,
        vocab_size=100,  # mismatch
    )
    snn = SNNLM(cfg)
    with pytest.raises(ValueError, match="vocab_size"):
        RTModelWrapper(snn, tiny_rv)


def test_rt_generate(tiny_rv):
    cfg = _cfg(tiny_rv)
    snn = SNNLM(cfg)
    model = RTModelWrapper(snn, tiny_rv)
    model.eval()
    idx = torch.randint(0, tiny_rv.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=3, temperature=1.0)
    assert out.shape == (1, 7)


def test_rt_getattr_delegation(tiny_rv):
    cfg = _cfg(tiny_rv)
    snn = SNNLM(cfg)
    model = RTModelWrapper(snn, tiny_rv)
    assert model.config is snn.config
    assert model.neural_memory is snn.neural_memory
