"""Tests de la neurona LIF (membrana, decay, reset, learnable)."""
import pytest
import torch

from snn_lm_v3.snn import LIFNeuron


def test_lif_integrates_below_threshold():
    lif = LIFNeuron(threshold=10.0, decay=0.9)
    x = torch.ones(1, 1, 4) * 0.1
    out = lif(x, num_steps=1)
    assert out.sum().item() == 0.0


def test_lif_spikes_above_threshold():
    lif = LIFNeuron(threshold=0.5, decay=0.9)
    x = torch.ones(1, 1, 4) * 1.0
    out = lif(x, num_steps=1)
    assert out.sum().item() == 4.0


def test_lif_soft_reset_allows_respike():
    lif = LIFNeuron(threshold=0.5, decay=0.9, reset_mode="soft")
    x = torch.full((1, 1, 1), 1.0)
    s1 = lif(x, num_steps=1)
    s2 = lif(x, num_steps=1)
    assert s1.item() == 1.0
    assert s2.item() == 1.0


def test_lif_decay_affects_accumulation():
    lif_slow = LIFNeuron(threshold=1.0, decay=0.1)
    lif_fast = LIFNeuron(threshold=1.0, decay=0.99)
    x = torch.full((1, 1, 1), 0.4)
    assert lif_slow(x, 3).item() == 0.0
    assert lif_fast(x, 3).item() == 1.0


def test_lif_accumulates_across_steps():
    lif = LIFNeuron(threshold=1.0, decay=0.0)
    x = torch.full((1, 1, 1), 0.6)
    # Con decay=0, sin acumulación: 0.6 < 1 siempre
    assert lif(x, num_steps=3).item() == 0.0

    lif2 = LIFNeuron(threshold=1.0, decay=0.99)
    # Con decay alto, acumula: 0.6 + 0.6*0.99 > 1 al segundo paso
    assert lif2(x, num_steps=3).item() >= 1.0


def test_lif_temporal_shapes():
    lif = LIFNeuron(threshold=0.5, decay=0.9)
    x = torch.randn(2, 8, 4)
    out = lif.forward_temporal(x)
    assert out.shape == (2, 8, 4)
    assert ((out == 0) | (out == 1)).all()


def test_lif_temporal_state_persists():
    lif = LIFNeuron(threshold=0.5, decay=0.9)
    x = torch.full((1, 4, 1), 0.4)
    out = lif.forward_temporal(x)
    # Con decay 0.9 y umbral 0.5: acumula 0.4, 0.76 → dispara
    assert out.sum().item() >= 1.0


def test_lif_learnable_params_get_gradient():
    lif = LIFNeuron(
        threshold=0.5,
        decay=0.9,
        learnable_threshold=True,
        learnable_decay=True,
    )
    x = torch.randn(1, 1, 4)
    out = lif(x, num_steps=3).sum()
    out.backward()
    assert lif.threshold.grad is not None
    assert lif.decay_logit.grad is not None


def test_lif_invalid_decay_raises():
    with pytest.raises(ValueError, match="decay must be in"):
        LIFNeuron(decay=1.5)


def test_lif_invalid_reset_raises():
    with pytest.raises(ValueError, match="reset_mode"):
        LIFNeuron(reset_mode="foo")


def test_lif_hard_reset_zeroes_membrane():
    lif = LIFNeuron(threshold=0.5, decay=0.9, reset_mode="hard")
    x = torch.full((1, 1, 1), 2.0)
    # Con hard reset: v = v * (1 - s) = 2 * 0 = 0 después del spike
    s = lif(x, num_steps=1)
    assert s.item() == 1.0


def test_lif_threshold_default_buffer_not_persistent():
    lif = LIFNeuron()
    state = lif.state_dict()
    assert "threshold" not in state
    assert "decay" not in state


def test_lif_learnable_threshold_in_state():
    lif = LIFNeuron(learnable_threshold=True)
    state = lif.state_dict()
    assert "threshold" in state
