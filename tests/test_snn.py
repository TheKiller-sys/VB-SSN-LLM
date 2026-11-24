"""Tests de la activación spike y del FFN configurable."""
import pytest
import torch

from snn_lm_v3.snn import (
    spike_fn, SpikingActivation, SpikingFFN, SurrogateSpike,
)


def test_spike_forward_threshold():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    s = spike_fn(x, alpha=2.0)
    assert s.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_spike_backward_triangle():
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    spike_fn(x, alpha=2.0).sum().backward()
    expected = torch.tensor([0.0, 0.375, 0.5, 0.375, 0.0])
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_spike_backward_non_negative():
    x = torch.randn(8, requires_grad=True)
    spike_fn(x, alpha=2.0).sum().backward()
    assert (x.grad >= 0).all()


def test_spiking_activation_residual_mode():
    act = SpikingActivation(threshold=0.5, alpha=2.0, residual=True)
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = act(x)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 1.0]))


def test_spiking_activation_binary_mode():
    act = SpikingActivation(threshold=0.5, alpha=2.0, residual=False)
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = act(x)
    assert torch.allclose(out, torch.tensor([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("mode", ["static", "multi_step", "temporal", "gelu"])
def test_ffn_all_modes_shape(mode):
    ffn = SpikingFFN(8, mode=mode, num_steps=3, dropout=0.0)
    x = torch.randn(2, 5, 8)
    y = ffn(x)
    assert y.shape == (2, 5, 8)


@pytest.mark.parametrize("mode", ["static", "multi_step", "temporal", "gelu"])
def test_ffn_all_modes_gradient(mode):
    ffn = SpikingFFN(8, mode=mode, num_steps=3, dropout=0.0)
    x = torch.randn(1, 4, 8, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None
    assert ffn.fc1.weight.grad is not None
    assert ffn.fc2.weight.grad is not None


def test_ffn_use_spiking_false_forces_gelu():
    ffn = SpikingFFN(8, mode="multi_step", use_spiking=False)
    assert ffn.mode == "gelu"
    x = torch.randn(1, 4, 8)
    y = ffn(x)
    assert y.shape == (1, 4, 8)


def test_ffn_track_stats_fills_firing_rate():
    ffn = SpikingFFN(8, mode="multi_step", num_steps=3, track_stats=True)
    x = torch.randn(1, 4, 8)
    _ = ffn(x)
    assert 0.0 <= ffn.last_firing_rate.item() <= 1.0


def test_ffn_temporal_produces_binary_spikes():
    ffn = SpikingFFN(8, mode="temporal", num_steps=1, dropout=0.0, residual=False)
    x = torch.randn(1, 6, 8)
    h = ffn.fc1(x)
    s = ffn.lif.forward_temporal(h)
    assert ((s == 0) | (s == 1)).all()


def test_ffn_invalid_mode_raises():
    with pytest.raises(ValueError, match="invalid mode"):
        SpikingFFN(8, mode="foo")


def test_ffn_invalid_reset_mode_raises():
    with pytest.raises(ValueError):
        SpikingFFN(8, mode="multi_step", reset_mode="foo")


def test_ffn_temporal_has_lif():
    ffn = SpikingFFN(8, mode="temporal")
    assert hasattr(ffn, "lif")


def test_ffn_static_no_lif():
    ffn = SpikingFFN(8, mode="static")
    assert not hasattr(ffn, "lif")


def test_surrogate_spike_class_direct_use():
    x = torch.tensor([0.0, 1.0], requires_grad=True)
    out = SurrogateSpike.apply(x, 2.0)
    assert torch.equal(out, torch.tensor([1.0, 1.0]))
