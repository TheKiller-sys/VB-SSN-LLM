"""Tests para LoRA: forward-equivalencia, merge, save/load."""
import pytest
import torch
import torch.nn as nn

from snn_lm_v3.lora import (
    LoRALinear, apply_lora, freeze_base_for_lora,
    lora_state_dict, load_lora_state_dict, save_lora, load_lora,
    merge_lora, is_lora_active, count_lora_params,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 16, bias=False)
        self.proj = nn.Linear(16, 16, bias=False)
        self.other = nn.Linear(16, 8, bias=False)


def test_lora_forward_equals_base_initially():
    base = nn.Linear(16, 32, bias=False)
    x = torch.randn(4, 16)
    y_base = base(x).clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    y_lora = lora(x)
    assert torch.allclose(y_base, y_lora, atol=1e-6)


def test_lora_forward_changes_after_B_update():
    base = nn.Linear(16, 32, bias=False)
    x = torch.randn(4, 16)
    lora = LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.fill_(0.1)
    y_lora = lora(x)
    y_base = lora.base(x)
    assert not torch.allclose(y_base, y_lora)


def test_apply_lora_targets():
    model = TinyModel()
    n = apply_lora(model, r=4, alpha=8.0, targets=("qkv", "proj"))
    assert n == 2
    assert isinstance(model.qkv, LoRALinear)
    assert isinstance(model.proj, LoRALinear)
    assert isinstance(model.other, nn.Linear)  # no target


def test_apply_lora_raises_if_no_match():
    model = TinyModel()
    with pytest.raises(ValueError):
        apply_lora(model, targets=("nonexistent",))


def test_freeze_base_for_lora():
    model = TinyModel()
    apply_lora(model, r=4, alpha=8.0)
    n = freeze_base_for_lora(model)
    assert n == 4  # 2 layers × 2 tensors
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all((".lora_A" in n or ".lora_B" in n) for n in trainable)


def test_count_lora_params():
    model = TinyModel()
    apply_lora(model, r=4, alpha=8.0, targets=("qkv", "proj"))
    # 2 layers × (16*4 + 4*16) = 2 × 128 = 256
    assert count_lora_params(model) == 256


def test_merge_lora_preserves_output():
    torch.manual_seed(0)
    base = nn.Linear(16, 32, bias=False)
    lora = LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    x = torch.randn(4, 16)
    y_before = lora(x).clone()

    model = nn.Module()
    model.layer = lora
    merge_lora(model)
    y_after = model.layer(x)
    assert torch.allclose(y_before, y_after, atol=1e-5)


def test_save_load_roundtrip(tmp_path):
    model = TinyModel()
    apply_lora(model, r=4, alpha=8.0)
    freeze_base_for_lora(model)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_(0, 0.1)
    path = str(tmp_path / "adapter.pt")
    save_lora(model, path, extra={"test": 42})

    model2 = TinyModel()
    apply_lora(model2, r=4, alpha=8.0)
    meta = load_lora(model2, path)
    assert meta["test"] == 42

    for (n1, m1), (n2, m2) in zip(
        [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)],
        [(n, m) for n, m in model2.named_modules() if isinstance(m, LoRALinear)],
    ):
        assert torch.allclose(m1.lora_A, m2.lora_A)
        assert torch.allclose(m1.lora_B, m2.lora_B)


def test_load_lora_shape_mismatch(tmp_path):
    m1 = TinyModel()
    apply_lora(m1, r=4, alpha=8.0)
    path = str(tmp_path / "a.pt")
    save_lora(m1, path)

    m2 = TinyModel()
    apply_lora(m2, r=8, alpha=8.0)  # r distinto
    with pytest.raises(ValueError):
        load_lora(m2, path)


def test_is_lora_active():
    m = TinyModel()
    assert not is_lora_active(m)
    apply_lora(m, r=4, alpha=8.0)
    assert is_lora_active(m)
