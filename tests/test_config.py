"""Tests de GPTConfig: validación, sizes, save/load."""
import json
import os
import tempfile

import pytest

from snn_lm_v3 import GPTConfig, SIZE_TINY, SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE, SIZES


def test_default_config_valid():
    GPTConfig().validate()


def test_invalid_n_embd_n_head_raises():
    with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
        GPTConfig(n_embd=10, n_head=3).validate()


def test_invalid_dropout_raises():
    with pytest.raises(ValueError):
        GPTConfig(dropout=1.0).validate()
    with pytest.raises(ValueError):
        GPTConfig(dropout=-0.1).validate()


def test_invalid_lr_raises():
    with pytest.raises(ValueError):
        GPTConfig(lr=0.0).validate()


def test_min_lr_greater_than_lr_raises():
    with pytest.raises(ValueError):
        GPTConfig(lr=1e-4, min_lr=1e-3).validate()


def test_invalid_snn_mode_raises():
    with pytest.raises(ValueError):
        GPTConfig(snn_mode="foo").validate()


def test_invalid_snn_reset_mode_raises():
    with pytest.raises(ValueError):
        GPTConfig(snn_reset_mode="foo").validate()


def test_snn_decay_bounds():
    with pytest.raises(ValueError):
        GPTConfig(snn_decay=0.0).validate()
    with pytest.raises(ValueError):
        GPTConfig(snn_decay=1.0).validate()


def test_snn_num_steps_min():
    with pytest.raises(ValueError):
        GPTConfig(snn_num_steps=0).validate()


def test_temporal_mode_warns():
    with pytest.warns(RuntimeWarning):
        GPTConfig(snn_mode="temporal").validate()


def test_sizes_complete():
    assert "n_layer" in SIZE_TINY
    assert "n_layer" in SIZE_SMALL
    assert "n_layer" in SIZE_MEDIUM
    assert "n_layer" in SIZE_LARGE
    assert set(SIZES.keys()) == {"tiny", "small", "medium", "large"}


def test_save_load_roundtrip():
    cfg = GPTConfig(n_layer=3, n_head=4, n_embd=64, seq_len=128)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        cfg.save(path)
        assert os.path.exists(path)
        cfg2 = GPTConfig.load(path)
    assert cfg2.n_layer == 3
    assert cfg2.n_head == 4
    assert cfg2.n_embd == 64
    assert cfg2.seq_len == 128


def test_load_ignores_unknown_keys():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 8,
                    "bogus_field": 42,
                    "another": "x",
                },
                f,
            )
        cfg = GPTConfig.load(path)
    assert cfg.n_layer == 1
    assert cfg.n_embd == 8


def test_load_validates():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cfg.json")
        with open(path, "w") as f:
            json.dump({"n_embd": 10, "n_head": 3}, f)
        with pytest.raises(ValueError):
            GPTConfig.load(path)
