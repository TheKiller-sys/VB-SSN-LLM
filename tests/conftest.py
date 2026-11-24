"""Fixtures compartidos por toda la suite."""
import random

import numpy as np
import pytest
import torch

from snn_lm_v3 import GPTConfig, SNNLM


@pytest.fixture(scope="session")
def gpt2_enc():
    """Encoding GPT-2 (skip si tiktoken no está disponible)."""
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception:
        pytest.skip("gpt2 encoding not available")


@pytest.fixture
def set_seed():
    def _set(s: int = 0) -> None:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        random.seed(s)
        np.random.seed(s)

    return _set


@pytest.fixture
def tiny_cfg():
    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        seq_len=16,
        vocab_size=100,
        dropout=0.0,
        use_spiking=True,
        snn_mode="multi_step",
        snn_num_steps=2,
        snn_track_stats=True,
    )


@pytest.fixture
def tiny_model(tiny_cfg, set_seed):
    set_seed(0)
    return SNNLM(tiny_cfg)
