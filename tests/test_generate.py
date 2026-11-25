"""Tests del wrapper generate()."""
import pytest
import torch

from snn_lm_v3 import GPTConfig, SNNLM, generate


def _cfg():
    return GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=16, vocab_size=50257
    )


def test_generate_wrapper_returns_str(gpt2_enc):
    model = SNNLM(_cfg())
    out = generate(model, prompt="hello", max_new_tokens=3, temperature=1.0)
    assert isinstance(out, str)


def test_generate_empty_prompt_returns_str(gpt2_enc):
    model = SNNLM(_cfg())
    out = generate(model, prompt="", max_new_tokens=2, temperature=1.0)
    assert isinstance(out, str)


def test_generate_with_seed_is_reproducible(gpt2_enc):
    """Determinismo con memoria desactivada (evita side-effects entre llamadas)."""
    model = SNNLM(_cfg())
    out1 = generate(
        model, prompt="hello", max_new_tokens=5, seed=42, update_memory=False,
    )
    out2 = generate(
        model, prompt="hello", max_new_tokens=5, seed=42, update_memory=False,
    )
    assert out1 == out2


def test_generate_no_cache(gpt2_enc):
    model = SNNLM(_cfg())
    out = generate(
        model, prompt="hi", max_new_tokens=3, use_cache=False
    )
    assert isinstance(out, str)
