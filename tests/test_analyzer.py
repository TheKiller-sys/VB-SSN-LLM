"""Tests del ContinuousAnalyzer."""
import time

import pytest
import torch

from snn_lm_v3 import GPTConfig, SNNLM, ContinuousAnalyzer


def _model():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=8, vocab_size=100,
        neural_memory_enabled=True,
        abstraction_enabled=True,
        self_knowledge_enabled=False,
    )
    return SNNLM(cfg)


def test_analyzer_init():
    model = _model()
    analyzer = ContinuousAnalyzer(model, interval_s=0.1)
    assert not analyzer.is_running()
    analyzer.start()
    assert analyzer.is_running()
    analyzer.stop()
    assert not analyzer.is_running()


def test_analyzer_run_once():
    model = _model()
    analyzer = ContinuousAnalyzer(model, interval_s=60.0)
    result = analyzer.run_once()
    assert isinstance(result, dict)
    assert analyzer.get_state()["metrics"]["runs"] == 1


def test_analyzer_prunes_memory():
    model = _model()
    model.eval()
    model(torch.randint(0, 100, (1, 8)), update_memory=True)
    n_before = int(model.neural_memory.size.item())
    assert n_before > 0

    analyzer = ContinuousAnalyzer(
        model, interval_s=60.0, prune_memory=True, refresh_codebook=False,
    )
    analyzer.run_once()  # phase 0: prune
    n_after = int(model.neural_memory.size.item())
    assert n_after == n_before  # sin NaNs, no debe prunear


def test_analyzer_refreshes_codebook():
    model = _model()
    model.eval()
    for _ in range(3):
        model(torch.randint(0, 100, (1, 8)), update_memory=True)

    analyzer = ContinuousAnalyzer(
        model, interval_s=60.0, prune_memory=False, refresh_codebook=True,
    )
    # Saltar a la fase de codebook
    analyzer._phase = 1
    analyzer.run_once()
    state = analyzer.get_state()
    assert state["metrics"]["codebook_refresh_runs"] == 1


def test_analyzer_background_loop():
    model = _model()
    analyzer = ContinuousAnalyzer(model, interval_s=0.05)
    analyzer.start()
    time.sleep(0.25)
    analyzer.stop()
    assert analyzer.get_state()["metrics"]["runs"] >= 1


def test_analyzer_stop_is_idempotent():
    model = _model()
    analyzer = ContinuousAnalyzer(model, interval_s=0.1)
    analyzer.start()
    analyzer.stop()
    analyzer.stop()  # no debe lanzar
