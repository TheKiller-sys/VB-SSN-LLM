"""Tests del CorpusObserver."""
import pytest

from snn_lm_v3.reasoning import (
    CorpusObserver,
    STOPWORDS,
    MIN_TOKEN_LEN,
    MIN_OCCURRENCES_FOR_SYMBOL,
)


def test_tokenize_filters_stopwords():
    obs = CorpusObserver()
    tokens = obs.tokenize("the quick brown fox jumps over the lazy dog")
    assert "the" not in tokens
    assert "over" not in tokens
    assert "dog" not in tokens  # 3 chars < MIN_TOKEN_LEN
    assert "quick" in tokens
    assert "brown" in tokens
    assert "jumps" in tokens


def test_tokenize_empty():
    obs = CorpusObserver()
    assert obs.tokenize("") == []
    assert obs.tokenize(None) == []


def test_tokenize_lowercases():
    obs = CorpusObserver()
    tokens = obs.tokenize("HELLO World")
    assert "hello" in tokens
    assert "world" in tokens


def test_observe_updates_vocabulary():
    obs = CorpusObserver()
    obs.observe("clever fox quickly jumped brown lazy")
    assert "clever" in obs.vocabulary
    assert obs.vocabulary["clever"] == 1


def test_observe_accumulates_vocabulary():
    obs = CorpusObserver()
    for _ in range(3):
        obs.observe("clever fox quickly jumped brown lazy")
    assert obs.vocabulary["clever"] == 3


def test_observe_builds_transitions():
    obs = CorpusObserver()
    for _ in range(3):
        obs.observe("clever fox quickly jumped brown lazy")
    # Debe haber al menos 1 transición con count >= 3
    candidates = obs.candidate_transitions()
    assert len(candidates) >= 1


def test_observe_populates_state_log():
    obs = CorpusObserver()
    obs.observe("clever fox")
    assert len(obs.state_log) == 1
    assert "facts" in obs.state_log[0]
    assert "at" in obs.state_log[0]


def test_observe_bounded_state_log():
    from snn_lm_v3.reasoning.corpus import MAX_STATE_LOG
    obs = CorpusObserver()
    for i in range(MAX_STATE_LOG + 50):
        obs.observe(f"clever fox jumped quickly {i}")
    assert len(obs.state_log) <= MAX_STATE_LOG


def test_candidate_symbols_threshold():
    obs = CorpusObserver()
    obs.observe("clever fox")  # clever=1
    assert all(c < MIN_OCCURRENCES_FOR_SYMBOL for _, c in obs.candidate_symbols())

    for _ in range(MIN_OCCURRENCES_FOR_SYMBOL):
        obs.observe("clever fox")
    names = [name for name, _ in obs.candidate_symbols()]
    assert "clever" in names


def test_candidate_transitions_threshold():
    obs = CorpusObserver()
    # Una sola observación no alcanza el umbral de count
    obs.observe("clever fox quickly jumped brown lazy")
    # Con una sola observación puede haber 1 transición con count=1
    candidates = obs.candidate_transitions()
    assert all(c >= 3 for _, _, c in candidates)


def test_get_state():
    obs = CorpusObserver()
    obs.observe("clever fox quickly jumped")
    state = obs.get_state()
    assert "observations" in state
    assert "vocabulary_size" in state
    assert "transitions" in state
    assert state["observations"] == 1


def test_reset_clears():
    obs = CorpusObserver()
    obs.observe("clever fox quickly jumped")
    obs.reset()
    assert len(obs.observations) == 0
    assert len(obs.vocabulary) == 0
    assert len(obs.transitions) == 0
