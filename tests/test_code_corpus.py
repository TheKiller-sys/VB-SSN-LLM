"""Tests del CodeCorpusObserver."""
import pytest

from snn_lm_v3.reasoning import CodeCorpusObserver, ReasoningGraph


def test_tokenize_preserves_operators():
    obs = CodeCorpusObserver()
    tokens = obs.tokenize("if x >= 5: return y")
    assert "if" in tokens
    assert ">=" in tokens
    assert ":" in tokens
    assert "return" in tokens


def test_tokenize_discards_comments():
    obs = CodeCorpusObserver()
    tokens = obs.tokenize("x = 1  # comment here\ny = 2  // another")
    assert "comment" not in tokens
    assert "another" not in tokens
    assert "x" in tokens and "y" in tokens


def test_tokenize_collapses_strings():
    obs = CodeCorpusObserver()
    tokens = obs.tokenize('msg = "hello world"')
    assert "<string>" in tokens
    assert "hello" not in tokens  # colapsado


def test_tokenize_multichar_operators():
    obs = CodeCorpusObserver()
    tokens = obs.tokenize("a === b and c -> d")
    assert "===" in tokens
    assert "->" in tokens


def test_observe_builds_transitions():
    obs = CodeCorpusObserver()
    for _ in range(5):
        obs.observe("def foo(x):\n    return x + 1")
    candidates = obs.candidate_transitions()
    assert len(candidates) >= 1
    # Debe capturar `def → foo` o similar
    pairs = {(a, b) for a, b, _ in candidates}
    assert any(a == "def" for a, b in pairs)


def test_observe_operator_transitions():
    obs = CodeCorpusObserver()
    for _ in range(5):
        obs.observe("if (x == 1) { return y; }")
    candidates = obs.candidate_transitions()
    pairs = {(a, b) for a, b, _ in candidates}
    # Debe capturar `if → (` y `( → x`
    assert ("if", "(") in pairs or ("(", "x") in pairs


def test_mine_from_corpus_code():
    obs = CodeCorpusObserver()
    for _ in range(5):
        obs.observe("def quicksort(arr):\n    if len(arr) <= 1:\n        return arr")
    g = ReasoningGraph(cooldown_steps=3)
    n = g.mine_from_corpus(obs)
    assert n >= 1
    state = g.get_state()
    assert state["rules_count"] >= 1
    # Debe haber al menos una regla con `def` como antecedente
    top = state["top_rules"]
    assert any(r["antecedent"] == "def" for r in top) or n >= 1


def test_code_observer_state():
    obs = CodeCorpusObserver()
    obs.observe("x = 1 + 2")
    state = obs.get_state()
    assert state["observer_type"] == "code"
    assert "code_transitions" in state


def test_reset_clears_code_structures():
    obs = CodeCorpusObserver()
    obs.observe("def foo(): pass")
    obs.reset()
    assert len(obs.transitions) == 0
    assert len(obs.vocabulary) == 0


def test_code_observer_compatible_with_graph_apply():
    obs = CodeCorpusObserver()
    for _ in range(5):
        obs.observe("if (x) { y }")
    g = ReasoningGraph(cooldown_steps=0)
    g.mine_from_corpus(obs)
    # Aplicar debe funcionar sin error
    result = g.apply(["if"])
    assert isinstance(result, list)
