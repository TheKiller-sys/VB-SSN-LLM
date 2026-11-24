"""Tests del ConsistencyVerifier."""
import pytest

from snn_lm_v3.reasoning import (
    ConsistencyVerifier, InternalSignals, Verdict,
    Problem,
)


def test_verdict_valid():
    v = ConsistencyVerifier()
    assert v.verdict([]) == Verdict.VALID


def test_verdict_levels():
    v = ConsistencyVerifier()
    assert v.verdict([Problem(type="x", severity=0.3)]) == Verdict.VALID_WITH_WARNINGS
    assert v.verdict([Problem(type="x", severity=0.7)]) == Verdict.SUSPECT
    assert v.verdict([Problem(type="x", severity=0.9)]) == Verdict.INVALID


def test_verify_calm_vs_silent_substrate():
    v = ConsistencyVerifier()
    signals = InternalSignals(mean_spike_rate=0.0001)
    problems = v.verify("estoy en calma total", signals)
    assert any(p.type == "contradicts_physiology" for p in problems)


def test_verify_flow_vs_silent_substrate():
    v = ConsistencyVerifier()
    signals = InternalSignals(mean_spike_rate=0.0001)
    problems = v.verify("me siento en flujo", signals)
    assert any(p.signal == "mean_spike_rate" for p in problems)


def test_verify_confidence_vs_high_entropy():
    v = ConsistencyVerifier()
    signals = InternalSignals(output_entropy=0.95, mean_spike_rate=0.1)
    problems = v.verify("tengo seguridad total", signals)
    assert any(p.type == "overconfidence_physiological" for p in problems)


def test_verify_certainty_vs_high_entropy():
    v = ConsistencyVerifier()
    signals = InternalSignals(output_entropy=0.95, mean_spike_rate=0.1)
    problems = v.verify("definitivamente es correcto", signals)
    assert any(p.type == "overconfidence_physiological" for p in problems)


def test_verify_self_knowledge_vs_high_ppl():
    v = ConsistencyVerifier()
    signals = InternalSignals(perplexity=200, mean_spike_rate=0.1)
    problems = v.verify("me conozco a la perfección", signals)
    assert any(p.signal == "perplexity" for p in problems)


def test_verify_no_problems_when_consistent():
    v = ConsistencyVerifier()
    signals = InternalSignals(
        mean_spike_rate=0.1, output_entropy=0.3, perplexity=10, spike_variance=0.001
    )
    problems = v.verify("texto neutro sin narrativa especial", signals)
    assert len(problems) == 0


def test_verify_against_corpus_oov():
    from snn_lm_v3.reasoning import CorpusObserver, ReasoningGraph
    obs = CorpusObserver()
    obs.observe("clever fox quickly jumped brown lazy")
    graph = ReasoningGraph()
    graph.mine_from_corpus(obs)
    v = ConsistencyVerifier()
    problems = v.verify_against_corpus(
        "xyz abc qqq unknown zzz nonsense tokens", graph, obs
    )
    assert any(p.type == "ungrounded_premise" for p in problems)


def test_verify_against_corpus_repetition():
    from snn_lm_v3.reasoning import CorpusObserver, ReasoningGraph
    obs = CorpusObserver()
    for _ in range(3):
        obs.observe("clever fox quickly jumped brown lazy")
    graph = ReasoningGraph()
    graph.mine_from_corpus(obs)
    v = ConsistencyVerifier()
    # Transición "clever fox" repetida en muchas posiciones
    problems = v.verify_against_corpus(
        "clever fox clever fox clever fox clever fox clever fox",
        graph, obs,
    )
    assert any(p.signal == "repetition" for p in problems)


def test_verify_against_corpus_empty():
    from snn_lm_v3.reasoning import CorpusObserver, ReasoningGraph
    obs = CorpusObserver()
    graph = ReasoningGraph()
    v = ConsistencyVerifier()
    problems = v.verify_against_corpus("", graph, obs)
    assert problems == []


def test_problem_to_dict():
    p = Problem(type="x", severity=0.5, note="test")
    d = p.to_dict()
    assert d["type"] == "x"
    assert d["severity"] == 0.5
    assert d["note"] == "test"
    assert "signal" not in d or d.get("signal") == ""


def test_metrics_accumulate():
    v = ConsistencyVerifier()
    signals = InternalSignals(mean_spike_rate=0.0001)
    v.verify("calma", signals)
    v.verify("paz", signals)
    assert v.metrics["checks"] == 2


def test_reset_metrics():
    v = ConsistencyVerifier()
    signals = InternalSignals(mean_spike_rate=0.0001)
    v.verify("calma", signals)
    v.reset()
    assert v.metrics["checks"] == 0
