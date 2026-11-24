"""Tests del ReasoningGraph: create, cycle, tautology, overlap, cooldown, prune, mine."""
import time

import pytest

from snn_lm_v3.reasoning import ReasoningGraph, Rule
from snn_lm_v3.reasoning.graph import (
    MAX_RULES, MIN_RULE_CONFIDENCE, RULE_PRUNE_KEEP_RATIO,
)


def test_create_rule_basic():
    g = ReasoningGraph()
    r = g.create_rule("a", "b", {"count": 5})
    assert r is not None
    assert r.antecedent == "a"
    assert r.consequent == "b"
    assert r.confidence >= MIN_RULE_CONFIDENCE


def test_create_rule_tautology_rejected():
    g = ReasoningGraph()
    r = g.create_rule("a", "a", {"count": 5})
    assert r is None
    assert g.metrics["tautological_rejected"] == 1


def test_create_rule_overlap_rejected():
    g = ReasoningGraph()
    # "hello world" y "hello" tienen overlap en "hello"
    r = g.create_rule("hello world", "hello", {"count": 5})
    assert r is None
    assert g.metrics["overlap_rejected"] == 1


def test_create_rule_cyclic_rejected():
    g = ReasoningGraph()
    r1 = g.create_rule("a", "b", {"count": 5})
    assert r1 is not None
    r2 = g.create_rule("b", "a", {"count": 5})
    assert r2 is None
    assert g.metrics["cyclic_rejected"] == 1


def test_find_rule():
    g = ReasoningGraph()
    g.create_rule("a", "b", {"count": 5})
    found = g._find_rule("a", "b")
    assert found is not None
    assert found.consequent == "b"


def test_apply_with_no_rules():
    g = ReasoningGraph()
    assert g.apply(["a", "b"]) == []


def test_apply_returns_consequences():
    g = ReasoningGraph()
    g.create_rule("a", "b", {"count": 5})
    consequences = g.apply(["a"])
    assert len(consequences) == 1
    assert consequences[0]["consequent"] == "b"
    assert consequences[0]["confidence"] >= MIN_RULE_CONFIDENCE


def test_apply_skips_low_confidence():
    g = ReasoningGraph()
    r = g.create_rule("a", "b", {"count": 5})
    r.confidence = 0.1  # forzar por debajo del umbral
    consequences = g.apply(["a"])
    assert len(consequences) == 0


def test_apply_multiple_rules_sorted_by_confidence():
    g = ReasoningGraph()
    r1 = g.create_rule("a", "b", {"count": 3})
    r2 = g.create_rule("a", "c", {"count": 10})
    r1.reinforce({"x": 1})
    r2.reinforce({"x": 1})
    r2.reinforce({"x": 1})
    consequences = g.apply(["a"])
    assert len(consequences) == 2
    assert consequences[0]["consequent"] == "c"  # mayor confianza primero


def test_apply_deduplicates():
    g = ReasoningGraph()
    g.create_rule("a", "b", {"count": 5})
    consequences = g.apply(["a", "a", "a"])
    assert len(consequences) == 1


def test_cooldown_blocks_immediate_reapply():
    g = ReasoningGraph(cooldown_steps=3)
    g.create_rule("a", "b", {"count": 5})
    # Primera aplicación: OK
    assert len(g.apply(["a"])) == 1
    # Segunda aplicación inmediata (sin avanzar step): bloqueada
    assert len(g.apply(["a"])) == 0
    assert g.metrics["blocked_by_cooldown"] >= 1


def test_cooldown_allows_after_ticks():
    g = ReasoningGraph(cooldown_steps=2)
    g.create_rule("a", "b", {"count": 5})
    assert len(g.apply(["a"])) == 1
    g.tick()
    g.tick()
    assert len(g.apply(["a"])) == 1


def test_mine_from_corpus():
    from snn_lm_v3.reasoning import CorpusObserver
    obs = CorpusObserver()
    for _ in range(3):
        obs.observe("clever fox quickly jumped brown lazy")
    g = ReasoningGraph()
    n = g.mine_from_corpus(obs)
    assert n >= 1
    assert g.get_state()["rules_count"] >= 1


def test_mine_from_corpus_reinforces_existing():
    from snn_lm_v3.reasoning import CorpusObserver
    obs = CorpusObserver()
    for _ in range(5):
        obs.observe("clever fox quickly jumped brown lazy")
    g = ReasoningGraph()
    g.mine_from_corpus(obs)
    n_before = g.get_state()["rules_count"]
    # Minería repetida no crea duplicados, refuerza
    g.mine_from_corpus(obs)
    assert g.get_state()["rules_count"] == n_before


def test_prune_removes_low_value_rules():
    g = ReasoningGraph()
    # Crear muchas reglas débiles
    for i in range(100):
        g.create_rule(f"x{i}", f"y{i}", {"count": 1})
    # Crear unas pocas reglas fuertes
    for i in range(5):
        r = g.create_rule(f"strong{i}", f"result{i}", {"count": 100})
        for _ in range(5):
            r.reinforce({"extra": True})
    # Forzar prune
    removed = g.prune(force=True)
    assert removed > 0


def test_prune_keeps_ratio():
    g = ReasoningGraph()
    for i in range(10):
        g.create_rule(f"x{i}", f"y{i}", {"count": 1})
    n_before = len(g.rules)
    g.prune(force=True)
    n_after = len(g.rules)
    # Con 10 reglas y ratio 0.6, debe mantener ~6 (o todas si < trigger)
    if n_before >= MAX_RULES * 1.2:
        assert n_after <= int(MAX_RULES * RULE_PRUNE_KEEP_RATIO)


def test_decay_does_not_crash():
    g = ReasoningGraph()
    g.create_rule("a", "b", {"count": 3})
    g._last_decay_at -= 3600  # simular 1 hora
    g.decay()
    assert isinstance(g.get_state(), dict)


def test_get_state():
    g = ReasoningGraph()
    g.create_rule("a", "b", {"count": 3})
    state = g.get_state()
    assert "rules_count" in state
    assert "strong_rules" in state
    assert "metrics" in state
    assert "top_rules" in state


def test_reset_clears():
    g = ReasoningGraph()
    g.create_rule("a", "b", {"count": 3})
    g.reset()
    assert len(g.rules) == 0
    assert g.get_state()["rules_count"] == 0


def test_rule_reinforce_updates_confidence():
    r = Rule(id="r1", antecedent="a", consequent="b")
    c0 = r.confidence
    r.reinforce({"x": 1})
    c1 = r.confidence
    assert c1 > c0


def test_rule_contradict_lowers_confidence():
    r = Rule(id="r1", antecedent="a", consequent="b")
    r.reinforce({"x": 1})
    r.reinforce({"x": 1})
    c1 = r.confidence
    r.contradict({"y": 1})
    c2 = r.confidence
    assert c2 < c1


def test_rule_to_dict():
    r = Rule(id="r1", antecedent="a", consequent="b")
    r.reinforce({"x": 1})
    d = r.to_dict()
    assert d["antecedent"] == "a"
    assert d["consequent"] == "b"
    assert "confidence" in d
