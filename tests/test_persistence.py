"""Tests de save_graph / load_graph."""
import os
import tempfile

import pytest

from snn_lm_v3.reasoning import (
    ReasoningGraph, save_graph, load_graph,
)


def test_save_load_roundtrip():
    g = ReasoningGraph(cooldown_steps=5)
    g.create_rule("a", "b", {"count": 5})
    g.create_rule("c", "d", {"count": 3})
    n_rules = len(g.rules)
    n_metrics = dict(g.metrics)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.pt")
        save_graph(g, path)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        g2 = load_graph(path)

    assert len(g2.rules) == n_rules
    assert g2.cooldown_steps == 5
    assert g2.metrics["rules_created"] == n_metrics["rules_created"]


def test_load_nonexistent_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "missing.pt")
        with pytest.raises(FileNotFoundError):
            load_graph(path)


def test_save_creates_dir():
    g = ReasoningGraph()
    g.create_rule("a", "b", {"count": 3})
    with tempfile.TemporaryDirectory() as d:
        nested = os.path.join(d, "a", "b", "graph.pt")
        save_graph(g, nested)
        assert os.path.exists(nested)


def test_loaded_graph_can_apply():
    g = ReasoningGraph(cooldown_steps=1)
    g.create_rule("a", "b", {"count": 5})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.pt")
        save_graph(g, path)
        g2 = load_graph(path)
    consequences = g2.apply(["a"])
    assert len(consequences) == 1
    assert consequences[0]["consequent"] == "b"


def test_loaded_graph_preserves_confidence():
    g = ReasoningGraph()
    r = g.create_rule("a", "b", {"count": 5})
    r.reinforce({"extra": 1})
    conf = r.confidence
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.pt")
        save_graph(g, path)
        g2 = load_graph(path)
    rules = list(g2.rules.values())
    assert len(rules) == 1
    assert abs(rules[0].confidence - conf) < 1e-6
