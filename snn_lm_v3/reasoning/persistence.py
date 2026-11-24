"""Persistencia del ReasoningGraph en disco (formato torch seguro).

Uso:
    save_graph(graph, "checkpoints/graph.pt")
    graph = load_graph("checkpoints/graph.pt")
"""
from __future__ import annotations

import os

import torch

GRAPH_FORMAT_VERSION = 1


def save_graph(graph, path: str) -> None:
    """Guarda las reglas y símbolos del grafo en disco (atómico)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": GRAPH_FORMAT_VERSION,
        "rules": [r.to_persistence() for r in graph.rules.values()],
        "symbols": dict(graph._symbols),
        "cooldown_steps": graph.cooldown_steps,
        "step": graph._step,
        "metrics": dict(graph.metrics),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_graph(path: str):
    """Carga un grafo guardado. Devuelve un ReasoningGraph listo para usar."""
    from .graph import ReasoningGraph, Rule

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version", 0)
    if version != GRAPH_FORMAT_VERSION:
        import logging
        logging.warning(
            f"Graph version {version} != {GRAPH_FORMAT_VERSION}. May be incompatible."
        )

    graph = ReasoningGraph(cooldown_steps=payload.get("cooldown_steps", 3))
    graph._step = payload.get("step", 0)
    graph._symbols = dict(payload.get("symbols", {}))
    graph.metrics.update(payload.get("metrics", {}))

    for data in payload.get("rules", []):
        rule = Rule.from_persistence(data)
        if rule is None:
            continue
        graph.rules[rule.id] = rule
        graph.by_antecedent.setdefault(
            graph._rule_key(rule.antecedent), []
        ).append(rule.id)

    return graph
