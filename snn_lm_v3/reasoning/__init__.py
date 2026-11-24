"""Módulo de razonamiento simbólico sobre corpus para SNN-LM.

Porta los conceptos útiles de SymbolicAbstractionLayer.js y VerifierCritic.js
adaptados a un LM:

  - CorpusObserver: convierte texto en observaciones, transiciones y símbolos.
  - ReasoningGraph: grafo de reglas con anti-ciclo, cooldown, tautología, pruning.
  - ConsistencyVerifier: verifica texto generado contra corpus y contra señales.
  - SignalProbe: extrae señales internas del modelo (firing rate, entropy, etc.).
  - save_graph / load_graph: persistencia del grafo.
"""
from .signals import SignalProbe, InternalSignals
from .corpus import (
    CorpusObserver, Observation, Transition,
    STOPWORDS,
    MIN_TOKEN_LEN, MIN_OCCURRENCES_FOR_SYMBOL, MIN_OCCURRENCES_FOR_RULE,
    MAX_OBSERVATIONS, MAX_STATE_LOG, MAX_SEQUENCE_LEN,
)
from .graph import (
    ReasoningGraph, Rule,
    MAX_RULES, MIN_RULE_CONFIDENCE,
    DEFAULT_COOLDOWN_STEPS,
)
from .verifier import (
    ConsistencyVerifier, Verdict, Problem,
    NARRATIVE_PATTERNS,
)
from .persistence import save_graph, load_graph

__all__ = [
    # signals
    "SignalProbe", "InternalSignals",
    # corpus
    "CorpusObserver", "Observation", "Transition",
    "STOPWORDS",
    "MIN_TOKEN_LEN", "MIN_OCCURRENCES_FOR_SYMBOL", "MIN_OCCURRENCES_FOR_RULE",
    "MAX_OBSERVATIONS", "MAX_STATE_LOG", "MAX_SEQUENCE_LEN",
    # graph
    "ReasoningGraph", "Rule",
    "MAX_RULES", "MIN_RULE_CONFIDENCE",
    "DEFAULT_COOLDOWN_STEPS",
    # verifier
    "ConsistencyVerifier", "Verdict", "Problem",
    "NARRATIVE_PATTERNS",
    # persistence
    "save_graph", "load_graph",
]
