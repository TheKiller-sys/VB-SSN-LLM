"""Razonamiento simbólico neuro-simbólico: corpus, grafo, verificación."""
from .corpus import (
    CorpusObserver, Observation, Transition,
    STOPWORDS, MIN_TOKEN_LEN,
    MIN_OCCURRENCES_FOR_SYMBOL, MIN_OCCURRENCES_FOR_RULE,
    MIN_TRANSITION_DELAY_CHARS, MAX_TRANSITION_DELAY_CHARS,
)
from .code_corpus import CodeCorpusObserver, CodeTransition

from .graph import (
    ReasoningGraph, Rule,
    MAX_RULES, MIN_RULE_CONFIDENCE, DEFAULT_COOLDOWN_STEPS,
)

from .persistence import save_graph, load_graph, GRAPH_FORMAT_VERSION

from .signals import SignalProbe, InternalSignals

from .verifier import (
    ConsistencyVerifier, Verdict, Problem,
    NARRATIVE_PATTERNS,
)

__all__ = [
    # corpus
    "CorpusObserver", "Observation", "Transition",
    "STOPWORDS", "MIN_TOKEN_LEN",
    "MIN_OCCURRENCES_FOR_SYMBOL", "MIN_OCCURRENCES_FOR_RULE",
    "MIN_TRANSITION_DELAY_CHARS", "MAX_TRANSITION_DELAY_CHARS",
    # code corpus (v4.1)
    "CodeCorpusObserver", "CodeTransition",
    # graph
    "ReasoningGraph", "Rule",
    "MAX_RULES", "MIN_RULE_CONFIDENCE", "DEFAULT_COOLDOWN_STEPS",
    # persistence
    "save_graph", "load_graph", "GRAPH_FORMAT_VERSION",
    # signals
    "SignalProbe", "InternalSignals",
    # verifier
    "ConsistencyVerifier", "Verdict", "Problem",
    "NARRATIVE_PATTERNS",
]
