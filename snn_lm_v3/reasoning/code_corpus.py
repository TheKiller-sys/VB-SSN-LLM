"""CodeCorpusObserver: razonamiento simbólico sobre CÓDIGO.

Diferencias con CorpusObserver (prosa):
  - Tokeniza con regex de código (operadores, keywords, strings).
  - Ignora comentarios (ruido semántico).
  - Delays en transiciones medidos en TOKENS, no en caracteres.
  - Preserva operadores estructurales: `{ → }`, `def → ident`, `if → (`.

Compatibilidad total:
  - Misma API que CorpusObserver.
  - ReasoningGraph.mine_from_corpus(observer) funciona sin cambios.
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .corpus import (
    Observation, CorpusObserver,
    MAX_OBSERVATIONS, MAX_STATE_LOG, MAX_TRANSITIONS,
    MAX_DELAYS_PER_TRANSITION, MAX_SEQUENCE_LEN,
)


# ─────────────────────────────────────────────────────────────
# Regex de tokenización de código
# ─────────────────────────────────────────────────────────────

# Operadores multi-char (más largos primero para evitar fragmentación)
MULTI_CHAR_OPS = [
    "===", "!==", ">>>", "<<=", ">>=", "**=", "//=", "->", "=>", "::",
    "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=",
    "/=", "%=", "&=", "|=", "^=", "<<", ">>", "**", "//",
]

SINGLE_CHAR_OPS = "(){}[];,.:=+-*/%<>&|^~!?@"

_CODE_TOKEN_RE = re.compile(
    r'(?P<docstr>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
    r'|(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`)'
    r'|(?P<comment>\#[^\n]*|//[^\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)'
    r'|(?P<number>\d+\.?\d*(?:[eE][+-]?\d+)?)'
    r'|(?P<multi_op>' + "|".join(re.escape(op) for op in MULTI_CHAR_OPS) + r')'
    r'|(?P<single_op>[' + re.escape(SINGLE_CHAR_OPS) + r'])'
    r'|(?P<newline>\n)',
    re.MULTILINE,
)


# ─────────────────────────────────────────────────────────────
# Transición tipada para código
# ─────────────────────────────────────────────────────────────

@dataclass
class CodeTransition:
    a: str
    b: str
    count: int = 0
    delays: list[int] = field(default_factory=list)

    def record(self, delay: int) -> None:
        self.count += 1
        self.delays.append(delay)
        if len(self.delays) > MAX_DELAYS_PER_TRANSITION:
            self.delays.pop(0)

    @property
    def mean_delay(self) -> float:
        return sum(self.delays) / len(self.delays) if self.delays else 0.0


# ─────────────────────────────────────────────────────────────
# Observer
# ─────────────────────────────────────────────────────────────

class CodeCorpusObserver(CorpusObserver):
    """Observa código: operadores, keywords, identificadores, strings.

    Uso:
        obs = CodeCorpusObserver()
        for chunk in code_chunks:
            obs.observe(chunk)
        graph = ReasoningGraph(cooldown_steps=3)
        graph.mine_from_corpus(obs)
    """

    # Ventana de co-ocurrencia (en tokens). Código tiene estructura local
    # muy densa: `def → ident`, `ident → (`, `( → ident`, `) → :`.
    TRANSITION_WINDOW = 8

    # Umbrales adaptados a código (keywords aparecen MUCHO más que en prosa)
    MIN_SYMBOL_OCCURRENCES = 3
    MIN_RULE_COUNT = 3

    def __init__(self, enc_name: str = "cl100k_base"):
        super().__init__(enc_name=enc_name)
        # Reemplazar dict de transiciones por el tipado de código
        self.transitions: dict[tuple[str, str], CodeTransition] = {}

    # ---------------------------------------------------------- tokenización
    def tokenize(self, text: str) -> list[str]:
        """Tokeniza código. Comentarios descartados, strings/docstrings
        colapsados a un token único, operadores y keywords preservados."""
        if not text:
            return []

        tokens: list[str] = []
        for m in _CODE_TOKEN_RE.finditer(text):
            kind = m.lastgroup
            value = m.group()
            if kind == "comment" or kind == "newline":
                continue
            if kind == "docstr":
                tokens.append("<docstr>")
                continue
            if kind == "string":
                tokens.append("<string>")
                continue
            tokens.append(value)
        return tokens

    # ---------------------------------------------------------- observación
    def observe(self, text: str, ref: str = "") -> Observation:
        self.chunk_counter += 1
        tokens = self.tokenize(text)

        obs = Observation(
            text=text,
            tokens=tokens,
            at=time.time(),
            ref=ref,
            chunk_id=self.chunk_counter,
        )
        self.observations.append(obs)
        if len(self.observations) > MAX_OBSERVATIONS:
            self.observations.pop(0)

        for t in tokens:
            self.vocabulary[t] += 1

        for i in range(len(tokens)):
            window = tokens[i : i + MAX_SEQUENCE_LEN]
            if len(window) >= 2:
                self.sequences.append(window)
        if len(self.sequences) > MAX_SEQUENCE_LEN * 100:
            self.sequences = self.sequences[-MAX_SEQUENCE_LEN * 100 :]

        self._record_transitions(text, tokens)

        self.state_log.append({
            "at": obs.at,
            "chunk_id": self.chunk_counter,
            "facts": list(tokens),
        })
        if len(self.state_log) > MAX_STATE_LOG:
            self.state_log.pop(0)

        self.metrics["observations"] += 1
        self.metrics["tokens_total"] += len(tokens)
        self.metrics["tokens_unique"] = len(self.vocabulary)
        return obs

    # ---------------------------------------------------------- transiciones
    def _record_transitions(self, text: str, tokens: list[str]) -> None:
        """Registra (A, B) con delay en TOKENS dentro de ventana local."""
        seen: set[tuple[str, str]] = set()
        n = len(tokens)

        for i, a in enumerate(tokens):
            end = min(i + self.TRANSITION_WINDOW, n)
            for j in range(i + 1, end):
                b = tokens[j]
                if a == b:
                    continue
                pair = (a, b)
                if pair in seen:
                    continue
                seen.add(pair)

                if pair not in self.transitions:
                    if len(self.transitions) >= MAX_TRANSITIONS:
                        continue
                    self.transitions[pair] = CodeTransition(a=a, b=b)
                self.transitions[pair].record(j - i)

        self.metrics["transitions"] = len(self.transitions)

    # ---------------------------------------------------------- símbolos
    def candidate_symbols(self) -> list[tuple[str, int]]:
        """Tokens significativos (excluye strings/docstrings genéricos)."""
        return [
            (t, c) for t, c in self.vocabulary.items()
            if c >= self.MIN_SYMBOL_OCCURRENCES
            and t not in ("<string>", "<docstr>")
        ]

    def candidate_transitions(self) -> list[tuple[str, str, int]]:
        return [
            (a, b, tr.count)
            for (a, b), tr in self.transitions.items()
            if tr.count >= self.MIN_RULE_COUNT
        ]

    # ---------------------------------------------------------- estado
    def get_state(self) -> dict:
        base = super().get_state()
        base["observer_type"] = "code"
        base["code_transitions"] = len(self.transitions)
        return base

    def reset(self) -> None:
        super().reset()
        self.transitions = {}
        self.vocabulary = Counter()
