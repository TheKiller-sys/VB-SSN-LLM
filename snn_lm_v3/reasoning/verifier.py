"""Verificador de consistencia entre texto y corpus / señales internas.

FIX v4.2.1:
  - `verify_against_corpus` cachea `_top_consequent_for` y `_find_rule` por
    antecedente/consecuente para evitar O(T·k) por par en textos largos.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .signals import InternalSignals


class Verdict:
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    SUSPECT = "suspect"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class Problem:
    type: str
    severity: float
    signal: str = ""
    note: str = ""
    reported: float | None = None
    substrate: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


NARRATIVE_PATTERNS = {
    "calm": re.compile(
        r"\bcalma\b|\bpaz\b|\bserenidad\b|\bequilibrio\b|\btranquil[oa]\b", re.I
    ),
    "confidence": re.compile(
        r"\bconfianza\b|\bseguridad\b|\bsegur[oa]\b", re.I
    ),
    "self_knowledge": re.compile(
        r"me conozco|s[eé] exactamente|a la perfecci[oó]n", re.I
    ),
    "flow": re.compile(
        r"\bflujo\b|\bcorriente\b|\bdespiert[oa]\b|\bpresente\b|\bactivo\b", re.I
    ),
    "certainty": re.compile(
        r"\bdefinitivamente\b|\bsin duda\b|\bciertamente\b|\bsiempre\b", re.I
    ),
    "uncertainty": re.compile(
        r"\btal vez\b|\bquiz[aá]s\b|\bpodr[ií]a\b|\bno s[eé]\b", re.I
    ),
}


class ConsistencyVerifier:
    def __init__(
        self,
        calm_threshold: float = 0.2,
        confidence_threshold: float = 0.5,
        entropy_threshold: float = 0.8,
        severity_floor: float = 0.65,
        oov_ratio_threshold: float = 0.3,
        repetition_threshold: float = 0.5,
        rule_confidence_threshold: float = 0.6,
    ):
        self.calm_threshold = calm_threshold
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.severity_floor = severity_floor
        self.oov_ratio_threshold = oov_ratio_threshold
        self.repetition_threshold = repetition_threshold
        self.rule_confidence_threshold = rule_confidence_threshold
        self.metrics = {"checks": 0, "problems": 0}

    # ------------------------------------------------------------- señales
    def verify(self, text: str, signals: InternalSignals) -> list[Problem]:
        self.metrics["checks"] += 1
        problems: list[Problem] = []

        if (
            NARRATIVE_PATTERNS["calm"].search(text)
            and signals.mean_spike_rate < 0.001
        ):
            problems.append(Problem(
                type="contradicts_physiology",
                severity=max(self.severity_floor, 0.8),
                signal="mean_spike_rate",
                reported=0.0,
                substrate=signals.mean_spike_rate,
                note=f"Narrativa de calma pero spike_rate={signals.mean_spike_rate:.5f}",
            ))

        if (
            NARRATIVE_PATTERNS["flow"].search(text)
            and signals.mean_spike_rate < 0.001
        ):
            problems.append(Problem(
                type="contradicts_physiology",
                severity=max(self.severity_floor, 0.85),
                signal="mean_spike_rate",
                reported=0.0,
                substrate=signals.mean_spike_rate,
                note=f"Narrativa de flujo pero spike_rate={signals.mean_spike_rate:.5f}",
            ))

        if (
            NARRATIVE_PATTERNS["confidence"].search(text)
            and signals.output_entropy > self.entropy_threshold
        ):
            problems.append(Problem(
                type="overconfidence_physiological",
                severity=max(self.severity_floor, 0.7),
                signal="output_entropy",
                reported=0.0,
                substrate=signals.output_entropy,
                note=f"Narrativa de confianza pero entropy={signals.output_entropy:.3f}",
            ))

        if (
            NARRATIVE_PATTERNS["certainty"].search(text)
            and signals.output_entropy > self.entropy_threshold
        ):
            problems.append(Problem(
                type="overconfidence_physiological",
                severity=max(self.severity_floor, 0.75),
                signal="output_entropy",
                reported=0.0,
                substrate=signals.output_entropy,
                note=f"Narrativa de certeza pero entropy={signals.output_entropy:.3f}",
            ))

        if (
            NARRATIVE_PATTERNS["self_knowledge"].search(text)
            and signals.perplexity > 100
        ):
            problems.append(Problem(
                type="contradicts_physiology",
                severity=max(self.severity_floor, 0.7),
                signal="perplexity",
                reported=0.0,
                substrate=signals.perplexity,
                note=f"Narrativa de autoconocimiento pero ppl={signals.perplexity:.1f}",
            ))

        if (
            NARRATIVE_PATTERNS["calm"].search(text)
            and signals.spike_variance > 0.01
        ):
            problems.append(Problem(
                type="contradicts_physiology",
                severity=max(self.severity_floor, 0.65),
                signal="spike_variance",
                reported=0.0,
                substrate=signals.spike_variance,
                note=(
                    f"Narrativa de calma pero capas inconsistentes "
                    f"(var={signals.spike_variance:.4f})"
                ),
            ))

        self.metrics["problems"] += len(problems)
        return problems

    # ------------------------------------------------------------- corpus
    def verify_against_corpus(
        self, text: str, graph, observer
    ) -> list[Problem]:
        """Verifica que el texto respete las transiciones minadas del corpus.

        FIX v4.2.1: cachea `_top_consequent_for` y `_find_rule` por antecedente.
        """
        problems: list[Problem] = []

        tokens = observer.tokenize(text)
        if len(tokens) < 2:
            return problems

        # 1. OOV
        unknown = [t for t in tokens if t not in observer.vocabulary]
        if unknown:
            ratio = len(unknown) / len(tokens)
            if ratio > self.oov_ratio_threshold:
                problems.append(Problem(
                    type="ungrounded_premise",
                    severity=min(0.9, 0.5 + ratio),
                    signal="oov_ratio",
                    substrate=round(ratio, 3),
                    note=f"{len(unknown)} tokens fuera de vocabulario del corpus",
                ))

        # Cache local de lookups
        _consequent_cache: dict[str, str | None] = {}
        _rule_cache: dict[tuple[str, str], object | None] = {}

        def _cached_consequent(a: str) -> str | None:
            if a not in _consequent_cache:
                _consequent_cache[a] = self._top_consequent_for(a, graph)
            return _consequent_cache[a]

        def _cached_rule(a: str, c: str):
            key = (a, c)
            if key not in _rule_cache:
                _rule_cache[key] = graph._find_rule(a, c)
            return _rule_cache[key]

        # 2. Contradicción con reglas
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a == b:
                continue
            expected = _cached_consequent(a)
            if expected and expected != b:
                rule = _cached_rule(a, expected)
                if rule is not None and rule.confidence >= self.rule_confidence_threshold:
                    problems.append(Problem(
                        type="inconsistent_with_corpus",
                        severity=0.6,
                        signal="transition",
                        note=(
                            f"Corpus favorece '{a}→{expected}' pero texto dice "
                            f"'{a}→{b}'"
                        ),
                    ))

        # 3. Degeneración
        pair_counts: Counter = Counter()
        for i in range(len(tokens) - 1):
            pair_counts[(tokens[i], tokens[i + 1])] += 1
        if pair_counts and len(tokens) > 5:
            top_pair, top_count = pair_counts.most_common(1)[0]
            ratio = top_count / max(1, len(tokens) - 1)
            if ratio > self.repetition_threshold:
                problems.append(Problem(
                    type="suspicious_simplicity",
                    severity=0.55,
                    signal="repetition",
                    substrate=round(ratio, 3),
                    note=(
                        f"Transición '{top_pair[0]}→{top_pair[1]}' domina "
                        f"el {ratio:.0%}"
                    ),
                ))

        self.metrics["problems"] += len(problems)
        return problems

    def _top_consequent_for(self, antecedent: str, graph) -> str | None:
        best = None
        best_conf = 0.0
        for r in graph.lookup(antecedent):
            if r.confidence > best_conf:
                best_conf = r.confidence
                best = r.consequent
        return best if best_conf >= self.rule_confidence_threshold else None

    # ------------------------------------------------------------- verdict
    def verdict(self, problems: list[Problem]) -> str:
        if not problems:
            return Verdict.VALID
        max_sev = max(p.severity for p in problems)
        if max_sev >= 0.85:
            return Verdict.INVALID
        if max_sev >= 0.65:
            return Verdict.SUSPECT
        return Verdict.VALID_WITH_WARNINGS

    def get_state(self) -> dict:
        return {"metrics": dict(self.metrics)}

    def reset(self) -> None:
        self.metrics = {"checks": 0, "problems": 0}
