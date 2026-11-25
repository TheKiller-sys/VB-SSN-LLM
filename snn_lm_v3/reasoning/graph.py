"""Grafo de razonamiento simbólico sobre corpus.

Portado de SymbolicAbstractionLayer.js v2.1:
  - anti-ciclo (_would_create_cycle)
  - cooldown por regla (adaptado: cooldown en pasos, no en ms)
  - tautología + overlap filtering
  - pruning por confianza × uso × recencia
  - decay de reglas no usadas
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable


MAX_RULES = 3000
RULE_PRUNE_TRIGGER_RATIO = 1.2
RULE_PRUNE_KEEP_RATIO = 0.6
MIN_RULE_CONFIDENCE = 0.5
RULE_DECAY_RATE = 0.01
RULE_DECAY_THRESHOLD = 0.2
MAX_EVIDENCE_PER_RULE = 50
MAX_COUNTEREXAMPLES_PER_RULE = 20
DEFAULT_COOLDOWN_STEPS = 3


@dataclass
class Rule:
    id: str
    antecedent: str
    consequent: str
    confidence: float = 0.4
    evidence: list = field(default_factory=list)
    counterexamples: list = field(default_factory=list)
    uses: int = 0
    created_at: float = field(default_factory=time.time)
    last_used_at: float = 0.0
    last_used_step: int = -10**9
    source: str = "signals"

    def reinforce(self, evidence: dict) -> None:
        self.evidence.append({**evidence, "at": time.time()})
        if len(self.evidence) > MAX_EVIDENCE_PER_RULE:
            self.evidence.pop(0)
        self._recompute()

    def contradict(self, counterexample: dict) -> None:
        self.counterexamples.append({**counterexample, "at": time.time()})
        if len(self.counterexamples) > MAX_COUNTEREXAMPLES_PER_RULE:
            self.counterexamples.pop(0)
        self._recompute()

    def _recompute(self) -> None:
        pos = len(self.evidence)
        neg = len(self.counterexamples)
        self.confidence = (pos + 1) / (pos + neg + 2)

    def decay(self, dt_hours: float) -> None:
        age_h = (time.time() - self.created_at) / 3600
        if age_h > 12 and self.uses < 3:
            self.confidence = max(0.0, self.confidence - RULE_DECAY_RATE * dt_hours)

    @property
    def is_dead(self) -> bool:
        return self.confidence < RULE_DECAY_THRESHOLD and self.uses == 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "confidence": round(self.confidence, 4),
            "uses": self.uses,
            "evidence": len(self.evidence),
            "counterexamples": len(self.counterexamples),
        }

    def to_persistence(self) -> dict:
        return {
            "id": self.id,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
            "confidence": self.confidence,
            "evidence": self.evidence[-MAX_EVIDENCE_PER_RULE:],
            "counterexamples": self.counterexamples[-MAX_COUNTEREXAMPLES_PER_RULE:],
            "uses": self.uses,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "last_used_step": self.last_used_step,
            "source": self.source,
        }

    @classmethod
    def from_persistence(cls, data: dict) -> "Rule | None":
        if not data or not data.get("id"):
            return None
        r = cls(
            id=data["id"],
            antecedent=data["antecedent"],
            consequent=data["consequent"],
        )
        r.confidence = data.get("confidence", 0.4)
        r.evidence = data.get("evidence", [])
        r.counterexamples = data.get("counterexamples", [])
        r.uses = data.get("uses", 0)
        r.created_at = data.get("created_at", time.time())
        r.last_used_at = data.get("last_used_at", 0.0)
        r.last_used_step = data.get("last_used_step", -10**9)
        r.source = data.get("source", "signals")
        return r


class ReasoningGraph:
    """Motor de reglas con anti-ciclo, cooldown y pruning.

    Uso:
        graph = ReasoningGraph()
        n = graph.mine_from_corpus(observer)
        graph.tick()
        consequences = graph.apply(facts)
    """

    def __init__(self, cooldown_steps: int = DEFAULT_COOLDOWN_STEPS):
        self.rules: dict[str, Rule] = {}
        self.by_antecedent: dict[str, list[str]] = {}
        self._symbols: dict[str, int] = {}
        self.cooldown_steps = cooldown_steps
        self._step = 0
        self._last_decay_at = time.time()
        self.metrics = {
            "rules_created": 0,
            "rules_pruned": 0,
            "rules_applied": 0,
            "cyclic_rejected": 0,
            "tautological_rejected": 0,
            "overlap_rejected": 0,
            "blocked_by_cooldown": 0,
            "facts_observed": 0,
        }

    # ------------------------------------------------------------- reloj
    def tick(self, n: int = 1) -> None:
        """Avanza el reloj interno del grafo (para cooldown)."""
        if n < 0:
            raise ValueError("tick n must be >= 0")
        self._step += int(n)

    # ------------------------------------------------------------- utilidades
    @staticmethod
    def _rule_key(a: str) -> str:
        """Clave canónica: strings multipalabra se ordenan e ignoran mayúsculas
        para que "hello world" y "world hello" mapeen al mismo bucket.
        Tuplas/sets conservan el comportamiento original."""
        if isinstance(a, str):
            return " ".join(sorted(a.lower().split()))
        return "+".join(sorted(a))

    def _same_consequent(self, a, b) -> bool:
        return str(a).lower() == str(b).lower()

    def lookup(self, antecedent: str) -> list[Rule]:
        """Devuelve las reglas cuyo antecedente coincide (vía `_rule_key`)."""
        key = self._rule_key(antecedent)
        out: list[Rule] = []
        for rid in self.by_antecedent.get(key, []):
            r = self.rules.get(rid)
            if r is not None:
                out.append(r)
        return out

    def _find_rule(self, a: str, c: str) -> Rule | None:
        for r in self.lookup(a):
            if self._same_consequent(r.consequent, c):
                return r
        return None

    def _is_tautology(self, a: str, c: str) -> bool:
        return a.lower() == c.lower()

    def _has_overlap(self, a: str, c: str) -> bool:
        a_tokens = set(str(a).lower().split())
        c_tokens = set(str(c).lower().split())
        return bool(a_tokens & c_tokens)

    def _would_create_cycle(self, a: str, c: str) -> bool:
        return self._find_rule(c, a) is not None

    # ------------------------------------------------------------- creación
    def create_rule(
        self, antecedent: str, consequent: str, evidence: dict | None = None
    ) -> Rule | None:
        """Crea una regla. NO avanza el reloj: el cooldown sólo debe consumirse
        con `tick()` o con `apply()`."""
        if self._is_tautology(antecedent, consequent):
            self.metrics["tautological_rejected"] += 1
            return None
        if self._has_overlap(antecedent, consequent):
            self.metrics["overlap_rejected"] += 1
            return None
        if self._would_create_cycle(antecedent, consequent):
            self.metrics["cyclic_rejected"] += 1
            return None

        if len(self.rules) >= MAX_RULES:
            # Al alcanzar el tope hay que forzar prune, porque el trigger
            # normal (1.2 × MAX_RULES) es inalcanzable cuando MAX_RULES es cap.
            self.prune(force=True)
            if len(self.rules) >= MAX_RULES:
                return None

        rid = f"r_{int(time.time()*1000)}_{len(self.rules)}"
        rule = Rule(id=rid, antecedent=antecedent, consequent=consequent)
        if evidence:
            rule.reinforce(evidence)
        # Asegurar que pasa el umbral tras un refuerzo inicial
        if rule.confidence < MIN_RULE_CONFIDENCE:
            if evidence:
                rule.reinforce(evidence)
            if rule.confidence < MIN_RULE_CONFIDENCE:
                return None

        self.rules[rid] = rule
        self.by_antecedent.setdefault(self._rule_key(antecedent), []).append(rid)
        self.metrics["rules_created"] += 1
        return rule

    # ------------------------------------------------------------- aplicación
    def apply(self, facts: Iterable[str]) -> list[dict]:
        """Aplica reglas que matchean, respetando cooldown por paso."""
        facts = list(facts)
        consequences: list[dict] = []
        seen: set[str] = set()

        for fact in facts:
            for rule in self.lookup(fact):
                if rule.confidence < MIN_RULE_CONFIDENCE:
                    continue
                if rule.id in seen:
                    continue
                if self._step - rule.last_used_step < self.cooldown_steps:
                    self.metrics["blocked_by_cooldown"] += 1
                    continue

                seen.add(rule.id)
                rule.uses += 1
                rule.last_used_at = time.time()
                rule.last_used_step = self._step
                self.metrics["rules_applied"] += 1

                consequences.append({
                    "rule_id": rule.id,
                    "antecedent": rule.antecedent,
                    "consequent": rule.consequent,
                    "confidence": round(rule.confidence, 4),
                })

        return sorted(consequences, key=lambda c: -c["confidence"])

    # ------------------------------------------------------------- mining
    def mine_from_corpus(self, observer) -> int:
        """Mina reglas desde un CorpusObserver."""
        new_rules = 0

        for token, count in observer.candidate_symbols():
            self._ensure_symbol(token, count)

        for a, b, count in observer.candidate_transitions():
            existing = self._find_rule(a, b)
            if existing:
                existing.reinforce({"source": "corpus", "count": count})
                continue
            if self.create_rule(a, b, {"source": "corpus", "count": count}):
                new_rules += 1

        return new_rules

    def mine_from_log(self, log: list[dict]) -> int:
        """Mina reglas desde un log de observaciones.

        Cada entrada: {at, facts: list[str]}. Pares consecutivos con delay
        razonable refuerzan o crean reglas.
        """
        if len(log) < 6:
            return 0

        new_rules = 0
        for i in range(len(log) - 1):
            a_entry = log[i]
            b_entry = log[i + 1]
            if not a_entry.get("facts") or not b_entry.get("facts"):
                continue
            if b_entry["at"] - a_entry["at"] > 60:
                continue
            for fa in a_entry["facts"]:
                for fb in b_entry["facts"]:
                    if fa == fb:
                        continue
                    existing = self._find_rule(fa, fb)
                    if existing:
                        existing.reinforce({"source": "log"})
                        continue
                    if self.create_rule(fa, fb, {"source": "log"}):
                        new_rules += 1
        return new_rules

    def _ensure_symbol(self, token: str, count: int) -> None:
        self._symbols[token] = count

    # ------------------------------------------------------------- pruning
    def prune(self, force: bool = False) -> int:
        trigger = int(MAX_RULES * RULE_PRUNE_TRIGGER_RATIO)
        if not force and len(self.rules) < trigger:
            return 0

        keep_n = int(MAX_RULES * RULE_PRUNE_KEEP_RATIO)
        # Si se fuerza el prune en un grafo más pequeño que keep_n, reducir
        # el objetivo a una fracción del tamaño actual para que pode de verdad.
        if force and len(self.rules) <= keep_n:
            keep_n = max(1, int(len(self.rules) * RULE_PRUNE_KEEP_RATIO))

        now = time.time()

        scored = []
        for r in self.rules.values():
            age_h = (now - r.created_at) / 3600
            recency = math.exp(-age_h / 48)
            score = r.confidence * math.log2(r.uses + 1) * recency
            scored.append((score, r.id))

        scored.sort(reverse=True)
        keep = {rid for _, rid in scored[:keep_n]}

        removed = 0
        for rid in list(self.rules.keys()):
            if rid in keep:
                continue
            r = self.rules.pop(rid)
            lst = self.by_antecedent.get(self._rule_key(r.antecedent), [])
            if rid in lst:
                lst.remove(rid)
            removed += 1

        self.metrics["rules_pruned"] += removed
        return removed

    def decay(self) -> None:
        now = time.time()
        dt_h = (now - self._last_decay_at) / 3600
        if dt_h < 0.01:
            return
        self._last_decay_at = now
        for r in list(self.rules.values()):
            r.decay(dt_h * 24)
            if r.is_dead:
                self.rules.pop(r.id, None)
                lst = self.by_antecedent.get(self._rule_key(r.antecedent), [])
                if r.id in lst:
                    lst.remove(r.id)

    # ------------------------------------------------------------- estado
    def get_state(self) -> dict:
        return {
            "rules_count": len(self.rules),
            "strong_rules": sum(
                1 for r in self.rules.values() if r.confidence >= 0.7
            ),
            "symbols_tracked": len(self._symbols),
            "step": self._step,
            "cooldown_steps": self.cooldown_steps,
            "metrics": dict(self.metrics),
            "top_rules": [
                r.to_dict()
                for r in sorted(
                    self.rules.values(), key=lambda r: -r.confidence
                )[:5]
            ],
        }

    def reset(self) -> None:
        self.rules = {}
        self.by_antecedent = {}
        self._symbols = {}
        self._step = 0
        self._last_decay_at = time.time()
        self.metrics = {
            "rules_created": 0,
            "rules_pruned": 0,
            "rules_applied": 0,
            "cyclic_rejected": 0,
            "tautological_rejected": 0,
            "overlap_rejected": 0,
            "blocked_by_cooldown": 0,
            "facts_observed": 0,
        }
