"""ReasoningBias: sesga logits durante decoding con reglas del ReasoningGraph.

Diseño:
  - `facts` se extraen de los últimos `facts_window` tokens generados.
  - `graph.apply(facts)` devuelve consecuencias (1 salto).
  - `TokenSymbolMapper` traduce consecuente -> token_id.
  - Bias additivo: `logits[token_id] += strength * confidence`.

Caching:
  - Se re-mina (graph.apply) cada `remine_every` tokens para reducir coste.
  - Se mantiene una ventana de consecuencias activas entre re-minados.

Thread safety: no requerido (se usa en decoding single-thread).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from .symbol_mapping import TokenSymbolMapper

logger = logging.getLogger(__name__)


@dataclass
class ReasoningBiasState:
    """Estado mutable entre pasos de decoding."""
    token_buffer: deque = field(default_factory=lambda: deque(maxlen=64))
    active_consequences: list = field(default_factory=list)
    last_mine_step: int = -10**9
    total_applied: int = 0
    total_recomputes: int = 0


class ReasoningBias:
    """Aplica bias de reglas a logits durante decoding."""

    def __init__(
        self,
        graph,
        mapper: TokenSymbolMapper,
        strength: float = 0.1,
        top_k: int = 16,
        remine_every: int = 8,
        facts_window: int = 8,
        min_confidence: float = 0.5,
    ):
        if strength < 0:
            raise ValueError("strength must be >= 0")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if remine_every < 1:
            raise ValueError("remine_every must be >= 1")
        if facts_window < 1:
            raise ValueError("facts_window must be >= 1")

        self.graph = graph
        self.mapper = mapper
        self.strength = float(strength)
        self.top_k = int(top_k)
        self.remine_every = int(remine_every)
        self.facts_window = int(facts_window)
        self.min_confidence = float(min_confidence)

        self.state = ReasoningBiasState()

    # --------------------------------------------------------- internals
    def _token_to_symbols(self, token_ids: list[int]) -> list[str]:
        symbols: list[str] = []
        for tid in token_ids:
            for sym in self.mapper.token_to_symbols.get(int(tid), ()):
                symbols.append(sym)
        return symbols

    def _mine(self) -> None:
        """Recomputa consecuencias activas desde el buffer."""
        recent = list(self.state.token_buffer)[-self.facts_window * 2:]
        facts = self._token_to_symbols(recent)
        if not facts:
            self.state.active_consequences = []
            self.state.total_recomputes += 1
            return

        try:
            consequences = self.graph.apply(facts)
        except Exception as e:
            logger.debug(f"graph.apply failed: {e}")
            consequences = []

        # Filtrar por confianza mínima
        filtered = [
            c for c in consequences
            if float(c.get("confidence", 0.0)) >= self.min_confidence
        ]
        self.state.active_consequences = filtered
        self.state.total_recomputes += 1

    # --------------------------------------------------------- api
    def reset(self) -> None:
        self.state = ReasoningBiasState()

    def observe_token(self, token_id: int) -> None:
        """Registra un token generado en el buffer."""
        self.state.token_buffer.append(int(token_id))

    def maybe_remine(self, step: int, force: bool = False) -> None:
        """Re-mina si toca (cada remine_every pasos)."""
        if force or step - self.state.last_mine_step >= self.remine_every:
            self._mine()
            self.state.last_mine_step = step

    def apply(
        self, logits: torch.Tensor, step: int, force_remine: bool = False,
    ) -> int:
        """Aplica bias a logits in-place. Devuelve nº de consecuencias aplicadas.

        Args:
            logits: (..., vocab_size) — típicamente (1, vocab).
            step: contador de paso (para remine scheduling).
            force_remine: si True, re-mina antes de aplicar.
        """
        if self.strength <= 0.0:
            return 0

        self.maybe_remine(step, force=force_remine)
        if not self.state.active_consequences:
            return 0

        applied = self.mapper.bias_logits_for_consequences(
            logits,
            self.state.active_consequences,
            strength=self.strength,
            top_k=self.top_k,
        )
        self.state.total_applied += applied
        return applied

    # --------------------------------------------------------- reporting
    def get_state(self) -> dict:
        return {
            "strength": self.strength,
            "top_k": self.top_k,
            "remine_every": self.remine_every,
            "facts_window": self.facts_window,
            "buffer_size": len(self.state.token_buffer),
            "active_consequences": len(self.state.active_consequences),
            "total_applied": self.state.total_applied,
            "total_recomputes": self.state.total_recomputes,
        }

    def extra_repr(self) -> str:
        return (
            f"strength={self.strength}, top_k={self.top_k}, "
            f"remine_every={self.remine_every}, "
            f"mapper_symbols={len(self.mapper)}"
        )
