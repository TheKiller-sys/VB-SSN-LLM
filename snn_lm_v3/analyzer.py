"""Analizador continuo en background para SNN-LM v4.0.

Corre en un thread daemon y ejecuta tareas periódicas que NO tocan el
transformer: consolidar memoria, refrescar codebook, podar, detectar
conflictos. Todo ocurre bajo `torch.no_grad()`.

Uso:
    analyzer = ContinuousAnalyzer(model, cfg)
    analyzer.start()
    ...
    analyzer.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import torch


logger = logging.getLogger(__name__)


class ContinuousAnalyzer:
    """Motor de análisis continuo en background.

    Tareas (rotación round-robin):
      1. Podar memoria (si `prune_memory`).
      2. Refrescar codebook (reinit códigos muertos).
      3. (extensible) Detectar conflictos en ReasoningGraph externo.
      4. (extensible) Refresh de SelfKnowledge.
    """

    def __init__(
        self,
        model,
        interval_s: float = 60.0,
        prune_memory: bool = True,
        refresh_codebook: bool = True,
        extra_tasks: Optional[list[Callable[[], dict]]] = None,
    ):
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")

        self.model = model
        self.interval_s = interval_s
        self.prune_memory = prune_memory
        self.refresh_codebook = refresh_codebook
        self.extra_tasks = extra_tasks or []

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._phase = 0
        self.metrics = {
            "runs": 0,
            "prune_runs": 0,
            "codebook_refresh_runs": 0,
            "extra_task_runs": 0,
            "errors": 0,
            "last_run_at": 0.0,
            "dead_codes_reinit_total": 0,
        }

    # ------------------------------------------------------- public
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="snnlm-analyzer", daemon=True,
        )
        self._thread.start()
        logger.info(
            f"ContinuousAnalyzer started (interval={self.interval_s}s)"
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("ContinuousAnalyzer stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> dict:
        """Ejecuta una pasada síncrona (útil para tests)."""
        return self._run_phase()

    def get_state(self) -> dict:
        return {
            "running": self.is_running(),
            "interval_s": self.interval_s,
            "metrics": dict(self.metrics),
            "phase": self._phase,
        }

    # ------------------------------------------------------- internals
    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self._run_phase()
            except Exception as e:
                self.metrics["errors"] += 1
                logger.warning(f"ContinuousAnalyzer error: {e}")

    def _run_phase(self) -> dict:
        self.metrics["runs"] += 1
        self.metrics["last_run_at"] = time.time()

        task_idx = self._phase % 3
        self._phase += 1
        result: dict = {}

        if task_idx == 0 and self.prune_memory:
            result["prune"] = self._prune_memory()
            self.metrics["prune_runs"] += 1
        elif task_idx == 1 and self.refresh_codebook:
            result["codebook"] = self._refresh_codebook()
            self.metrics["codebook_refresh_runs"] += 1
        elif task_idx == 2 and self.extra_tasks:
            for i, fn in enumerate(self.extra_tasks):
                try:
                    result[f"extra_{i}"] = fn()
                except Exception as e:
                    result[f"extra_{i}_error"] = str(e)
            self.metrics["extra_task_runs"] += 1

        return result

    def _prune_memory(self) -> dict:
        """Poda la memoria neural: recorta si excede max_size (ya lo hace
        el ring buffer) y compacta slots no usados.

        Nota: el ring buffer ya implementa FIFO por tamaño, así que aquí
        sólo hacemos limpieza de ceros / NaNs.
        """
        mem = getattr(self.model, "neural_memory", None)
        if mem is None:
            return {"pruned": 0, "reason": "no_memory"}

        with torch.no_grad():
            n = int(mem.size.item())
            if n == 0:
                return {"pruned": 0, "size": 0}

            if n == mem.max_size:
                active_keys = mem.keys
            else:
                active_keys = mem.keys[:n]

            # Filtrar filas no finitas (no deberían existir)
            finite_mask = torch.isfinite(active_keys).all(dim=-1)
            n_invalid = int((~finite_mask).sum().item())
            if n_invalid > 0:
                # Recompactar in-place
                keep = active_keys[finite_mask]
                n_keep = keep.shape[0]
                mem.keys[:n_keep] = keep
                mem.values[:n_keep] = keep
                mem.keys[n_keep:].zero_()
                mem.values[n_keep:].zero_()
                mem.size.fill_(n_keep)
                mem.ptr.fill_(0 if n_keep == mem.max_size else n_keep)

            return {"pruned": n_invalid, "size": int(mem.size.item())}

    def _refresh_codebook(self) -> dict:
        """Reinicializa códigos muertos del codebook."""
        abs_head = getattr(self.model, "abstraction_head", None)
        if abs_head is None:
            return {"reinit": 0, "reason": "no_abstraction"}

        with torch.no_grad():
            n_dead = abs_head.reinit_dead_codes()
        self.metrics["dead_codes_reinit_total"] += n_dead
        return {"reinit": n_dead}
