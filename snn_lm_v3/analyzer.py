"""Analizador continuo en background para SNN-LM v4.1.

Tareas (round-robin):
  1. Podar memoria neural (limpiar NaNs, recompactar).
  2. Refrescar codebooks: abstraction_head Y resonant_vocab (si RT).
  3. Tareas extra del usuario.

Todo bajo torch.no_grad(). Hilo daemon.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import torch


logger = logging.getLogger(__name__)


def _unwrap(model):
    seen = set()
    while hasattr(model, "_orig_mod") and id(model) not in seen:
        seen.add(id(model))
        model = model._orig_mod
    return model


def _get_rt_vocab(model):
    """Devuelve model.rv si es RTModelWrapper, else None."""
    try:
        from .tokenizer import RTModelWrapper
        if isinstance(model, RTModelWrapper):
            return model.rv
    except ImportError:
        pass
    return None


def _get_abstraction_head(model):
    """Devuelve abstraction_head del SNNLM subyacente (o None)."""
    try:
        from .tokenizer import RTModelWrapper
        if isinstance(model, RTModelWrapper):
            return getattr(model.snnlm, "abstraction_head", None)
    except ImportError:
        pass
    return getattr(model, "abstraction_head", None)


class ContinuousAnalyzer:
    """Motor de análisis continuo en background.

    Rotación de 3 fases:
      0 -> prune_memory
      1 -> refresh_codebooks (abstraction + RT)
      2 -> extra_tasks
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

        self.model = _unwrap(model)
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
            "dead_codes_reinit_abs_total": 0,
            "dead_codes_reinit_rt_total": 0,
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
        logger.info(f"ContinuousAnalyzer started (interval={self.interval_s}s)")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("ContinuousAnalyzer stopped")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_once(self) -> dict:
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
            result["codebooks"] = self._refresh_codebooks()
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
        mem = getattr(self.model, "neural_memory", None)
        if mem is None:
            return {"pruned": 0, "reason": "no_memory"}

        with torch.no_grad():
            n = int(mem.size.item())
            if n == 0:
                return {"pruned": 0, "size": 0}
            active_keys = mem.keys if n == mem.max_size else mem.keys[:n]

            finite_mask = torch.isfinite(active_keys).all(dim=-1)
            n_invalid = int((~finite_mask).sum().item())
            if n_invalid > 0:
                keep = active_keys[finite_mask]
                n_keep = keep.shape[0]
                mem.keys[:n_keep] = keep
                mem.values[:n_keep] = keep
                mem.keys[n_keep:].zero_()
                mem.values[n_keep:].zero_()
                mem.size.fill_(n_keep)
                mem.ptr.fill_(0 if n_keep == mem.max_size else n_keep)

            return {"pruned": n_invalid, "size": int(mem.size.item())}

    def _refresh_codebooks(self) -> dict:
        """Reinicializa códigos muertos en abstraction_head Y resonant_vocab."""
        result = {"reinit_abs": 0, "reinit_rt": 0}
        abs_head = _get_abstraction_head(self.model)
        rv = _get_rt_vocab(self.model)

        with torch.no_grad():
            if abs_head is not None:
                n = abs_head.reinit_dead_codes()
                result["reinit_abs"] = n
                self.metrics["dead_codes_reinit_abs_total"] += n
            if rv is not None:
                n = rv.reinit_dead_codes()
                result["reinit_rt"] = n
                self.metrics["dead_codes_reinit_rt_total"] += n

        if abs_head is None and rv is None:
            result["reason"] = "no_codebook"
        return result
