import threading
from pathlib import Path
from typing import Optional

from snn_lm_v3 import SNNLM, load_checkpoint, resolve_device
from snn_lm_v3.generate import generate as _generate


class ModelManager:
    def __init__(self):
        self._model = None
        self._config = None
        self._ckpt_path: Optional[str] = None
        self._device = "auto"
        self._lock = threading.RLock()
        self._reasoning_bias = None
        self._rag_kb = None

    def load(self, ckpt_path: str, device: str = "auto") -> dict:
        with self._lock:
            res = load_checkpoint(ckpt_path, SNNLM, device)
            self._model = res["model"]
            self._config = res["config"]
            self._ckpt_path = ckpt_path
            self._device = str(resolve_device(device))
            return self.info()

    def info(self) -> dict:
        if self._model is None:
            return {"loaded": False}
        return {
            "loaded": True,
            "ckpt": self._ckpt_path,
            "device": self._device,
            "params": self._model.num_params(),
            "core_type": self._config.core_type,
            "ffn_mode": self._config.ffn_mode,
            "vocab_size": self._config.vocab_size,
            "seq_len": self._config.seq_len,
            "rt": bool(self._config.use_rt),
            "memory_size": (
                int(self._model.neural_memory.size.item())
                if getattr(self._model, "neural_memory", None) else 0
            ),
        }

    def generate(self, prompt: str, **kw) -> str:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model loaded. Load a checkpoint first.")
            return _generate(
                self._model, prompt=prompt,
                rag_kb=self._rag_kb,
                reasoning_bias=self._reasoning_bias,
                auto_reasoning_bias=False,
                **kw,
            )

    def set_rag(self, kb):        # None para desactivar
        with self._lock:
            self._rag_kb = kb

    def set_reasoning_bias(self, bias):
        with self._lock:
            self._reasoning_bias = bias

    def promote(self, ckpt_path: str) -> dict:
        """Hot-swap atómico: carga el nuevo y solo reemplaza si OK."""
        new_info = self.load(ckpt_path, self._device)
        return new_info
