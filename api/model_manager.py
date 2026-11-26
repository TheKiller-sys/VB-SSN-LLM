from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from snn_lm_v4 import (
    SNNLM, load_checkpoint, resolve_device,
    SignalProbe, ConsistencyVerifier, ReasoningBias,
    TokenSymbolMapper, load_graph, KnowledgeBase,
    make_model_encoder, make_sentence_transformer_encoder,
)
from snn_lm_v4.generate import generate as _generate


class ModelManager:
    def __init__(self):
        self._model = None
        self._config = None
        self._ckpt_path: Optional[str] = None
        self._device = "auto"
        self._lock = threading.RLock()
        self._reasoning_bias = None
        self._rag_kb = None
        self._graph = None

    # ---------------------------------------------------------- load
    def load(self, ckpt_path: str, device: str = "auto") -> dict:
        with self._lock:
            res = load_checkpoint(ckpt_path, SNNLM, device)
            self._model = res["model"]
            self._config = res["config"]
            self._ckpt_path = ckpt_path
            self._device = str(resolve_device(device))
            self._model.eval()
            return self.info()

    def promote(self, ckpt_path: str) -> dict:
        return self.load(ckpt_path, self._device)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._config = None
            self._ckpt_path = None
            self._reasoning_bias = None
            self._rag_kb = None
            self._graph = None

    # ---------------------------------------------------------- info
    def info(self) -> dict:
        with self._lock:
            if self._model is None:
                return {"loaded": False}
            m = self._model
            return {
                "loaded": True,
                "ckpt": self._ckpt_path,
                "device": self._device,
                "params": m.num_params(),
                "core_type": self._config.core_type,
                "ffn_mode": self._config.ffn_mode,
                "vocab_size": self._config.vocab_size,
                "seq_len": self._config.seq_len,
                "n_layer": self._config.n_layer,
                "n_head": self._config.n_head,
                "n_embd": self._config.n_embd,
                "rt": bool(self._config.use_rt),
                "snn_mode": self._config.snn_mode,
                "memory_size": (
                    int(m.neural_memory.size.item())
                    if getattr(m, "neural_memory", None) else 0
                ),
                "has_abstraction": bool(getattr(m, "abstraction_head", None)),
                "has_self_knowledge": bool(getattr(m, "self_knowledge_head", None)),
                "rag_active": self._rag_kb is not None,
                "reasoning_bias_active": self._reasoning_bias is not None,
            }

    def knowledge(self) -> dict:
        with self._lock:
            if self._model is None:
                return {"loaded": False}
            m = self._model
            if hasattr(m, "get_knowledge_summary"):
                return m.get_knowledge_summary()
            return {"enabled": False}

    # ---------------------------------------------------------- generate
    def generate(self, prompt: str, **kw) -> str:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model loaded.")
            return _generate(
                self._model, prompt=prompt,
                rag_kb=self._rag_kb,
                reasoning_bias=self._reasoning_bias,
                auto_reasoning_bias=kw.pop("auto_reasoning_bias", False),
                **kw,
            )

    def verify(self, prompt: str, max_new_tokens: int = 200, **gen_kw) -> dict:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model loaded.")
            text = _generate(
                self._model, prompt=prompt,
                max_new_tokens=max_new_tokens,
                rag_kb=self._rag_kb,
                reasoning_bias=self._reasoning_bias,
                auto_reasoning_bias=False,
                **gen_kw,
            )
            problems = []
            try:
                probe = SignalProbe(self._model, self._device)
                signals = probe.probe(text)
                verifier = ConsistencyVerifier()
                problems.extend(verifier.verify(text, signals))
                if self._graph is not None:
                    from snn_lm_v4 import CorpusObserver
                    obs = CorpusObserver()
                    obs.observe(prompt, ref="prompt")
                    problems.extend(
                        verifier.verify_against_corpus(text, self._graph, obs)
                    )
                verdict = verifier.verdict(problems)
            except Exception as e:
                signals = None
                verdict = f"error: {e}"
            return {
                "text": text,
                "verdict": verdict,
                "signals": signals.to_dict() if signals else None,
                "problems": [p.to_dict() for p in problems],
            }

    # ---------------------------------------------------------- RAG
    def set_rag(self, kb) -> None:
        with self._lock:
            self._rag_kb = kb

    def clear_rag(self) -> None:
        self.set_rag(None)

    def load_rag_from_path(self, path: str, encoder_kind: str = "hidden") -> int:
        with self._lock:
            if encoder_kind == "hidden":
                if self._model is None:
                    raise RuntimeError("Load a model first for hidden encoder.")
                encoder = make_model_encoder(self._model)
            elif encoder_kind.startswith("sentence_transformer:"):
                name = encoder_kind.split(":", 1)[1]
                encoder = make_sentence_transformer_encoder(name)
            else:
                raise ValueError(f"Unknown encoder: {encoder_kind}")
            kb = KnowledgeBase.load(path, encoder_fn=encoder)
            self._rag_kb = kb
            return len(kb)

    def rag_search(self, query: str, k: int = 4) -> list[dict]:
        with self._lock:
            if self._rag_kb is None:
                raise RuntimeError("No RAG KB loaded.")
            return [
                {"text": t, "score": s}
                for t, s in self._rag_kb.search(query, k=k)
            ]

    # ---------------------------------------------------------- reasoning
    def set_reasoning_bias(self, bias) -> None:
        with self._lock:
            self._reasoning_bias = bias

    def load_reasoning_from_paths(
        self, graph_path: str, mapper_path: str,
        strength: float = 0.1, top_k: int = 16,
        remine_every: int = 8, facts_window: int = 8,
        min_confidence: float = 0.5,
    ) -> dict:
        with self._lock:
            graph = load_graph(graph_path)
            mapper = TokenSymbolMapper.load(mapper_path)
            bias = ReasoningBias(
                graph=graph, mapper=mapper, strength=strength,
                top_k=top_k, remine_every=remine_every,
                facts_window=facts_window, min_confidence=min_confidence,
            )
            self._graph = graph
            self._reasoning_bias = bias
            return bias.get_state()

    def clear_reasoning(self) -> None:
        with self._lock:
            self._reasoning_bias = None
            self._graph = None

    # ---------------------------------------------------------- memory
    def consolidate(self) -> dict:
        with self._lock:
            if self._model is None:
                raise RuntimeError("No model loaded.")
            if hasattr(self._model, "consolidate"):
                return self._model.consolidate()
            return {}

    def reset_memory(self) -> None:
        with self._lock:
            if self._model is None:
                return
            if hasattr(self._model, "reset_memory"):
                self._model.reset_memory()
