"""Retrieval-Augmented Generation (RAG) para SNN-LM.

Componentes:
  - `KnowledgeBase`: vector store con similitud coseno, persistencia .pt.
  - `make_model_encoder(model)`: encoder usando hidden states del SNNLM.
  - `make_sentence_transformer_encoder(name)`: encoder usando sentence-transformers (opcional).
  - `generate_with_rag(model, prompt, kb, ...)`: wrapper sobre `generate()`.

Diseño:
  - Los embeddings se L2-normalizan al añadir → búsqueda por coseno = dot product.
  - Thread-safe para lectura concurrente; `add()` toma lock exclusivo.
  - Persistencia atómica (tmp + os.replace).
  - Si `sentence_transformers` no está instalado, `make_sentence_transformer_encoder`
    lanza ImportError con instrucciones claras. El encoder por defecto es el modelo.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


EncoderFn = Callable[[str], np.ndarray]


# ─────────────────────────────────────────────────────────────────────
# KnowledgeBase
# ─────────────────────────────────────────────────────────────────────

class KnowledgeBase:
    """Vector store mínimo para RAG. Indexa textos por embedding.

    Thread-safe para `search()` concurrente. `add()` es exclusivo.
    """

    def __init__(
        self,
        dim: int,
        encoder_fn: Optional[EncoderFn] = None,
        max_docs: int = 100_000,
    ):
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if max_docs <= 0:
            raise ValueError(f"max_docs must be > 0, got {max_docs}")
        self.dim = dim
        self.encoder_fn = encoder_fn
        self.max_docs = max_docs
        self.keys = np.zeros((0, dim), dtype=np.float32)
        self.texts: list[str] = []
        self._lock = threading.RLock()

    # ------------------------------------------------------------ add
    def add(
        self,
        docs: Sequence[str],
        batch_size: int = 32,
        skip_empty: bool = True,
    ) -> int:
        """Añade documentos al índice. Devuelve el número efectivamente añadidos.

        Si `docs` hace exceder `max_docs`, se descartan los más antiguos (FIFO).
        """
        if self.encoder_fn is None:
            raise RuntimeError(
                "KnowledgeBase.add requires `encoder_fn`. "
                "Construct with encoder_fn=make_model_encoder(model) or similar."
            )
        if not docs:
            return 0

        docs_clean = [d for d in docs if (d and d.strip()) or not skip_empty]
        if not docs_clean:
            return 0

        new_keys: list[np.ndarray] = []
        new_texts: list[str] = []
        for i in range(0, len(docs_clean), batch_size):
            batch = docs_clean[i : i + batch_size]
            for text in batch:
                emb = np.asarray(self.encoder_fn(text), dtype=np.float32)
                if emb.shape != (self.dim,):
                    raise ValueError(
                        f"Encoder returned shape {emb.shape}, expected ({self.dim},)"
                    )
                norm = float(np.linalg.norm(emb))
                if norm < 1e-8:
                    continue  # embedding degenerado
                emb = emb / norm
                if not np.all(np.isfinite(emb)):
                    continue
                new_keys.append(emb)
                new_texts.append(text)

        if not new_keys:
            return 0

        keys_arr = np.stack(new_keys, axis=0)
        with self._lock:
            self.keys = np.concatenate([self.keys, keys_arr], axis=0)
            self.texts.extend(new_texts)
            # Trim FIFO
            overflow = len(self.texts) - self.max_docs
            if overflow > 0:
                self.keys = self.keys[overflow:]
                self.texts = self.texts[overflow:]

        return len(new_texts)

    # ------------------------------------------------------------ search
    def search(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[str, float]]:
        """Devuelve top-k (texto, score_coseno) para `query`.

        Score ∈ [-1, 1]. Si el KB está vacío o el encoder no está, devuelve [].
        """
        if k <= 0:
            return []
        if self.encoder_fn is None:
            logger.warning("KnowledgeBase.search: encoder_fn is None, returning []")
            return []

        with self._lock:
            if self.keys.shape[0] == 0:
                return []
            keys = self.keys  # referencia; numpy no muta el array
            texts = self.texts

        q = np.asarray(self.encoder_fn(query), dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"Encoder returned shape {q.shape}, expected ({self.dim},)")
        norm = float(np.linalg.norm(q))
        if norm < 1e-8:
            return []
        q = q / norm
        if not np.all(np.isfinite(q)):
            return []

        scores = keys @ q  # (N,) cosine
        k_eff = min(k, scores.shape[0])
        top_idx = np.argpartition(-scores, k_eff - 1)[:k_eff]
        # Ordenar los k por score descendente
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(texts[i], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            payload = {
                "version": 1,
                "dim": self.dim,
                "keys": self.keys.copy(),
                "texts": list(self.texts),
                "max_docs": self.max_docs,
            }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, encoder_fn: Optional[EncoderFn] = None) -> "KnowledgeBase":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        version = payload.get("version", 0)
        if version != 1:
            raise ValueError(f"KB format version {version} != 1")
        kb = cls(
            dim=int(payload["dim"]),
            encoder_fn=encoder_fn,
            max_docs=int(payload.get("max_docs", 100_000)),
        )
        kb.keys = np.asarray(payload["keys"], dtype=np.float32)
        kb.texts = list(payload["texts"])
        if kb.keys.shape[0] != len(kb.texts):
            raise ValueError(
                f"KB corrupted: keys.shape[0]={kb.keys.shape[0]} != "
                f"len(texts)={len(kb.texts)}"
            )
        if kb.keys.shape[0] > 0 and kb.keys.shape[1] != kb.dim:
            raise ValueError(
                f"KB dim mismatch: keys.shape[1]={kb.keys.shape[1]} != {kb.dim}"
            )
        return kb

    # ------------------------------------------------------------ info
    def __len__(self) -> int:
        return len(self.texts)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, n_docs={len(self)}, max_docs={self.max_docs}"


# ─────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────

def _unwrap(model):
    seen: set[int] = set()
    while hasattr(model, "_orig_mod") and id(model) not in seen:
        seen.add(id(model))
        model = model._orig_mod
    return model


def _is_rt_model(model) -> bool:
    model = _unwrap(model)
    if model.__class__.__name__ == "RTModelWrapper":
        return True
    return (
        hasattr(model, "rv")
        and hasattr(model, "snnlm")
        and hasattr(model.rv, "encode")
    )


def make_model_encoder(
    model,
    enc_name: str = "cl100k_base",
    pooling: str = "mean",
) -> EncoderFn:
    """Encoder que usa los hidden states del SNNLM (mean o last-token pooling).

    Compatible con SNNLM puro y RTModelWrapper. NO actualiza memoria ni KV-cache.

    Args:
        model: SNNLM o RTModelWrapper.
        enc_name: tokenizer para SNNLM puro (ignorado si RT).
        pooling: "mean" | "last".
    """
    if pooling not in ("mean", "last"):
        raise ValueError(f"pooling must be 'mean'|'last', got {pooling!r}")

    model = _unwrap(model)
    device = next(model.parameters()).device
    is_rt = _is_rt_model(model)

    if not is_rt:
        import tiktoken
        _enc = tiktoken.get_encoding(enc_name)
    else:
        _enc = None

    @torch.no_grad()
    def encode(text: str) -> np.ndarray:
        model.eval()
        if is_rt:
            ids = model.rv.encode(text)
            ids = ids[: model.config.seq_len]
            if not ids:
                ids = [0]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            q_emb, _, _ = model.rv.embed_and_quantize(idx, update=False)
            logits, _, _, hidden = model.snnlm._forward(
                idx, update_memory=False, precomputed_embeddings=q_emb,
            )
        else:
            ids = _enc.encode_ordinary(text)[: model.config.seq_len]
            if not ids:
                ids = [_enc.eot_token]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _, _, hidden = model._forward(idx, update_memory=False)

        if pooling == "mean":
            vec = hidden.mean(dim=1)[0]
        else:
            vec = hidden[0, -1]
        return vec.float().cpu().numpy()

    return encode


def make_sentence_transformer_encoder(name: str = "all-MiniLM-L6-v2") -> EncoderFn:
    """Encoder independiente del SNNLM. Requiere `sentence-transformers` instalado."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for make_sentence_transformer_encoder. "
            "Install with: pip install sentence-transformers"
        ) from e

    st = SentenceTransformer(name)
    st.eval()

    def encode(text: str) -> np.ndarray:
        vec = st.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vec, dtype=np.float32)

    return encode


# ─────────────────────────────────────────────────────────────────────
# RAG-aware generate
# ─────────────────────────────────────────────────────────────────────

def build_rag_prompt(
    prompt: str,
    kb: KnowledgeBase,
    k: int = 4,
    max_context_chars: int = 2000,
    separator: str = "\n\n",
) -> str:
    """Construye el prompt enriquecido con contexto recuperado.

    Si no hay contexto (KB vacío o encoder ausente), devuelve `prompt` tal cual.
    """
    if kb is None or len(kb) == 0:
        return prompt
    results = kb.search(prompt, k=k)
    if not results:
        return prompt

    parts = [text for text, _score in results]
    context = separator.join(parts)
    if max_context_chars > 0 and len(context) > max_context_chars:
        context = context[:max_context_chars]

    if not context.strip():
        return prompt

    return f"{context}{separator}{prompt}"
