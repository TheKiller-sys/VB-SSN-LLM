"""TokenSymbolMapper: mapea símbolos del ReasoningGraph a token IDs.

El grafo opera sobre palabras/identificadores (símbolos). El tokenizer
produce IDs BPE. Para sesgar logits durante decoding, necesitamos un
mapeo símbolo -> token_id.

Diseño:
  - `symbol -> token_id`: primer token BPE del símbolo. Aproximado
    (palabras multi-token solo se mapean a su primer sub-token).
  - `token_id -> [symbols]`: índice inverso para lookup rápido.
  - Construcción offline (O(V·encode)). Se cachea en .pt junto al grafo.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, Optional

import torch

logger = logging.getLogger(__name__)


class TokenSymbolMapper:
    """Bidireccional entre símbolos del grafo y token IDs del tokenizer."""

    def __init__(
        self,
        symbol_to_token: dict[str, int],
        token_to_symbols: dict[int, list[str]],
        encoder_name: str,
        vocab_size: int,
    ):
        self.symbol_to_token = dict(symbol_to_token)
        self.token_to_symbols = {
            int(k): list(v) for k, v in token_to_symbols.items()
        }
        self.encoder_name = encoder_name
        self.vocab_size = int(vocab_size)

    # --------------------------------------------------------- build
    @classmethod
    def from_encoder_and_graph(
        cls,
        encoder,
        graph,
        encoder_name: str = "cl100k_base",
        vocab_size: Optional[int] = None,
    ) -> "TokenSymbolMapper":
        """Construye el mapper a partir de un encoder (tiktoken o RT) y un grafo.

        `encoder` debe exponer `encode_ordinary(str) -> list[int]` (tiktoken)
        o `encode(str) -> list[int]` (RT). Se detecta automáticamente.
        """
        if vocab_size is None:
            if hasattr(encoder, "n_vocab"):
                vocab_size = int(encoder.n_vocab)
            elif hasattr(encoder, "vocab_size"):
                vocab_size = int(encoder.vocab_size)
            else:
                raise ValueError(
                    "vocab_size required: encoder has no n_vocab/vocab_size"
                )

        symbols = set()
        if hasattr(graph, "_symbols"):
            symbols.update(graph._symbols.keys())
        if hasattr(graph, "rules"):
            for r in graph.rules.values():
                symbols.add(r.antecedent)
                symbols.add(r.consequent)

        encode_fn = cls._resolve_encoder(encoder)

        s2t: dict[str, int] = {}
        t2s: dict[int, list[str]] = {}
        for sym in symbols:
            try:
                ids = encode_fn(sym)
            except Exception as e:
                logger.debug(f"encode failed for symbol {sym!r}: {e}")
                continue
            if not ids:
                continue
            tok = int(ids[0])
            if tok < 0 or tok >= vocab_size:
                continue
            s2t[sym] = tok
            t2s.setdefault(tok, []).append(sym)

        return cls(
            symbol_to_token=s2t,
            token_to_symbols=t2s,
            encoder_name=encoder_name,
            vocab_size=vocab_size,
        )

    @staticmethod
    def _resolve_encoder(encoder):
        if hasattr(encoder, "encode_ordinary"):
            return encoder.encode_ordinary
        if hasattr(encoder, "encode"):
            return lambda s: list(encoder.encode(s))
        raise ValueError(
            "encoder must expose encode_ordinary(str) or encode(str)->list[int]"
        )

    # --------------------------------------------------------- bias
    def bias_logits_for_consequences(
        self,
        logits: torch.Tensor,
        consequences: Iterable[dict],
        strength: float,
        top_k: int = 16,
    ) -> int:
        """Aplica bias additivo a logits en el último eje.

        Args:
            logits: (..., vocab_size) tensor in-place.
            consequences: iterable of dicts con keys 'consequent', 'confidence'.
            strength: escala del bias (<=0 no hace nada).
            top_k: número máximo de consecuencias a aplicar.

        Returns:
            número de consecuencias efectivamente aplicadas.
        """
        if strength <= 0.0 or logits.size(-1) != self.vocab_size:
            return 0

        cons_sorted = sorted(
            consequences, key=lambda c: -float(c.get("confidence", 0.0)),
        )[:top_k]

        applied = 0
        for c in cons_sorted:
            sym = c.get("consequent")
            if sym is None:
                continue
            tok = self.symbol_to_token.get(sym)
            if tok is None or tok >= logits.size(-1):
                continue
            conf = float(c.get("confidence", 0.0))
            logits[..., tok] = logits[..., tok] + strength * conf
            applied += 1
        return applied

    # --------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "version": 1,
            "symbol_to_token": self.symbol_to_token,
            "token_to_symbols": self.token_to_symbols,
            "encoder_name": self.encoder_name,
            "vocab_size": self.vocab_size,
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "TokenSymbolMapper":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("version") != 1:
            raise ValueError(
                f"Unknown TokenSymbolMapper version: {payload.get('version')}"
            )
        return cls(
            symbol_to_token=payload["symbol_to_token"],
            token_to_symbols=payload["token_to_symbols"],
            encoder_name=payload.get("encoder_name", "cl100k_base"),
            vocab_size=int(payload["vocab_size"]),
        )

    def __len__(self) -> int:
        return len(self.symbol_to_token)

    def extra_repr(self) -> str:
        return (
            f"symbols={len(self.symbol_to_token)}, "
            f"tokens={len(self.token_to_symbols)}, "
            f"vocab={self.vocab_size}, encoder={self.encoder_name}"
        )
