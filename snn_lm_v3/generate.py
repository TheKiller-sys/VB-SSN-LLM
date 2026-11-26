"""Wrapper de generación: detecta RT, integra RAG opcional.

`update_memory` por defecto es `False` para evitar que la generación
contamine la memoria episódica. Pasa `update_memory=True` explícitamente
para continual learning.
"""
from __future__ import annotations

import torch
import tiktoken


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
        and hasattr(model.rv, "decode")
    )


@torch.no_grad()
def generate(
    model,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    enc_name: str | None = None,
    device=None,
    use_cache: bool = True,
    update_memory: bool = False,  # ── FIX v4.2: default seguro ──
    rag_kb=None,
    rag_top_k: int = 4,
    rag_max_context_chars: int = 2000,
) -> str:
    """Genera texto. Si `rag_kb` se da, enriquece el prompt con contexto."""
    device = device or next(model.parameters()).device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    if rag_kb is not None:
        from .rag import build_rag_prompt
        prompt = build_rag_prompt(
            prompt, rag_kb,
            k=rag_top_k,
            max_context_chars=rag_max_context_chars,
        )

    is_rt = _is_rt_model(model)

    if is_rt:
        ids = model.encode(prompt) if prompt else [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx, max_new_tokens, temperature, top_k, top_p,
            use_cache=use_cache, update_memory=update_memory,
        )
        return model.decode(out[0].tolist())

    enc_name = enc_name or getattr(model.config, "enc_name", "cl100k_base")
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode_ordinary(prompt) if prompt else [enc.eot_token]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens, temperature, top_k, top_p,
        use_cache=use_cache, update_memory=update_memory,
    )
    return enc.decode(out[0].tolist())
