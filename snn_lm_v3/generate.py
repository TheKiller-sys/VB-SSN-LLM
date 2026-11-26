"""Wrapper de generación: detecta RT sin importar el módulo, evita fallbacks silenciosos."""
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
    """Detecta RTModelWrapper sin importar `tokenizer` (funciona con torch.compile)."""
    model = _unwrap(model)
    if model.__class__.__name__ == "RTModelWrapper":
        return True
    # Fallback: chequeo estructural (por si el nombre cambió)
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
    update_memory: bool = True,
) -> str:
    device = device or next(model.parameters()).device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    is_rt = _is_rt_model(model)

    if is_rt:
        ids = model.encode(prompt) if prompt else [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx, max_new_tokens, temperature, top_k, top_p,
            use_cache=use_cache, update_memory=update_memory,
        )
        return model.decode(out[0].tolist())

    # Standard tiktoken
    enc_name = enc_name or getattr(model.config, "enc_name", "cl100k_base")
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode_ordinary(prompt) if prompt else [enc.eot_token]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens, temperature, top_k, top_p,
        use_cache=use_cache, update_memory=update_memory,
    )
    return enc.decode(out[0].tolist())
