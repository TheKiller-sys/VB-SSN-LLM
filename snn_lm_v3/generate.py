"""Wrapper de generación: tokeniza prompt, llama model.generate, decodifica."""
from __future__ import annotations

import torch
import tiktoken


@torch.no_grad()
def generate(
    model,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    enc_name: str = "gpt2",
    device=None,
    use_cache: bool = True,
) -> str:
    enc = tiktoken.get_encoding(enc_name)
    device = device or next(model.parameters()).device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    ids = enc.encode_ordinary(prompt) if prompt else [enc.eot_token]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens, temperature, top_k, top_p, use_cache=use_cache
    )
    return enc.decode(out[0].tolist())
