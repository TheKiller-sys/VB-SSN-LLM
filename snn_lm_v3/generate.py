"""Wrapper de generación: detecta RT, integra RAG y reasoning bias.

v4.3:
  - Construye/carga `ReasoningBias` si `config.reasoning_bias_enabled`.
  - Construye/carga `TokenSymbolMapper` (cacheado en disco).
  - Pasa `reasoning_bias` a `model.generate()` (SNNLM o RTModelWrapper).
"""
from __future__ import annotations

import logging
import os

import torch
import tiktoken

from .symbol_mapping import TokenSymbolMapper
from .reasoning_bias import ReasoningBias

logger = logging.getLogger(__name__)


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


def _build_or_load_mapper(
    model,
    graph,
    mapper_path: str,
    encoder_name: str,
) -> TokenSymbolMapper:
    """Carga mapper de disco o lo construye y guarda."""
    if os.path.exists(mapper_path):
        try:
            mapper = TokenSymbolMapper.load(mapper_path)
            logger.info(
                f"TokenSymbolMapper cargado: {len(mapper)} símbolos "
                f"desde {mapper_path}"
            )
            return mapper
        except Exception as e:
            logger.warning(
                f"Fallo cargando mapper ({e}); reconstruyendo desde grafo."
            )

    if _is_rt_model(model):
        encoder = model.rv
        vocab_size = model.rv.vocab_size
    else:
        encoder = tiktoken.get_encoding(encoder_name)
        vocab_size = encoder.n_vocab

    mapper = TokenSymbolMapper.from_encoder_and_graph(
        encoder=encoder,
        graph=graph,
        encoder_name=encoder_name,
        vocab_size=vocab_size,
    )
    try:
        mapper.save(mapper_path)
        logger.info(
            f"TokenSymbolMapper construido y guardado: {len(mapper)} símbolos "
            f"-> {mapper_path}"
        )
    except Exception as e:
        logger.warning(f"No se pudo guardar mapper: {e}")
    return mapper


def _build_reasoning_bias(model, config, encoder_name: str):
    """Construye ReasoningBias si está habilitado y hay grafo+mapper."""
    if not getattr(config, "reasoning_bias_enabled", False):
        return None

    from .reasoning.persistence import load_graph

    graph_path = config.reasoning_graph_path
    if not graph_path or not os.path.exists(graph_path):
        logger.warning(
            f"reasoning_bias_enabled=True pero graph no existe en "
            f"{graph_path!r}. Bias desactivado."
        )
        return None

    try:
        graph = load_graph(graph_path)
    except Exception as e:
        logger.warning(f"Fallo cargando grafo: {e}. Bias desactivado.")
        return None

    try:
        mapper = _build_or_load_mapper(
            model, graph, config.reasoning_mapper_path, encoder_name,
        )
    except Exception as e:
        logger.warning(
            f"Fallo construyendo TokenSymbolMapper: {e}. Bias desactivado."
        )
        return None

    bias = ReasoningBias(
        graph=graph,
        mapper=mapper,
        strength=config.reasoning_bias_strength,
        top_k=config.reasoning_bias_top_k,
        remine_every=config.reasoning_bias_remine_every,
        facts_window=config.reasoning_bias_facts_window,
        min_confidence=config.reasoning_bias_min_confidence,
    )
    logger.info(
        f"ReasoningBias activo: {len(mapper)} símbolos, "
        f"strength={bias.strength}, top_k={bias.top_k}, "
        f"remine_every={bias.remine_every}"
    )
    return bias


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
    update_memory: bool = False,
    rag_kb=None,
    rag_top_k: int = 4,
    rag_max_context_chars: int = 2000,
    reasoning_bias: ReasoningBias | None = None,
    auto_reasoning_bias: bool = True,
) -> str:
    """Genera texto. Si `rag_kb`, enriquece el prompt. Si `reasoning_bias`
    (o `auto_reasoning_bias` con `config.reasoning_bias_enabled`), sesga logits
    durante decoding.
    """
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
    inner = _unwrap(model)
    if is_rt:
        config = inner.config
    else:
        config = inner.config

    # Construcción automática del bias (solo si no se pasó uno explícito)
    bias = reasoning_bias
    if bias is None and auto_reasoning_bias:
        try:
            bias = _build_reasoning_bias(
                model, config, enc_name or "cl100k_base",
            )
        except Exception as e:
            logger.warning(f"Auto ReasoningBias falló: {e}")
            bias = None

    if is_rt:
        ids = model.encode(prompt) if prompt else [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx, max_new_tokens, temperature, top_k, top_p,
            use_cache=use_cache, update_memory=update_memory,
            reasoning_bias=bias,
        )
        return model.decode(out[0].tolist())

    enc_name = enc_name or getattr(config, "enc_name", "cl100k_base")
    enc = tiktoken.get_encoding(enc_name)
    ids = enc.encode_ordinary(prompt) if prompt else [enc.eot_token]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = inner.generate(
        idx, max_new_tokens, temperature, top_k, top_p,
        use_cache=use_cache, update_memory=update_memory,
        reasoning_bias=bias,
    )
    return enc.decode(out[0].tolist())
