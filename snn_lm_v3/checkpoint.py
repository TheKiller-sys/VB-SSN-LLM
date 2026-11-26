"""Guardado y carga de checkpoints con soporte RT, LoRA y validación estricta.

FIX v4.2.1:
  - `load_checkpoint` toma `r`/`alpha` reales desde `lora_state` (por módulo),
    no desde `lora_config` (que puede estar desincronizado).
  - Detección robusta de `r`/`alpha` mixtos en el adapter.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

CKPT_VERSION = 5  # bumped: LoRA-aware + memory-aware


# Buffers persistent=False que NO aparecen en state_dict y por tanto pueden
# aparecer como "missing" en PyTorch antiguos. Filtro explícito (no substrings).
_IGNORE_MISSING_EXACT = {
    "rope_cos", "rope_sin",
    "last_firing_rate",
}
_IGNORE_MISSING_SUFFIXES = (
    ".keys", ".values", ".ptr", ".size",       # NeuralMemory buffers
    ".last_firing_rate",                       # track_stats buffers
)


def _unwrap(model):
    seen: set[int] = set()
    while hasattr(model, "_orig_mod") and id(model) not in seen:
        seen.add(id(model))
        model = model._orig_mod
    return model


def _is_rt(model) -> bool:
    model = _unwrap(model)
    return model.__class__.__name__ == "RTModelWrapper"


def _has_rt_keys(state_dict: dict) -> bool:
    return any(k.startswith("rv.") for k in state_dict.keys())


def _has_lora_keys(state_dict: dict) -> bool:
    return any(".lora_A" in k or ".lora_B" in k for k in state_dict.keys())


def _is_ignorable_missing(key: str) -> bool:
    if key in _IGNORE_MISSING_EXACT:
        return True
    return any(key.endswith(s) for s in _IGNORE_MISSING_SUFFIXES)


def _extract_lora_r_alpha(ckpt: dict, cfg) -> tuple[int, float]:
    """Devuelve (r, alpha) autoritativos desde `lora_state`, con fallback a
    `lora_config` y luego a `cfg`."""
    lora_cfg = ckpt.get("lora_config") or {}
    lora_state = ckpt.get("lora_state") or {}

    if lora_state:
        rs = {int(e["r"]) for e in lora_state.values() if "r" in e}
        alphas = {float(e["alpha"]) for e in lora_state.values() if "alpha" in e}
        if len(rs) > 1:
            raise ValueError(
                f"LoRA state has mixed r values: {sorted(rs)}. "
                f"Non-uniform r is not supported."
            )
        if len(alphas) > 1:
            raise ValueError(
                f"LoRA state has mixed alpha values: {sorted(alphas)}. "
                f"Non-uniform alpha is not supported."
            )
        r = int(next(iter(rs))) if rs else int(lora_cfg.get("r", cfg.lora_r))
        alpha = float(next(iter(alphas))) if alphas else float(lora_cfg.get("alpha", cfg.lora_alpha))
        return r, alpha

    return (
        int(lora_cfg.get("r", cfg.lora_r)),
        float(lora_cfg.get("alpha", cfg.lora_alpha)),
    )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    config,
    step: int,
    path: str,
    best_val_loss: float | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    unwrapped = _unwrap(model)

    memory_state = None
    if hasattr(unwrapped, "memory_state_for_checkpoint"):
        memory_state = unwrapped.memory_state_for_checkpoint()

    is_rt = _is_rt(unwrapped)

    from .lora import is_lora_active, lora_state_dict
    lora_active = is_lora_active(unwrapped)
    lora_state = lora_state_dict(unwrapped) if lora_active else None

    payload = {
        "version": CKPT_VERSION,
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config.__dict__,
        "step": step,
        "best_val_loss": best_val_loss,
        "memory_state": memory_state,
        "use_rt": is_rt,
        "use_lora": lora_active,
        "lora_state": lora_state,
        "lora_config": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "targets": list(config.lora_targets),
        } if lora_active else None,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    if is_rt:
        payload["rt_config"] = dict(unwrapped.rv.cfg.__dict__)
        payload["bpe_path"] = getattr(config, "rt_bpe_path", None)

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, model_class, device="cpu", load_optimizer: bool = False) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    version = ckpt.get("version", 0)
    if version != CKPT_VERSION:
        logging.warning(
            f"Checkpoint version {version} != {CKPT_VERSION}. May be incompatible."
        )

    from .config import GPTConfig
    raw = ckpt["config"]
    cfg = GPTConfig(**{k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__})

    ckpt_vocab = raw.get("vocab_size")
    if ckpt_vocab is not None:
        model_vocab = ckpt["model"]["tok_emb.weight"].shape[0]
        if ckpt_vocab != model_vocab:
            raise ValueError(
                f"Checkpoint vocab {ckpt_vocab} != tok_emb shape {model_vocab}"
            )

    use_rt = ckpt.get("use_rt", False)
    use_lora = ckpt.get("use_lora", False)
    state_dict = ckpt["model"]
    has_rt_state = _has_rt_keys(state_dict)
    has_lora_state = _has_lora_keys(state_dict)

    if use_rt and not has_rt_state:
        raise ValueError(
            "Checkpoint says use_rt=True but has no 'rv.*' keys in state_dict."
        )
    if not use_rt and has_rt_state:
        raise ValueError(
            "Checkpoint has 'rv.*' keys but use_rt=False. Refusing to load."
        )
    if not use_rt and cfg.use_rt:
        raise ValueError(
            "Checkpoint was saved without RT but config.use_rt=True. Refusing."
        )
    if use_lora and not has_lora_state:
        raise ValueError(
            "Checkpoint says use_lora=True but has no 'lora_A/B' keys."
        )
    if has_lora_state and not use_lora:
        raise ValueError(
            "Checkpoint has LoRA keys but use_lora=False in payload."
        )

    if use_rt:
        from .tokenizer import ResonantVocab, RVConfig, RTModelWrapper
        rv_data = ckpt.get("rt_config") or {}
        rv_cfg = RVConfig(**{k: v for k, v in rv_data.items() if k in RVConfig.__dataclass_fields__})
        bpe_path = ckpt.get("bpe_path")
        if not bpe_path or not os.path.exists(bpe_path):
            raise ValueError(
                f"RT checkpoint requires existing bpe_path. Got: {bpe_path!r}."
            )
        rv = ResonantVocab.from_bpe(bpe_path, rv_cfg)
        snnlm = model_class(cfg)
        model = RTModelWrapper(snnlm, rv, aux_loss_weight=cfg.rt_aux_loss_weight)
    else:
        model = model_class(cfg)

    if use_lora:
        from .lora import apply_lora
        lora_cfg = ckpt.get("lora_config") or {}
        # ── FIX v4.2.1: r/alpha autoritativos vienen de lora_state ──
        r, alpha = _extract_lora_r_alpha(ckpt, cfg)
        apply_lora(
            model,
            r=r,
            alpha=alpha,
            dropout=float(lora_cfg.get("dropout", cfg.lora_dropout)),
            targets=tuple(lora_cfg.get("targets", cfg.lora_targets)),
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected[:10]}")

    # ── FIX v4.2: filtro robusto (sin substrings frágiles) ──
    real_missing = [k for k in missing if not _is_ignorable_missing(k)]
    if real_missing:
        raise ValueError(f"Missing parameters in checkpoint: {real_missing[:10]}")

    if use_lora and ckpt.get("lora_state"):
        from .lora import load_lora_state_dict
        load_lora_state_dict(model, ckpt["lora_state"])

    model.to(device)

    if "memory_state" in ckpt and ckpt["memory_state"] is not None:
        if hasattr(model, "load_memory_state_from_checkpoint"):
            model.load_memory_state_from_checkpoint(ckpt["memory_state"], device)

    result = {
        "model": model,
        "config": cfg,
        "step": ckpt.get("step", 0),
        "best_val_loss": ckpt.get("best_val_loss"),
        "use_rt": use_rt,
        "use_lora": use_lora,
    }

    if load_optimizer:
        result["optimizer_state"] = ckpt.get("optimizer")
        result["scheduler_state"] = ckpt.get("scheduler")
        rng = ckpt.get("rng_state") or {}
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])

    return result
