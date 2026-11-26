"""Guardado y carga de checkpoints con soporte RT y validación estricta."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

CKPT_VERSION = 4  # bumped: RT-aware


def _unwrap(model):
    """Unwraps torch.compile sin ciclos infinitos."""
    seen: set[int] = set()
    while hasattr(model, "_orig_mod") and id(model) not in seen:
        seen.add(id(model))
        model = model._orig_mod
    return model


def _is_rt(model) -> bool:
    """Detección de RT sin importar el módulo (funciona con torch.compile)."""
    model = _unwrap(model)
    return model.__class__.__name__ == "RTModelWrapper"


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


def _has_rt_keys(state_dict: dict) -> bool:
    """Detecta si un state_dict contiene claves del ResonantVocab."""
    return any(k.startswith("rv.") for k in state_dict.keys())


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

    # Validar vocab
    ckpt_vocab = raw.get("vocab_size")
    if ckpt_vocab is not None:
        model_vocab = ckpt["model"]["tok_emb.weight"].shape[0]
        if ckpt_vocab != model_vocab:
            raise ValueError(
                f"Checkpoint vocab {ckpt_vocab} != tok_emb shape {model_vocab}"
            )

    use_rt = ckpt.get("use_rt", False)
    state_dict = ckpt["model"]
    has_rt_state = _has_rt_keys(state_dict)

    # ── Validación estricta para evitar carga silenciosa ──
    if use_rt and not has_rt_state:
        raise ValueError(
            "Checkpoint says use_rt=True but has no 'rv.*' keys in state_dict. "
            "The checkpoint is corrupted or saved incorrectly."
        )
    if not use_rt and has_rt_state:
        raise ValueError(
            "Checkpoint has 'rv.*' keys but use_rt=False in payload. "
            "This checkpoint requires RT. Refusing to load to avoid silent "
            "corruption (rv weights would be dropped by strict=False)."
        )
    if not use_rt and cfg.use_rt:
        raise ValueError(
            "Checkpoint was saved without RT but config.use_rt=True. "
            "Would create a partial RT model. Refusing."
        )

    if use_rt:
        from .tokenizer import ResonantVocab, RVConfig, RTModelWrapper
        rv_data = ckpt.get("rt_config") or {}
        rv_cfg = RVConfig(**{k: v for k, v in rv_data.items() if k in RVConfig.__dataclass_fields__})
        bpe_path = ckpt.get("bpe_path")
        if not bpe_path or not os.path.exists(bpe_path):
            raise ValueError(
                f"RT checkpoint requires existing bpe_path. Got: {bpe_path!r}. "
                f"Restore the BPE file or set config.rt_bpe_path."
            )
        rv = ResonantVocab.from_bpe(bpe_path, rv_cfg)
        snnlm = model_class(cfg)
        model = RTModelWrapper(snnlm, rv, aux_loss_weight=cfg.rt_aux_loss_weight)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # strict=False solo debería tolerar buffers persistent=False ausentes
        if unexpected:
            raise ValueError(
                f"Unexpected keys in checkpoint: {unexpected[:10]}"
            )
    else:
        model = model_class(cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(
                f"Unexpected keys in checkpoint: {unexpected[:10]}"
            )

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
