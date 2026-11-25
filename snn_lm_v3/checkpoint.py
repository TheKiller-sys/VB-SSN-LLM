"""Guardado y carga de checkpoints con soporte RT."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

CKPT_VERSION = 4  # bumped: now RT-aware


def _unwrap(model):
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _is_rt(model) -> bool:
    try:
        from .tokenizer import RTModelWrapper
        return isinstance(model, RTModelWrapper)
    except ImportError:
        return False


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


def load_checkpoint(path: str, model_class, device="cpu", load_optimizer: bool = False) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    version = ckpt.get("version", 0)
    if version != CKPT_VERSION:
        logging.warning(f"Checkpoint version {version} != {CKPT_VERSION}. May be incompatible.")

    from .config import GPTConfig
    raw = ckpt["config"]
    cfg = GPTConfig(**{k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__})

    # Validate vocab
    ckpt_vocab = raw.get("vocab_size")
    if ckpt_vocab is not None:
        model_vocab = ckpt["model"]["tok_emb.weight"].shape[0]
        if ckpt_vocab != model_vocab:
            raise ValueError(f"Checkpoint vocab {ckpt_vocab} != tok_emb shape {model_vocab}")

    use_rt = ckpt.get("use_rt", False)

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
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model = model_class(cfg)
        model.load_state_dict(ckpt["model"], strict=False)

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
