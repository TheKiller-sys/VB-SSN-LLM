"""Guardado y carga de checkpoints con versionado y resume completo."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

CKPT_VERSION = 3


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


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

    payload = {
        "version": CKPT_VERSION,
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config.__dict__,
        "step": step,
        "best_val_loss": best_val_loss,
        "memory_state": memory_state,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy": np.random.get_state(),
        },
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str, model_class, device="cpu", load_optimizer: bool = False
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    version = ckpt.get("version", 0)
    if version != CKPT_VERSION:
        logging.warning(
            f"Checkpoint version {version} != {CKPT_VERSION}. May be incompatible."
        )

    from .config import GPTConfig

    raw = ckpt["config"]
    cfg = GPTConfig(
        **{k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
    )
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
