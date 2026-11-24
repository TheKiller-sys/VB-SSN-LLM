"""Guardado y carga de checkpoints con versionado y resume completo."""
from __future__ import annotations

import logging
import os

import numpy as np
import torch

CKPT_VERSION = 1


def _unwrap(model):
    """Soporta modelos envueltos por torch.compile."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, optimizer, scheduler, config, step: int, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "version": CKPT_VERSION,
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "config": config.__dict__,
            "step": step,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
                "numpy": np.random.get_state(),
            },
        },
        path,
    )


def load_checkpoint(
    path: str, model_class, device="cpu", load_optimizer: bool = False
) -> dict:
    """Carga checkpoint. Devuelve dict con model, config, step (y estados si load_optimizer)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    version = ckpt.get("version", 0)
    if version != CKPT_VERSION:
        logging.warning(
            f"Checkpoint version {version} != {CKPT_VERSION}. "
            "Puede haber incompatibilidad."
        )

    from .config import GPTConfig

    config = GPTConfig(
        **{
            k: v
            for k, v in ckpt["config"].items()
            if k in GPTConfig.__dataclass_fields__
        }
    )
    model = model_class(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    result = {
        "model": model,
        "config": config,
        "step": ckpt.get("step", 0),
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
