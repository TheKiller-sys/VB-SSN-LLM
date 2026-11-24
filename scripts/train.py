"""
Entrena SNN-LM v3 sobre archivos .txt.

Ejemplos:
    python scripts/train.py data/ --size tiny
    python scripts/train.py data/ --size medium --device cuda
    python scripts/train.py data/ --config my_config.json
    python scripts/train.py data/ --resume checkpoints/latest.pt
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from snn_lm_v3 import (
    GPTConfig, SIZE_TINY, SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE,
    SNNLM, make_dataloader, train, resolve_device,
)
from snn_lm_v3.checkpoint import load_checkpoint


SIZES = {
    "tiny": SIZE_TINY,
    "small": SIZE_SMALL,
    "medium": SIZE_MEDIUM,
    "large": SIZE_LARGE,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+")
    p.add_argument("--size", choices=list(SIZES.keys()), default="tiny")
    p.add_argument("--config", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = GPTConfig.load(args.config)
    else:
        cfg = GPTConfig(**SIZES[args.size])

    overrides = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "log_every": args.log_every,
        "eval_every": args.eval_every,
        "ckpt_every": args.ckpt_every,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "out_dir": args.out,
        "device": args.device,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.compile:
        cfg.compile = True

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train")

    device = resolve_device(cfg.device)
    logger.info(f"Device: {device}")

    optimizer = None
    optimizer_state = None
    scheduler_state = None
    start_step = 0

    if args.resume:
        logger.info(f"Cargando checkpoint {args.resume}")
        ckpt = load_checkpoint(
            args.resume, SNNLM, device, load_optimizer=True
        )
        model = ckpt["model"]
        start_step = ckpt["step"]
        optimizer_state = ckpt.get("optimizer_state")
        scheduler_state = ckpt.get("scheduler_state")
        logger.info(f"Resume desde step={start_step}")
    else:
        model = SNNLM(cfg)

    logger.info(f"Parámetros: {model.num_params():,}")

    train_dl, val_dl, enc = make_dataloader(
        args.data,
        cfg.seq_len,
        cfg.batch_size,
        enc_name="gpt2",
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )
    logger.info(f"Train batches/epoch: {len(train_dl)}, Val: {len(val_dl)}")

    if optimizer_state is not None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
            eps=1e-8,
        )

    train(
        model, train_dl, val_dl, cfg, logger,
        start_step=start_step,
        optimizer=optimizer,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
    )


if __name__ == "__main__":
    main()
