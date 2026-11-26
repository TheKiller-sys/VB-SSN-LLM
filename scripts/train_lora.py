"""Fine-tuning incremental con LoRA sobre un checkpoint v4.3.

Uso:
    python scripts/train_lora.py data/new/ --base checkpoints/run1/best.pt \\
        --out checkpoints/lora_new.pt --max-steps 500
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from snn_lm_v3 import (
    SNNLM, make_dataloader, make_rt_dataloader,
    train, resolve_device, load_checkpoint,
    apply_lora, freeze_base_for_lora, count_lora_params, save_lora,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+")
    p.add_argument("--base", required=True, help="Checkpoint base (frozen)")
    p.add_argument("--out", required=True, help="Ruta del adapter LoRA a guardar")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--targets", nargs="+", default=None,
                   help="Si None, se autodetectan según core/ffn del base.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--device", default="auto")
    p.add_argument("--extensions", nargs="+", default=[".txt"])
    p.add_argument("--split-by", choices=["file", "repo"], default="file")
    p.add_argument("--code-corpus", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger("train_lora")

    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    logger.info(f"Cargando base desde {args.base}")
    ckpt = load_checkpoint(args.base, SNNLM, device, load_optimizer=False)
    model = ckpt["model"]
    cfg = ckpt["config"]
    logger.info(f"Base core={cfg.core_type}, ffn={cfg.ffn_mode}")

    # Overrides de entrenamiento
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.grad_accum_steps = args.grad_accum
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    cfg.lr = args.lr
    cfg.min_lr = args.lr * 0.1
    cfg.warmup_steps = args.warmup_steps
    cfg.max_steps = args.max_steps
    cfg.log_every = args.log_every
    cfg.eval_every = args.eval_every
    cfg.eval_batches = args.eval_batches
    cfg.ckpt_every = args.ckpt_every
    cfg.seed = args.seed
    cfg.num_workers = args.num_workers
    cfg.out_dir = os.path.dirname(args.out) or "checkpoints"
    cfg.device = str(device)
    cfg.split_by = args.split_by
    if args.no_amp:
        cfg.use_amp = False
    cfg.lora_enabled = True
    cfg.lora_r = args.r
    cfg.lora_alpha = args.alpha
    cfg.lora_dropout = args.dropout
    if args.targets:
        cfg.lora_targets = list(args.targets)
    # else: apply_lora autodetecta
    cfg.lora_path = args.out
    cfg.validate()

    # Aplicar LoRA
    n_modules = apply_lora(
        model, r=args.r, alpha=args.alpha,
        dropout=args.dropout,
        targets=tuple(args.targets) if args.targets else None,
    )
    n_trainable = freeze_base_for_lora(model)
    logger.info(
        f"LoRA aplicado: {n_modules} módulos, "
        f"{count_lora_params(model):,} params entrenables"
    )

    # Dataloaders
    extensions = args.extensions
    if args.code_corpus and extensions == [".txt"]:
        from snn_lm_v3.dataset import CODE_EXTENSIONS
        extensions = list(CODE_EXTENSIONS)
    use_rt = ckpt.get("use_rt", False)
    if use_rt:
        rv = model.rv if hasattr(model, "rv") else model._orig_mod.rv
        train_dl, val_dl = make_rt_dataloader(
            args.data, cfg.seq_len, cfg.batch_size, rv,
            seed=cfg.seed, num_workers=cfg.num_workers,
            split_by=cfg.split_by, extensions=tuple(extensions),
        )
    else:
        train_dl, val_dl, _ = make_dataloader(
            args.data, cfg.seq_len, cfg.batch_size,
            enc_name="cl100k_base", seed=cfg.seed,
            num_workers=cfg.num_workers,
            split_by=cfg.split_by, extensions=tuple(extensions),
        )
    logger.info(f"Train batches: {len(train_dl)}, Val: {len(val_dl)}")

    model = train(model, train_dl, val_dl, cfg, logger)

    save_lora(model, args.out, extra={
        "base_ckpt": args.base,
        "step": int(cfg.max_steps),
        "r": args.r,
        "alpha": args.alpha,
        "targets": list(cfg.lora_targets) if cfg.lora_targets else None,
        "core_type": cfg.core_type,
        "ffn_mode": cfg.ffn_mode,
    })
    logger.info(f"Adapter LoRA guardado en {args.out}")


if __name__ == "__main__":
    main()
