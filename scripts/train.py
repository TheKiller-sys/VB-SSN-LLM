"""
Entrena SNN-LM v3 sobre archivos .txt.

Ejemplos:
    python scripts/train.py data/ --size tiny --device cuda
    python scripts/train.py data/ --size small --max-steps 50000
    python scripts/train.py data/ --config my_config.json
"""
import argparse
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from snn_lm_v3 import (
    GPTConfig, SIZE_TINY, SIZE_SMALL, SIZE_MEDIUM,
    SNNLM, make_dataloader, train,
)
from snn_lm_v3.checkpoint import load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+")
    p.add_argument("--size", choices=["tiny", "small", "medium"], default="tiny")
    p.add_argument("--config", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.config:
        cfg = GPTConfig.load(args.config)
    else:
        size = {"tiny": SIZE_TINY, "small": SIZE_SMALL, "medium": SIZE_MEDIUM}[args.size]
        cfg = GPTConfig(**size)

    for k, v in [
        ("max_steps", args.max_steps), ("batch_size", args.batch_size),
        ("seq_len", args.seq_len), ("lr", args.lr),
        ("log_every", args.log_every), ("eval_every", args.eval_every),
        ("ckpt_every", args.ckpt_every), ("seed", args.seed),
        ("out_dir", args.out), ("device", args.device),
    ]:
        if v is not None:
            setattr(cfg, k, v)

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    from snn_lm_v3 import get_logger if False else None  # noqa
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger("train")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.resume:
        logger.info(f"Cargando {args.resume}")
        model, _, step = load_checkpoint(args.resume, SNNLM, device)
    else:
        model = SNNLM(cfg)
        step = 0

    logger.info(f"Parámetros: {model.num_params():,}")

    train_dl, val_dl, enc = make_dataloader(
        args.data, cfg.seq_len, cfg.batch_size, enc_name="gpt2",
        seed=cfg.seed,
    )
    logger.info(f"Train batches/epoch: {len(train_dl)}, Val: {len(val_dl)}")

    train(model, train_dl, val_dl, cfg, logger)


if __name__ == "__main__":
    main()
