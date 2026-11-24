"""Entrena SNN-LM v3.1 sobre archivos .txt con razonamiento simbólico opcional.

Ejemplos:
    python scripts/train.py data/ --size tiny
    python scripts/train.py data/ --size medium --device cuda
    python scripts/train.py data/ --config my_config.json --resume checkpoints/latest.pt
    python scripts/train.py data/ --size tiny --max-steps 500 --no-reasoning
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from snn_lm_v3 import (
    GPTConfig, SIZES, SNNLM, make_dataloader, train, resolve_device,
    load_texts,
    CorpusObserver, ReasoningGraph, ConsistencyVerifier, Verdict,
    save_graph,
)
from snn_lm_v3.checkpoint import load_checkpoint


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
    p.add_argument("--eval-batches", type=int, default=None)
    p.add_argument("--ckpt-every", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--snn-mode", choices=["static", "multi_step", "temporal", "gelu"],
                   default=None, help="Modo de la SNN en el FFN.")
    p.add_argument("--snn-steps", type=int, default=None,
                   help="Pasos internos por token en multi_step.")
    p.add_argument("--snn-track-stats", action="store_true",
                   help="Registrar firing_rate (necesario para razonamiento con señales).")
    p.add_argument("--no-reasoning", action="store_true",
                   help="Desactivar la capa de razonamiento simbólico.")
    p.add_argument("--reasoning-chunk-size", type=int, default=500,
                   help="Tamaño de ventana al observar el corpus.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_reasoning_layer(data_paths, chunk_size: int, logger):
    """Observa el corpus y mina reglas simbólicas."""
    observer = CorpusObserver(enc_name="gpt2")
    graph = ReasoningGraph(cooldown_steps=3)

    docs = load_texts(data_paths)
    logger.info(f"Observando {len(docs)} documentos del corpus...")
    for i, doc in enumerate(docs):
        for j in range(0, len(doc), chunk_size):
            observer.observe(doc[j : j + chunk_size], ref=f"doc{i}_chunk{j}")

    logger.info(f"CorpusObserver: {observer.get_state()}")
    n_rules = graph.mine_from_corpus(observer)
    logger.info(
        f"ReasoningGraph: {n_rules} reglas minadas, "
        f"{graph.get_state()['rules_count']} totales "
        f"({graph.metrics['cyclic_rejected']} cíclicas rechazadas, "
        f"{graph.metrics['tautological_rejected']} tautológicas)"
    )
    return observer, graph


def main():
    args = parse_args()

    cfg = GPTConfig.load(args.config) if args.config else GPTConfig(**SIZES[args.size])

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
        "eval_batches": args.eval_batches,
        "ckpt_every": args.ckpt_every,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "out_dir": args.out,
        "device": args.device,
        "snn_mode": args.snn_mode,
        "snn_num_steps": args.snn_steps,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.compile:
        cfg.compile = True
    if args.no_amp:
        cfg.use_amp = False
    if args.snn_track_stats:
        cfg.snn_track_stats = True
    cfg.validate()

    os.makedirs(cfg.out_dir, exist_ok=True)
    cfg.save(os.path.join(cfg.out_dir, "config.json"))

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("train")

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    logger.info(f"Device: {device}")

    # ---------------- razonamiento sobre corpus ----------------
    observer, graph = None, None
    if not args.no_reasoning:
        try:
            observer, graph = build_reasoning_layer(
                args.data, args.reasoning_chunk_size, logger
            )
            save_graph(graph, os.path.join(cfg.out_dir, "graph.pt"))
            logger.info(f"Grafo guardado en {os.path.join(cfg.out_dir, 'graph.pt')}")
        except Exception as e:
            logger.warning(f"Razonamiento desactivado por error: {e}")
            observer, graph = None, None

    # ---------------- modelo ----------------
    optimizer = None
    optimizer_state = None
    scheduler_state = None
    best_val_loss = None
    start_step = 0

    if args.resume:
        logger.info(f"Cargando checkpoint {args.resume}")
        ckpt = load_checkpoint(args.resume, SNNLM, device, load_optimizer=True)
        model = ckpt["model"]
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss")
        optimizer_state = ckpt.get("optimizer_state")
        scheduler_state = ckpt.get("scheduler_state")
        logger.info(f"Resume desde step={start_step}")
        if optimizer_state is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2),
                eps=1e-8,
            )
    else:
        model = SNNLM(cfg)

    logger.info(f"Parámetros: {model.num_params():,}")

    train_dl, val_dl, _ = make_dataloader(
        args.data,
        cfg.seq_len,
        cfg.batch_size,
        enc_name="gpt2",
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )
    logger.info(f"Train batches/epoch: {len(train_dl)}, Val: {len(val_dl)}")

    model = train(
        model,
        train_dl,
        val_dl,
        cfg,
        logger,
        start_step=start_step,
        optimizer=optimizer,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        best_val_loss=best_val_loss,
    )

    # ---------------- verificación final contra corpus ----------------
    if graph is not None and observer is not None:
        from snn_lm_v3 import generate
        logger.info("Verificación final contra corpus...")
        text = generate(
            model,
            prompt="The quick brown",
            max_new_tokens=80,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
            seed=1,
            device=device,
        )
        verifier = ConsistencyVerifier()
        problems = verifier.verify_against_corpus(text, graph, observer)
        verdict = verifier.verdict(problems)
        logger.info(f"Veredicto: {verdict}")
        for p in problems:
            logger.info(f"  - [{p.type}] {p.note}")

    logger.info("Entrenamiento terminado.")


if __name__ == "__main__":
    main()
