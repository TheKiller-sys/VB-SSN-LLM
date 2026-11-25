"""Entrena SNN-LM v4.1 con o sin RT, con o sin código, con split por repo."""
from __future__ import annotations

import argparse
import logging
import os

import torch

from snn_lm_v3 import (
    GPTConfig, SIZES, SNNLM,
    make_dataloader, make_rt_dataloader,
    train, resolve_device, load_texts_with_paths,
    CorpusObserver, CodeCorpusObserver, ReasoningGraph,
    ConsistencyVerifier, save_graph, generate,
    ResonantVocab, RVConfig, RTModelWrapper,
)
from snn_lm_v3.dataset import CODE_EXTENSIONS
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
    p.add_argument("--snn-mode", choices=["static", "multi_step", "temporal", "gelu"], default=None)
    p.add_argument("--snn-steps", type=int, default=None)
    p.add_argument("--snn-track-stats", action="store_true")
    p.add_argument("--analyzer", action="store_true",
                   help="Activar ContinuousAnalyzer (prune + codebook refresh).")
    p.add_argument("--analyzer-interval", type=float, default=60.0)

    # Razonamiento
    p.add_argument("--no-reasoning", action="store_true")
    p.add_argument("--code-corpus", action="store_true",
                   help="Usar CodeCorpusObserver (para corpus de código).")
    p.add_argument("--reasoning-chunk-size", type=int, default=500)

    # Corpus / split
    p.add_argument("--split-by", choices=["file", "repo"], default="file",
                   help="Split train/val: 'file' o 'repo' (evita val leak).")
    p.add_argument("--extensions", nargs="+", default=[".txt"],
                   help="Extensiones a incluir. Usa '.py' '.js' para código.")

    # RT
    p.add_argument("--use-rt", action="store_true")
    p.add_argument("--bpe-path", default="checkpoints/bpe_code.json")
    p.add_argument("--n-concepts", type=int, default=12_092)
    p.add_argument("--concept-dim", type=int, default=132)
    p.add_argument("--aux-loss-weight", type=float, default=0.1)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_reasoning_layer(
    data_paths, chunk_size, logger, enc_name, use_code, extensions,
):
    """Construye el observer adecuado (prosa o código) y mina reglas."""
    if use_code:
        observer = CodeCorpusObserver(enc_name=enc_name)
        logger.info("Usando CodeCorpusObserver (corpus de código)")
    else:
        observer = CorpusObserver(enc_name=enc_name)
        logger.info("Usando CorpusObserver (prosa)")

    graph = ReasoningGraph(cooldown_steps=3)
    docs = load_texts_with_paths(data_paths, extensions=tuple(extensions))
    logger.info(f"Observando {len(docs)} documentos del corpus...")
    for i, (path, content) in enumerate(docs):
        for j in range(0, len(content), chunk_size):
            observer.observe(content[j:j + chunk_size], ref=f"{path}:{j}")

    state = observer.get_state()
    logger.info(f"Observer state: {state}")
    n_rules = graph.mine_from_corpus(observer)
    logger.info(
        f"ReasoningGraph: {n_rules} reglas nuevas, "
        f"{graph.get_state()['rules_count']} totales "
        f"({graph.metrics['cyclic_rejected']} cíclicas, "
        f"{graph.metrics['overlap_rejected']} overlap, "
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
        "split_by": args.split_by,
        "code_corpus": args.code_corpus or None,
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
    if args.analyzer:
        cfg.analyzer_enabled = True
        cfg.analyzer_interval_s = args.analyzer_interval

    # ── RT setup ──
    if args.use_rt:
        cfg.use_rt = True
        cfg.rt_bpe_path = args.bpe_path
        cfg.rt_n_concepts = args.n_concepts
        cfg.rt_concept_dim = args.concept_dim
        cfg.rt_aux_loss_weight = args.aux_loss_weight
        rv_probe = ResonantVocab.from_bpe(args.bpe_path)
        cfg.rt_bpe_vocab_size = rv_probe.vocab_size
        cfg.vocab_size = rv_probe.vocab_size
        print(f"[RT] BPE vocab={rv_probe.vocab_size}, n_concepts={args.n_concepts}")

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

    # ── Extensiones: si --code-corpus y no se especificaron, usar defaults de código
    extensions = args.extensions
    if args.code_corpus and extensions == [".txt"]:
        extensions = list(CODE_EXTENSIONS)
        logger.info(f"Extensiones de código auto: {len(extensions)} tipos")

    # ── Razonamiento ──
    observer, graph = None, None
    if not args.no_reasoning:
        try:
            observer, graph = build_reasoning_layer(
                args.data, args.reasoning_chunk_size, logger,
                enc_name="cl100k_base",
                use_code=args.code_corpus,
                extensions=extensions,
            )
            save_graph(graph, os.path.join(cfg.out_dir, "graph.pt"))
            logger.info(f"Grafo guardado en {os.path.join(cfg.out_dir, 'graph.pt')}")
        except Exception as e:
            logger.warning(f"Razonamiento desactivado: {e}")
            observer, graph = None, None

    # ── Modelo ──
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
        if optimizer_state is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=(cfg.beta1, cfg.beta2), eps=1e-8,
            )
    else:
        snnlm = SNNLM(cfg)
        if cfg.use_rt:
            rv = ResonantVocab.from_bpe(cfg.rt_bpe_path, RVConfig(
                n_concepts=cfg.rt_n_concepts,
                concept_dim=cfg.rt_concept_dim,
                embedding_dim=cfg.n_embd,
                ema_decay=cfg.rt_ema_decay,
                commitment_beta=cfg.rt_commitment_beta,
                min_count_for_update=cfg.rt_min_count_for_update,
                concept_top_k_tokens=cfg.rt_concept_top_k_tokens,
            ))
            model = RTModelWrapper(snnlm, rv, aux_loss_weight=cfg.rt_aux_loss_weight)
        else:
            model = snnlm

    logger.info(f"Parámetros: {model.num_params():,}")

    # ── Dataloaders ──
    if cfg.use_rt:
        train_dl, val_dl = make_rt_dataloader(
            args.data, cfg.seq_len, cfg.batch_size, model.rv,
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
    logger.info(f"Train batches/epoch: {len(train_dl)}, Val: {len(val_dl)}")

    model = train(
        model, train_dl, val_dl, cfg, logger,
        start_step=start_step, optimizer=optimizer,
        optimizer_state=optimizer_state, scheduler_state=scheduler_state,
        best_val_loss=best_val_loss,
    )

    # ── Verificación final ──
    if graph is not None and observer is not None:
        logger.info("Verificación final contra corpus...")
        text = generate(
            model, prompt="def " if args.code_corpus else "The quick brown",
            max_new_tokens=80, temperature=0.8, top_k=40, top_p=0.95,
            seed=1, device=device,
        )
        verifier = ConsistencyVerifier()
        problems = verifier.verify_against_corpus(text, graph, observer)
        logger.info(f"Veredicto: {verifier.verdict(problems)}")
        for p in problems:
            logger.info(f"  - [{p.type}] {p.note}")

    logger.info("Entrenamiento terminado.")


if __name__ == "__main__":
    main()
