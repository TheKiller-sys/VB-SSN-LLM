"""
Demo end-to-end: genera corpus sintético, entrena, evalúa, razona y genera.

    python main.py
    python main.py --size small --steps 1000 --device cuda
    python main.py --size tiny --steps 100 --no-reasoning
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile

from snn_lm_v3 import (
    GPTConfig, SIZES, SNNLM, make_dataloader, train, evaluate, generate,
    resolve_device,
    CorpusObserver, ReasoningGraph, ConsistencyVerifier,
)

DEMO = """The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
All that glitters is not gold.
The pen is mightier than the sword.
Actions speak louder than words.
Knowledge is power.
Time is money.
Practice makes perfect.
Every cloud has a silver lining.
""" * 500


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=list(SIZES.keys()), default="tiny")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-reasoning", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger("demo")

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "demo.txt"), "w", encoding="utf-8") as f:
            f.write(DEMO)

        # ---------------- razonamiento sobre corpus ----------------
        observer, graph = None, None
        if not args.no_reasoning:
            observer = CorpusObserver(enc_name="gpt2")
            for j in range(0, len(DEMO), 500):
                observer.observe(DEMO[j : j + 500], ref=f"demo_{j}")
            graph = ReasoningGraph(cooldown_steps=3)
            n_rules = graph.mine_from_corpus(observer)
            logger.info(
                f"CorpusObserver: {observer.get_state()['vocabulary_size']} tokens, "
                f"{observer.get_state()['transitions']} transiciones"
            )
            logger.info(
                f"ReasoningGraph: {n_rules} reglas minadas, "
                f"{graph.get_state()['rules_count']} totales"
            )

        # ---------------- modelo ----------------
        cfg = GPTConfig(
            **SIZES[args.size],
            seq_len=128,
            batch_size=8,
            lr=3e-4,
            min_lr=3e-5,
            max_steps=args.steps,
            warmup_steps=min(20, max(1, args.steps // 10)),
            log_every=max(1, args.steps // 10),
            eval_every=max(50, args.steps // 2),
            ckpt_every=max(50, args.steps // 2),
            eval_batches=10,
            out_dir=os.path.join(tmp, "ckpt"),
            device=args.device,
            use_amp=False,
            snn_track_stats=True,
        )
        cfg.validate()

        device = resolve_device(cfg.device)
        model = SNNLM(cfg).to(device)
        logger.info(f"Size={args.size} params={model.num_params():,} device={device}")

        train_dl, val_dl, _ = make_dataloader(
            [data_dir], cfg.seq_len, cfg.batch_size, seed=42
        )
        logger.info(f"Train batches={len(train_dl)} Val batches={len(val_dl)}")

        model = train(model, train_dl, val_dl, cfg, logger)

        metrics = evaluate(model, val_dl, device)
        logger.info(f"Eval final: {metrics}")

        # ---------------- generación + verificación ----------------
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
        logger.info(f"\n{text}")

        if graph is not None and observer is not None:
            verifier = ConsistencyVerifier()
            problems = verifier.verify_against_corpus(text, graph, observer)
            verdict = verifier.verdict(problems)
            logger.info(f"Veredicto: {verdict}")
            for p in problems:
                logger.info(f"  - [{p.type}] {p.note}")


if __name__ == "__main__":
    main()
