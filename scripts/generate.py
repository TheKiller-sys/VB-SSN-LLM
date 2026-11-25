"""Genera texto desde un checkpoint con verificación opcional contra corpus.

Ejemplos:
    snnlm-generate --ckpt checkpoints/best.pt --prompt "Once upon a time"
    snnlm-generate --ckpt checkpoints/best.pt --prompt "Once" --corpus data/ --verify
"""
from __future__ import annotations

import argparse
import logging
import os

from snn_lm_v3 import (
    SNNLM,
    generate,
    resolve_device,
    load_texts,
    CorpusObserver,
    ReasoningGraph,
    ConsistencyVerifier,
    SignalProbe,
)
from snn_lm_v3.checkpoint import load_checkpoint
from snn_lm_v3.reasoning import load_graph


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--corpus", default=None,
                   help="Directorio/archivo .txt para construir el grafo.")
    p.add_argument("--graph", default=None,
                   help="Ruta a un graph.pt guardado (si no, se construye de --corpus).")
    p.add_argument("--verify", action="store_true",
                   help="Verificar la generación contra el corpus y las señales.")
    p.add_argument("--chunk-size", type=int, default=500)
    return p.parse_args()


def build_or_load_graph(args):
    if args.graph and os.path.exists(args.graph):
        return load_graph(args.graph), None

    if not args.corpus:
        return None, None

    observer = CorpusObserver(enc_name="gpt2")
    docs = load_texts([args.corpus])
    for i, doc in enumerate(docs):
        for j in range(0, len(doc), args.chunk_size):
            observer.observe(doc[j : j + args.chunk_size], ref=f"ref{i}_{j}")
    graph = ReasoningGraph(cooldown_steps=3)
    graph.mine_from_corpus(observer)
    return graph, observer


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.ckpt, SNNLM, device)
    model, cfg, step = ckpt["model"], ckpt["config"], ckpt["step"]
    print(f"Checkpoint step={step}, params={model.num_params():,}, device={device}")

    graph, observer = build_or_load_graph(args)
    if graph is not None:
        print(f"Grafo: {graph.get_state()['rules_count']} reglas")

    verifier = ConsistencyVerifier() if args.verify else None
    probe = None
    if args.verify and cfg.snn_track_stats:
        probe = SignalProbe(model, device)

    for i in range(args.samples):
        seed = None if args.seed is None else args.seed + i
        text = generate(
            model,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=seed,
            device=device,
            use_cache=not args.no_cache,
        )
        print(f"\n--- muestra {i + 1} ---")
        print(text)

        if verifier is not None and graph is not None and observer is not None:
            problems = verifier.verify_against_corpus(text, graph, observer)
            if probe is not None:
                signals = probe.probe(text)
                problems += verifier.verify(text, signals)
            verdict = verifier.verdict(problems)
            print(f"[veredicto: {verdict}]")
            for p in problems:
                print(f"  - [{p.type}] {p.note}")


if __name__ == "__main__":
    main()
