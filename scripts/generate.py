"""Genera texto desde un checkpoint, con RAG y/o LoRA opcionales."""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_lm_v3 import (
    SNNLM, generate, resolve_device, load_texts,
    CorpusObserver, ReasoningGraph, ConsistencyVerifier,
    SignalProbe, load_graph,
    KnowledgeBase, make_model_encoder, make_sentence_transformer_encoder,
    apply_lora, load_lora, merge_lora, is_lora_active,
)
from snn_lm_v3.checkpoint import load_checkpoint


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
    p.add_argument("--corpus", default=None)
    p.add_argument("--graph", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--chunk-size", type=int, default=500)

    # RAG
    p.add_argument("--rag", action="store_true", help="Activar RAG")
    p.add_argument("--kb", default=None, help="Ruta al KnowledgeBase (.pt)")
    p.add_argument("--rag-k", type=int, default=4)
    p.add_argument("--rag-max-chars", type=int, default=2000)
    p.add_argument("--rag-encoder", default="hidden",
                   help="'hidden' o 'sentence_transformer:<name>'")

    # LoRA
    p.add_argument("--lora", default=None, help="Ruta al adapter LoRA")
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-targets", nargs="+", default=["qkv", "proj", "fc1", "fc2"])
    p.add_argument("--merge-lora", action="store_true",
                   help="Fusionar LoRA en los pesos base tras cargar")
    return p.parse_args()


def build_or_load_graph(args):
    if args.graph and os.path.exists(args.graph):
        return load_graph(args.graph), None
    if not args.corpus:
        return None, None
    observer = CorpusObserver(enc_name="cl100k_base")
    docs = load_texts([args.corpus])
    for i, doc in enumerate(docs):
        for j in range(0, len(doc), args.chunk_size):
            observer.observe(doc[j:j + args.chunk_size], ref=f"ref{i}_{j}")
    graph = ReasoningGraph(cooldown_steps=3)
    graph.mine_from_corpus(observer)
    return graph, observer


def build_rag_kb(args, model):
    if not args.rag:
        return None
    if not args.kb:
        raise SystemExit("--rag requires --kb")
    if not os.path.exists(args.kb):
        raise SystemExit(f"KB not found: {args.kb}")

    if args.rag_encoder == "hidden":
        encoder = make_model_encoder(model)
    elif args.rag_encoder.startswith("sentence_transformer:"):
        name = args.rag_encoder.split(":", 1)[1]
        encoder = make_sentence_transformer_encoder(name)
    else:
        raise SystemExit(f"Unknown --rag-encoder: {args.rag_encoder}")

    kb = KnowledgeBase.load(args.kb, encoder_fn=encoder)
    print(f"RAG: KB cargado ({len(kb)} docs, dim={kb.dim})")
    return kb


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    device = resolve_device(args.device)
    ckpt = load_checkpoint(args.ckpt, SNNLM, device)
    model, cfg, step = ckpt["model"], ckpt["config"], ckpt["step"]
    print(f"Checkpoint step={step}, params={model.num_params():,}, RT={ckpt['use_rt']}")

    # LoRA: aplicar sobre el modelo cargado
    if args.lora:
        if not os.path.exists(args.lora):
            raise SystemExit(f"LoRA adapter not found: {args.lora}")
        if not is_lora_active(model):
            apply_lora(
                model, r=args.lora_r, alpha=args.lora_alpha,
                targets=tuple(args.lora_targets),
            )
        meta = load_lora(model, args.lora)
        print(f"LoRA cargado: {args.lora} (meta={meta})")
        if args.merge_lora:
            n = merge_lora(model)
            print(f"LoRA fusionado en {n} módulos")
        model.eval()

    # RAG
    rag_kb = build_rag_kb(args, model)

    # Grafo
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
            model, prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            seed=seed, device=device,
            use_cache=not args.no_cache,
            rag_kb=rag_kb,
            rag_top_k=args.rag_k,
            rag_max_context_chars=args.rag_max_chars,
        )
        print(f"\n--- muestra {i + 1} ---")
        print(text)

        if verifier is not None and graph is not None and observer is not None:
            problems = verifier.verify_against_corpus(text, graph, observer)
            if probe is not None:
                signals = probe.probe(text)
                problems += verifier.verify(text, signals)
            print(f"[veredicto: {verifier.verdict(problems)}]")
            for p in problems:
                print(f"  - [{p.type}] {p.note}")


if __name__ == "__main__":
    main()
