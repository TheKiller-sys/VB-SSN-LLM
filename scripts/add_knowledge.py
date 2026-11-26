"""Añade documentos a un KnowledgeBase para RAG.

Sin cambios respecto a v4.2.1: la KB es independiente del core del modelo.

Ejemplos:
    python scripts/add_knowledge.py --corpus data/noticias/ --out checkpoints/kb.pt \\
        --encoder hidden --ckpt checkpoints/best.pt

    python scripts/add_knowledge.py --kb checkpoints/kb.pt \\
        --corpus data/articulos/ --encoder hidden --ckpt checkpoints/best.pt

    python scripts/add_knowledge.py --corpus data/ --out checkpoints/kb.pt \\
        --encoder sentence_transformer:all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_lm_v4 import (
    SNNLM, KnowledgeBase, load_checkpoint, resolve_device,
    make_model_encoder, make_sentence_transformer_encoder,
)
from snn_lm_v4.dataset import load_texts_with_paths


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--kb", default=None, help="KB existente a la que añadir")
    src.add_argument("--out", default=None, help="KB nueva a crear (requiere --corpus)")
    p.add_argument("--corpus", nargs="+", default=None)
    p.add_argument("--encoder", default="hidden",
                   help="'hidden' o 'sentence_transformer:<name>'")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint del modelo (requerido si --encoder hidden)")
    p.add_argument("--extensions", nargs="+", default=[".txt"])
    p.add_argument("--chunk-size", type=int, default=1000,
                   help="Tamaño de chunk en caracteres (0 = sin chunking)")
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--max-docs", type=int, default=100_000)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or len(text) <= size:
        return [text]
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(text), step):
        chunks.append(text[i : i + size])
        if i + size >= len(text):
            break
    return chunks


def build_encoder(args):
    if args.encoder == "hidden":
        if not args.ckpt:
            raise SystemExit("--encoder hidden requires --ckpt")
        device = resolve_device(args.device)
        ckpt = load_checkpoint(args.ckpt, SNNLM, device)
        model = ckpt["model"]
        cfg = ckpt["config"]
        print(
            f"Encoder: hidden states de {args.ckpt} (dim={cfg.n_embd}, "
            f"core={cfg.core_type})"
        )
        return make_model_encoder(model), cfg.n_embd
    if args.encoder.startswith("sentence_transformer:"):
        name = args.encoder.split(":", 1)[1]
        print(f"Encoder: sentence-transformers {name}")
        enc = make_sentence_transformer_encoder(name)
        probe = enc("dimension probe")
        return enc, probe.shape[0]
    raise SystemExit(f"Unknown encoder: {args.encoder}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    if not args.corpus:
        if args.kb:
            kb = KnowledgeBase.load(args.kb)
            print(f"KB cargado: {len(kb)} docs, dim={kb.dim}")
            return
        raise SystemExit("Need --corpus (to add) or --kb (to inspect)")

    encoder, dim = build_encoder(args)

    if args.kb and os.path.exists(args.kb):
        kb = KnowledgeBase.load(args.kb, encoder_fn=encoder)
        print(f"KB existente: {len(kb)} docs")
    else:
        if not args.out:
            raise SystemExit("--out required when creating new KB")
        kb = KnowledgeBase(dim=dim, encoder_fn=encoder, max_docs=args.max_docs)
        print(f"KB nueva: dim={dim}")

    docs = load_texts_with_paths(args.corpus, extensions=tuple(args.extensions))
    print(f"Documentos cargados: {len(docs)}")

    all_chunks: list[str] = []
    for path, content in docs:
        all_chunks.extend(chunk_text(content, args.chunk_size, args.chunk_overlap))
    print(f"Chunks a indexar: {len(all_chunks)}")

    added = kb.add(all_chunks)
    print(f"Añadidos: {added} docs. Total en KB: {len(kb)}")

    out_path = args.out or args.kb
    kb.save(out_path)
    print(f"KB guardado en {out_path}")


if __name__ == "__main__":
    main()
