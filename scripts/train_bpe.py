#!/usr/bin/env python
"""CLI para entrenar BPE sobre un corpus de código.

Uso:
    python scripts/train_bpe.py --corpus data/code/ --vocab-size 40000 --out checkpoints/bpe_code.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snn_lm_v4.tokenizer.bpe_trainer import train_bpe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--vocab-size", type=int, default=40_000)
    p.add_argument("--out", required=True)
    p.add_argument("--min-frequency", type=int, default=2)
    args = p.parse_args()

    print(f"Entrenando BPE vocab={args.vocab_size} sobre {args.corpus}...")
    path = train_bpe(args.corpus, args.vocab_size, args.out, args.min_frequency)
    print(f"OK → {path}")

    # Smoke test
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    test = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr"
    ids = tok.encode(test).ids
    ok = tok.decode(ids) == test
    print(f"Roundtrip: {'OK' if ok else 'FAIL'} | {len(ids)} tokens para {len(test)} chars")


if __name__ == "__main__":
    main()
