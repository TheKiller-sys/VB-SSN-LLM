"""
Genera texto desde un checkpoint.

Ejemplo:
    python scripts/generate.py --ckpt checkpoints/latest.pt \
        --prompt "Once upon a time" --max-tokens 200 --temperature 0.8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from snn_lm_v3 import SNNLM, generate
from snn_lm_v3.checkpoint import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, step = load_checkpoint(args.ckpt, SNNLM, device)
    print(f"Checkpoint step={step}, params={model.num_params():,}, device={device}")

    for i in range(args.samples):
        seed = None if args.seed is None else args.seed + i
        text = generate(
            model, prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p,
            seed=seed, device=device,
        )
        print(f"\n--- muestra {i+1} ---")
        print(text)


if __name__ == "__main__":
    main()
