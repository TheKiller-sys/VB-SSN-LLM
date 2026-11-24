"""
Demo end-to-end: genera corpus sintético, entrena tiny, evalúa, genera.
Ejecuta: python main.py
"""
import logging
import os
import tempfile

import torch

from snn_lm_v3 import (
    GPTConfig, SIZE_TINY, SNNLM,
    make_dataloader, train, evaluate, generate,
)
from snn_lm_v3.checkpoint import save_checkpoint


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
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    logger = logging.getLogger("demo")

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        with open(os.path.join(data_dir, "demo.txt"), "w", encoding="utf-8") as f:
            f.write(DEMO)

        cfg = GPTConfig(
            **SIZE_TINY,
            seq_len=128, batch_size=8, lr=3e-4,
            max_steps=200, warmup_steps=20,
            log_every=20, eval_every=100, ckpt_every=100,
            out_dir=os.path.join(tmp, "ckpt"),
            device="cpu", use_amp=False,
        )

        device = torch.device(cfg.device)
        model = SNNLM(cfg).to(device)
        logger.info(f"Parámetros: {model.num_params():,}")

        train_dl, val_dl, enc = make_dataloader(
            [data_dir], cfg.seq_len, cfg.batch_size, seed=42,
        )
        logger.info(f"Train batches: {len(train_dl)}")

        model = train(model, train_dl, val_dl, cfg, logger)

        metrics = evaluate(model, val_dl, device)
        logger.info(f"Eval final: {metrics}")

        logger.info("Generación:")
        text = generate(
            model, prompt="The quick brown",
            max_new_tokens=80, temperature=0.8, top_k=40, top_p=0.95,
            seed=1, device=device,
        )
        logger.info(f"\n{text}")


if __name__ == "__main__":
    main()
