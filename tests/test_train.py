"""Tests del bucle de entrenamiento y evaluación."""
import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from snn_lm_v3 import GPTConfig, SNNLM, train, evaluate


def _make_loaders(n_samples: int = 8, seq_len: int = 8, batch_size: int = 2):
    x = torch.randint(0, 100, (n_samples, seq_len))
    y = torch.randint(0, 100, (n_samples, seq_len))
    dl = DataLoader(TensorDataset(x, y), batch_size=batch_size)
    return dl


def _small_cfg(tmp_path, **kw):
    base = dict(
        n_layer=1, n_head=1, n_embd=16, seq_len=8, vocab_size=100,
        batch_size=2, max_steps=2, warmup_steps=1, log_every=1,
        eval_every=0, ckpt_every=0, eval_batches=2,
        use_amp=False, device="cpu", out_dir=str(tmp_path),
    )
    base.update(kw)
    return GPTConfig(**base)


def test_train_smoke(tmp_path):
    cfg = _small_cfg(tmp_path)
    model = SNNLM(cfg)
    dl = _make_loaders()
    trained = train(model, dl, dl, cfg, logger=None)
    assert trained is not None
    assert math.isfinite(
        evaluate(trained, dl, "cpu", max_batches=2)["loss"]
    )


def test_evaluate_returns_metrics():
    cfg = _small_cfg("/tmp")  # noqa: S108 - cfg needs out_dir, unused here
    model = SNNLM(cfg)
    dl = _make_loaders()
    metrics = evaluate(model, dl, "cpu", max_batches=2)
    assert "loss" in metrics
    assert "ppl" in metrics
    assert "n" in metrics
    assert metrics["n"] == 2
    assert metrics["loss"] > 0
    assert math.isfinite(metrics["ppl"])


def test_train_saves_checkpoints(tmp_path):
    cfg = _small_cfg(tmp_path, ckpt_every=1, max_steps=2)
    model = SNNLM(cfg)
    dl = _make_loaders()
    train(model, dl, dl, cfg, logger=None)
    assert (tmp_path / "final.pt").exists()


def test_train_with_grad_accum(tmp_path):
    cfg = _small_cfg(tmp_path, grad_accum_steps=2, max_steps=2)
    model = SNNLM(cfg)
    dl = _make_loaders(n_samples=8, batch_size=2)
    trained = train(model, dl, dl, cfg, logger=None)
    assert trained is not None


def test_train_resume(tmp_path):
    cfg = _small_cfg(tmp_path, ckpt_every=1, max_steps=2)
    model = SNNLM(cfg)
    dl = _make_loaders()
    train(model, dl, dl, cfg, logger=None)

    from snn_lm_v3.checkpoint import load_checkpoint
    ckpt = load_checkpoint(
        str(tmp_path / "final.pt"), SNNLM, "cpu", load_optimizer=True
    )
    model2 = ckpt["model"]
    start = ckpt["step"]
    cfg2 = _small_cfg(tmp_path, max_steps=start + 2, ckpt_every=0)
    trained = train(
        model2, dl, dl, cfg2, logger=None, start_step=start,
        optimizer_state=ckpt["optimizer_state"],
        scheduler_state=ckpt["scheduler_state"],
        best_val_loss=ckpt["best_val_loss"],
    )
    assert trained is not None
