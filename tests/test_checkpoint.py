"""Tests de checkpoints: roundtrip, atomicidad, optimizer, RNG."""
import os
import tempfile

import torch

from snn_lm_v3 import GPTConfig, SNNLM, save_checkpoint, load_checkpoint


def _cfg(**kw):
    base = dict(n_layer=1, n_head=1, n_embd=16, seq_len=16, vocab_size=100)
    base.update(kw)
    return GPTConfig(**base)


def test_save_load_roundtrip():
    torch.manual_seed(0)
    cfg = _cfg()
    model = SNNLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(model, optimizer, None, cfg, step=42, path=path)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")

        ckpt = load_checkpoint(path, SNNLM, "cpu")
        assert ckpt["step"] == 42
        assert ckpt["config"].n_layer == 1
        assert isinstance(ckpt["model"], SNNLM)

        for (n1, p1), (n2, p2) in zip(
            model.state_dict().items(), ckpt["model"].state_dict().items()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


def test_load_with_optimizer():
    cfg = _cfg()
    model = SNNLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(model, optimizer, None, cfg, step=5, path=path)
        ckpt = load_checkpoint(path, SNNLM, "cpu", load_optimizer=True)
        assert ckpt["optimizer_state"] is not None
        assert ckpt["scheduler_state"] is None


def test_save_checkpoint_creates_dir():
    cfg = _cfg()
    model = SNNLM(cfg)
    with tempfile.TemporaryDirectory() as d:
        nested = os.path.join(d, "a", "b", "ckpt.pt")
        save_checkpoint(model, None, None, cfg, step=1, path=nested)
        assert os.path.exists(nested)


def test_load_nonexistent_raises():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "missing.pt")
        try:
            load_checkpoint(path, SNNLM, "cpu")
            raised = False
        except FileNotFoundError:
            raised = True
        assert raised


def test_best_val_loss_preserved():
    cfg = _cfg()
    model = SNNLM(cfg)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        save_checkpoint(
            model, None, None, cfg, step=10, path=path, best_val_loss=3.14
        )
        ckpt = load_checkpoint(path, SNNLM, "cpu")
        assert ckpt["best_val_loss"] == 3.14
