"""Configuración completa del modelo, validación y resolución de device."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
import warnings


def resolve_device(device: str | None = "auto"):
    """Devuelve un `torch.device`. `auto` -> cuda > mps > cpu."""
    import torch

    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class GPTConfig:
    # ---- modelo ----
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    seq_len: int = 512
    dropout: float = 0.1
    tie_weights: bool = True
    init_std: float = 0.02

    # ---- FFN de spikes ----
    use_spiking: bool = True
    spiking_mult: int = 4
    spiking_threshold: float = 1.0
    spiking_alpha: float = 2.0
    spiking_residual: bool = True

    # ---- dinámica LIF ----
    snn_mode: str = "multi_step"          # static | multi_step | temporal | gelu
    snn_num_steps: int = 4
    snn_decay: float = 0.9
    snn_reset_mode: str = "soft"          # soft | hard
    snn_learnable_threshold: bool = False
    snn_learnable_decay: bool = False
    snn_track_stats: bool = False

    # ---- entrenamiento ----
    batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 500
    max_steps: int = 100_000
    grad_clip: float = 1.0
    use_amp: bool = True

    # ---- sistema ----
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    eval_batches: int = 50
    ckpt_every: int = 500
    out_dir: str = "checkpoints"
    device: str = "auto"
    compile: bool = False
    num_workers: int = 0
    pin_memory: bool = True

    def validate(self) -> None:
        checks = [
            (self.vocab_size > 0, "vocab_size must be > 0"),
            (self.n_layer > 0, "n_layer must be > 0"),
            (self.n_head > 0, "n_head must be > 0"),
            (self.n_embd > 0, "n_embd must be > 0"),
            (self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"),
            (self.seq_len > 0, "seq_len must be > 0"),
            (0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)"),
            (self.lr > 0, "lr must be > 0"),
            (0 <= self.min_lr <= self.lr, "min_lr must be in [0, lr]"),
            (self.max_steps > 0, "max_steps must be > 0"),
            (self.warmup_steps >= 0, "warmup_steps must be >= 0"),
            (self.grad_accum_steps >= 1, "grad_accum_steps must be >= 1"),
            (self.batch_size > 0, "batch_size must be > 0"),
            (self.spiking_mult >= 1, "spiking_mult must be >= 1"),
            (self.spiking_alpha > 0, "spiking_alpha must be > 0"),
            (
                self.snn_mode in ("static", "multi_step", "temporal", "gelu"),
                f"invalid snn_mode: {self.snn_mode}",
            ),
            (
                self.snn_reset_mode in ("soft", "hard"),
                f"invalid snn_reset_mode: {self.snn_reset_mode}",
            ),
            (self.snn_num_steps >= 1, "snn_num_steps must be >= 1"),
            (0.0 < self.snn_decay < 1.0, "snn_decay must be in (0, 1)"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

        if self.snn_mode == "temporal" and self.use_spiking:
            warnings.warn(
                "snn_mode='temporal' uses BPTT over the full sequence: "
                "slow and memory-heavy. Prefer 'multi_step' for production.",
                RuntimeWarning,
                stacklevel=3,
            )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GPTConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.validate()
        return cfg


SIZE_TINY = dict(n_layer=6, n_head=8, n_embd=256)
SIZE_SMALL = dict(n_layer=8, n_head=8, n_embd=512)
SIZE_MEDIUM = dict(n_layer=12, n_head=12, n_embd=768)
SIZE_LARGE = dict(n_layer=24, n_head=16, n_embd=1024)

SIZES = {
    "tiny": SIZE_TINY,
    "small": SIZE_SMALL,
    "medium": SIZE_MEDIUM,
    "large": SIZE_LARGE,
}
