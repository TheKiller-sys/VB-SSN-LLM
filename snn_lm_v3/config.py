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
    snn_mode: str = "multi_step"
    snn_num_steps: int = 4
    snn_decay: float = 0.9
    snn_reset_mode: str = "soft"
    snn_learnable_threshold: bool = False
    snn_learnable_decay: bool = False
    snn_track_stats: bool = False

    # ---- memoria neural (v4.0) ----
    neural_memory_enabled: bool = True
    neural_memory_max_size: int = 10_000
    neural_memory_k: int = 8
    neural_memory_hebbian_lr: float = 1e-3
    neural_memory_hebbian_decay: float = 0.999
    neural_memory_norm_clamp: float = 100.0
    neural_memory_use_fast_weights: bool = True
    neural_memory_use_knn: bool = True
    neural_memory_append_batch_size: int = 4096

    # ---- abstracción geométrica (v4.0) ----
    abstraction_enabled: bool = True
    abstraction_n_codes: int = 256
    abstraction_ema_decay: float = 0.99
    abstraction_min_count_for_update: float = 0.1
    abstraction_loss_weight: float = 0.05
    abstraction_residual_scale: float = 1.0

    # ---- autoconocimiento (v4.0) ----
    self_knowledge_enabled: bool = True
    self_knowledge_hidden: int = 0  # 0 = usar n_embd
    self_knowledge_loss_weight: float = 0.1

    # ---- razonador latente (v4.0) ----
    latent_reasoner_enabled: bool = False
    latent_reasoner_heads: int = 4
    latent_reasoner_layers: int = 2

    # ---- analizador en background (v4.0) ----
    analyzer_enabled: bool = False
    analyzer_interval_s: float = 60.0
    analyzer_prune_memory: bool = True
    analyzer_refresh_codebook: bool = True

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

    # En GPTConfig, dentro de la sección "---- sistema ----" o antes:
    code_corpus: bool = False       # NUEVO: activa CodeCorpusObserver
    split_by: str = "file"          # NUEVO: "file" | "repo"

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
            # memoria neural
            (self.neural_memory_max_size > 0, "neural_memory_max_size must be > 0"),
            (self.neural_memory_k >= 1, "neural_memory_k must be >= 1"),
            (
                self.neural_memory_k <= self.neural_memory_max_size,
                "neural_memory_k must be <= neural_memory_max_size",
            ),
            (self.neural_memory_hebbian_lr >= 0, "neural_memory_hebbian_lr must be >= 0"),
            (0.0 <= self.neural_memory_hebbian_decay <= 1.0,
             "neural_memory_hebbian_decay must be in [0, 1]"),
            (self.neural_memory_norm_clamp > 0, "neural_memory_norm_clamp must be > 0"),
            (self.neural_memory_append_batch_size > 0,
             "neural_memory_append_batch_size must be > 0"),
            # abstracción
            (self.abstraction_n_codes >= 2, "abstraction_n_codes must be >= 2"),
            (0.0 <= self.abstraction_ema_decay < 1.0,
             "abstraction_ema_decay must be in [0, 1)"),
            (self.abstraction_loss_weight >= 0, "abstraction_loss_weight must be >= 0"),
            (self.abstraction_residual_scale >= 0, "abstraction_residual_scale must be >= 0"),
            # autoconocimiento
            (self.self_knowledge_loss_weight >= 0,
             "self_knowledge_loss_weight must be >= 0"),
            (self.self_knowledge_hidden >= 0, "self_knowledge_hidden must be >= 0"),
            # razonador
            (self.latent_reasoner_heads >= 1, "latent_reasoner_heads must be >= 1"),
            (self.n_embd % self.latent_reasoner_heads == 0,
             "n_embd must be divisible by latent_reasoner_heads"),
            (self.latent_reasoner_layers >= 1, "latent_reasoner_layers must be >= 1"),
            # analyzer
            (self.analyzer_interval_s > 0, "analyzer_interval_s must be > 0"),
            # Dentro de la lista `checks`:
            (self.split_by in ("file", "repo"), f"invalid split_by: {self.split_by}"),
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

        if self.compile and self.neural_memory_enabled:
            warnings.warn(
                "torch.compile + neural_memory may cause graph breaks or "
                "buffer-mutation issues. Consider disabling compile if you "
                "observe failures.",
                RuntimeWarning,
                stacklevel=3,
            )

        if self.temporal_mode_uses_memory():
            warnings.warn(
                "snn_mode='temporal' + neural_memory_enabled: memory update "
                "happens once per forward over the whole sequence, which may "
                "dominate runtime. Consider snn_mode='multi_step'.",
                RuntimeWarning,
                stacklevel=3,
            )

    def temporal_mode_uses_memory(self) -> bool:
        return self.snn_mode == "temporal" and self.neural_memory_enabled

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
