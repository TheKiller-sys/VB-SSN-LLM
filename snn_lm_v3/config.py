"""Configuración completa del modelo, validación y resolución de device.

v4.3:
  - Añade `core_type` ("attention" | "temporal") con TemporalCore.
  - Añade parámetros de TemporalCore (expand, LIF decay, chunk).
  - Añade `ffn_mode` ("dense" | "spiking" | "hybrid") con FFN híbrido.
  - Añade `reasoning_bias_*` para bias de reglas en logits.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import os
import warnings


def resolve_device(device: str | None = "auto"):
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

    # ---- core (v4.3) ----
    core_type: str = "temporal"  # "attention" | "temporal"

    # ---- TemporalCore (v4.3) ----
    temporal_expand: int = 2
    temporal_lif_threshold: float = 1.0
    temporal_lif_decay: float = 0.9
    temporal_lif_alpha: float = 2.0
    temporal_chunk_size: int = 64
    temporal_use_spike_decay: bool = True
    temporal_decay_base: float = 0.9
    temporal_decay_target: float = 0.99

    # ---- Resonant Tokenizer (v4.1) ----
    use_rt: bool = False
    rt_bpe_vocab_size: int = 40_000
    rt_n_concepts: int = 12_092
    rt_concept_dim: int = 132
    rt_aux_loss_weight: float = 0.1
    rt_bpe_path: str = "checkpoints/bpe_code.json"
    rt_ema_decay: float = 0.99
    rt_commitment_beta: float = 0.25
    rt_min_count_for_update: float = 0.1
    rt_concept_top_k_tokens: int = 8

    # ---- FFN ----
    ffn_mode: str = "hybrid"  # "dense" | "spiking" | "hybrid" (v4.3)
    ffn_hybrid_spike_gate_init: float = 0.5  # sigmoide inversa => gate ~0.62
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
    self_knowledge_hidden: int = 0
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

    # ---- RAG (v4.2) ----
    rag_enabled: bool = False
    rag_kb_path: str = ""
    rag_top_k: int = 4
    rag_max_context_chars: int = 2000
    rag_encoder: str = "hidden"

    # ---- LoRA (v4.2) ----
    lora_enabled: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: list = field(
        default_factory=lambda: ["qkv", "proj", "fc1", "fc2"]
    )
    lora_path: str = ""

    # ---- Reasoning bias (v4.3) ----
    reasoning_bias_enabled: bool = False
    reasoning_bias_strength: float = 0.1
    reasoning_bias_top_k: int = 16
    reasoning_bias_remine_every: int = 8
    reasoning_bias_facts_window: int = 8
    reasoning_bias_min_confidence: float = 0.5
    reasoning_mapper_path: str = "checkpoints/symbol_mapper.pt"
    reasoning_graph_path: str = "checkpoints/graph.pt"

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

    # ---- corpus ----
    code_corpus: bool = False
    split_by: str = "file"

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
            (self.snn_mode in ("static", "multi_step", "temporal", "gelu"),
             f"invalid snn_mode: {self.snn_mode}"),
            (self.snn_reset_mode in ("soft", "hard"),
             f"invalid snn_reset_mode: {self.snn_reset_mode}"),
            (self.snn_num_steps >= 1, "snn_num_steps must be >= 1"),
            (0.0 < self.snn_decay < 1.0, "snn_decay must be in (0, 1)"),
            (self.neural_memory_max_size > 0, "neural_memory_max_size must be > 0"),
            (self.neural_memory_k >= 1, "neural_memory_k must be >= 1"),
            (self.neural_memory_k <= self.neural_memory_max_size,
             "neural_memory_k must be <= neural_memory_max_size"),
            (self.neural_memory_hebbian_lr >= 0, "neural_memory_hebbian_lr must be >= 0"),
            (0.0 <= self.neural_memory_hebbian_decay <= 1.0,
             "neural_memory_hebbian_decay must be in [0, 1]"),
            (self.neural_memory_norm_clamp > 0, "neural_memory_norm_clamp must be > 0"),
            (self.neural_memory_append_batch_size > 0,
             "neural_memory_append_batch_size must be > 0"),
            (self.abstraction_n_codes >= 2, "abstraction_n_codes must be >= 2"),
            (0.0 <= self.abstraction_ema_decay < 1.0,
             "abstraction_ema_decay must be in [0, 1)"),
            (self.abstraction_loss_weight >= 0, "abstraction_loss_weight must be >= 0"),
            (self.abstraction_residual_scale >= 0, "abstraction_residual_scale must be >= 0"),
            (self.self_knowledge_loss_weight >= 0,
             "self_knowledge_loss_weight must be >= 0"),
            (self.self_knowledge_hidden >= 0, "self_knowledge_hidden must be >= 0"),
            (self.latent_reasoner_heads >= 1, "latent_reasoner_heads must be >= 1"),
            (self.n_embd % self.latent_reasoner_heads == 0,
             "n_embd must be divisible by latent_reasoner_heads"),
            (self.latent_reasoner_layers >= 1, "latent_reasoner_layers must be >= 1"),
            (self.analyzer_interval_s > 0, "analyzer_interval_s must be > 0"),
            (self.rt_bpe_vocab_size >= 2, "rt_bpe_vocab_size must be >= 2"),
            (self.rt_n_concepts >= 2, "rt_n_concepts must be >= 2"),
            (self.rt_concept_dim > 0, "rt_concept_dim must be > 0"),
            (self.rt_aux_loss_weight >= 0, "rt_aux_loss_weight must be >= 0"),
            (0.0 <= self.rt_ema_decay < 1.0, "rt_ema_decay must be in [0, 1)"),
            (self.rt_commitment_beta >= 0, "rt_commitment_beta must be >= 0"),
            (self.rt_min_count_for_update >= 0, "rt_min_count_for_update must be >= 0"),
            (self.rt_concept_top_k_tokens >= 1, "rt_concept_top_k_tokens must be >= 1"),
            (self.split_by in ("file", "repo"), f"invalid split_by: {self.split_by}"),
            (self.rag_top_k >= 1, "rag_top_k must be >= 1"),
            (self.rag_max_context_chars >= 0, "rag_max_context_chars must be >= 0"),
            (
                self.rag_encoder == "hidden"
                or self.rag_encoder.startswith("sentence_transformer:"),
                f"invalid rag_encoder: {self.rag_encoder!r}",
            ),
            (self.lora_r > 0, "lora_r must be > 0"),
            (self.lora_alpha > 0, "lora_alpha must be > 0"),
            (0.0 <= self.lora_dropout < 1.0, "lora_dropout must be in [0, 1)"),
            (
                not self.lora_enabled or len(self.lora_targets) > 0,
                "lora_targets must be non-empty when lora_enabled=True",
            ),
            # ---- v4.3 ----
            (self.core_type in ("attention", "temporal"),
             f"invalid core_type: {self.core_type}"),
            (self.temporal_expand >= 1, "temporal_expand must be >= 1"),
            (self.temporal_lif_threshold > 0,
             "temporal_lif_threshold must be > 0"),
            (0.0 < self.temporal_lif_decay < 1.0,
             "temporal_lif_decay must be in (0,1)"),
            (self.temporal_lif_alpha > 0, "temporal_lif_alpha must be > 0"),
            (self.temporal_chunk_size >= 1, "temporal_chunk_size must be >= 1"),
            (0.0 < self.temporal_decay_base < 1.0,
             "temporal_decay_base must be in (0,1)"),
            (0.0 < self.temporal_decay_target < 1.0,
             "temporal_decay_target must be in (0,1)"),
            (self.ffn_mode in ("dense", "spiking", "hybrid"),
             f"invalid ffn_mode: {self.ffn_mode}"),
            (0.0 <= self.ffn_hybrid_spike_gate_init <= 1.0,
             "ffn_hybrid_spike_gate_init must be in [0,1]"),
            (self.reasoning_bias_strength >= 0,
             "reasoning_bias_strength must be >= 0"),
            (self.reasoning_bias_top_k >= 1, "reasoning_bias_top_k must be >= 1"),
            (self.reasoning_bias_remine_every >= 1,
             "reasoning_bias_remine_every must be >= 1"),
            (self.reasoning_bias_facts_window >= 1,
             "reasoning_bias_facts_window must be >= 1"),
            (0.0 <= self.reasoning_bias_min_confidence <= 1.0,
             "reasoning_bias_min_confidence must be in [0,1]"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

        if self.use_rt and self.vocab_size != self.rt_bpe_vocab_size:
            warnings.warn(
                f"use_rt=True but vocab_size={self.vocab_size} != "
                f"rt_bpe_vocab_size={self.rt_bpe_vocab_size}. "
                f"Override vocab_size to rt_bpe_vocab_size if using RT.",
                RuntimeWarning, stacklevel=3,
            )

        if self.snn_mode == "temporal" and self.use_spiking:
            warnings.warn(
                "snn_mode='temporal' uses BPTT over full sequence. Prefer 'multi_step'.",
                RuntimeWarning, stacklevel=3,
            )
        if self.compile and self.neural_memory_enabled:
            warnings.warn(
                "torch.compile + neural_memory may cause graph breaks.",
                RuntimeWarning, stacklevel=3,
            )
        if self.compile and self.use_rt:
            warnings.warn(
                "torch.compile + use_rt may cause graph breaks (VQ quantization).",
                RuntimeWarning, stacklevel=3,
            )
        if self.compile and self.lora_enabled:
            warnings.warn(
                "torch.compile + lora: apply_lora MUST be called before compile.",
                RuntimeWarning, stacklevel=3,
            )
        if self.temporal_mode_uses_memory():
            warnings.warn(
                "snn_mode='temporal' + neural_memory_enabled: slow.",
                RuntimeWarning, stacklevel=3,
            )
        if self.rag_enabled and not self.rag_kb_path:
            warnings.warn(
                "rag_enabled=True but rag_kb_path is empty. RAG will be no-op.",
                RuntimeWarning, stacklevel=3,
            )
        if self.core_type == "temporal" and self.snn_mode == "temporal":
            warnings.warn(
                "core_type='temporal' + snn_mode='temporal': doble recurrencia. "
                "Considera snn_mode='static' o 'multi_step' para el FFN.",
                RuntimeWarning, stacklevel=3,
            )
        if self.reasoning_bias_enabled and not os.path.exists(
            self.reasoning_mapper_path
        ):
            warnings.warn(
                f"reasoning_bias_enabled=True but mapper not found at "
                f"{self.reasoning_mapper_path!r}. Bias will be no-op.",
                RuntimeWarning, stacklevel=3,
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
    "tiny": SIZE_TINY, "small": SIZE_SMALL,
    "medium": SIZE_MEDIUM, "large": SIZE_LARGE,
}
