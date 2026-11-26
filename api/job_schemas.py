"""
Esquemas declarativos por tipo de job.

Cada campo define:
  name        -> clave en config (snake_case)
  flag        -> bandera CLI (kebab-case)
  type        -> int | float | str | bool | select | list | paths
  group       -> agrupación visual en el frontend
  default     -> valor por defecto
  required    -> si es obligatorio
  options     -> para select
  help        -> tooltip
  min/max     -> para number
"""
from __future__ import annotations


SIZES = ["tiny", "small", "medium", "large"]
CORES = ["attention", "temporal"]
FFN_MODES = ["dense", "spiking", "hybrid"]
SNN_MODES = ["static", "multi_step", "temporal", "gelu"]
SPLIT_BY = ["file", "repo"]
DEVICES = ["auto", "cpu", "cuda", "mps"]


TRAIN_SCHEMA = {
    "type": "train",
    "title": "Train from scratch / resume",
    "description": "Entrena SNN-LM v4.3 con core configurable.",
    "positional": [
        {
            "name": "data", "type": "paths", "required": True,
            "group": "Data", "help": "Directorios o archivos de corpus.",
        },
    ],
    "fields": [
        # Data
        {"name": "extensions", "flag": "--extensions", "type": "list",
         "default": [".txt"], "group": "Data"},
        {"name": "split_by", "flag": "--split-by", "type": "select",
         "options": SPLIT_BY, "default": "file", "group": "Data"},
        {"name": "code_corpus", "flag": "--code-corpus", "type": "bool",
         "default": False, "group": "Data"},

        # Model
        {"name": "size", "flag": "--size", "type": "select",
         "options": SIZES, "default": "tiny", "group": "Model"},
        {"name": "core", "flag": "--core", "type": "select",
         "options": CORES, "default": "temporal", "group": "Model"},
        {"name": "ffn_mode", "flag": "--ffn-mode", "type": "select",
         "options": FFN_MODES, "default": "hybrid", "group": "Model"},
        {"name": "temporal_expand", "flag": "--temporal-expand",
         "type": "int", "default": 2, "group": "Model"},
        {"name": "temporal_chunk_size", "flag": "--temporal-chunk-size",
         "type": "int", "default": 64, "group": "Model"},
        {"name": "temporal_decay_base", "flag": "--temporal-decay-base",
         "type": "float", "default": 0.9, "group": "Model"},
        {"name": "temporal_decay_target", "flag": "--temporal-decay-target",
         "type": "float", "default": 0.99, "group": "Model"},
        {"name": "no_spike_decay", "flag": "--no-spike-decay",
         "type": "bool", "default": False, "group": "Model"},
        {"name": "hybrid_gate_init", "flag": "--hybrid-gate-init",
         "type": "float", "default": 0.5, "group": "Model"},

        # SNN
        {"name": "snn_mode", "flag": "--snn-mode", "type": "select",
         "options": SNN_MODES, "default": "multi_step", "group": "SNN"},
        {"name": "snn_steps", "flag": "--snn-steps", "type": "int",
         "default": 4, "group": "SNN"},
        {"name": "snn_track_stats", "flag": "--snn-track-stats",
         "type": "bool", "default": False, "group": "SNN"},
        {"name": "analyzer", "flag": "--analyzer", "type": "bool",
         "default": False, "group": "SNN"},
        {"name": "analyzer_interval", "flag": "--analyzer-interval",
         "type": "float", "default": 60.0, "group": "SNN"},

        # Training
        {"name": "max_steps", "flag": "--max-steps", "type": "int",
         "default": 1000, "group": "Training"},
        {"name": "batch_size", "flag": "--batch-size", "type": "int",
         "default": 8, "group": "Training"},
        {"name": "grad_accum", "flag": "--grad-accum", "type": "int",
         "default": 1, "group": "Training"},
        {"name": "seq_len", "flag": "--seq-len", "type": "int",
         "default": 128, "group": "Training"},
        {"name": "lr", "flag": "--lr", "type": "float",
         "default": 3e-4, "group": "Training"},
        {"name": "min_lr", "flag": "--min-lr", "type": "float",
         "default": 3e-5, "group": "Training"},
        {"name": "warmup_steps", "flag": "--warmup-steps", "type": "int",
         "default": 100, "group": "Training"},
        {"name": "log_every", "flag": "--log-every", "type": "int",
         "default": 50, "group": "Training"},
        {"name": "eval_every", "flag": "--eval-every", "type": "int",
         "default": 500, "group": "Training"},
        {"name": "eval_batches", "flag": "--eval-batches", "type": "int",
         "default": 50, "group": "Training"},
        {"name": "ckpt_every", "flag": "--ckpt-every", "type": "int",
         "default": 500, "group": "Training"},
        {"name": "num_workers", "flag": "--num-workers", "type": "int",
         "default": 0, "group": "Training"},
        {"name": "seed", "flag": "--seed", "type": "int",
         "default": 42, "group": "Training"},
        {"name": "compile", "flag": "--compile", "type": "bool",
         "default": False, "group": "Training"},
        {"name": "no_amp", "flag": "--no-amp", "type": "bool",
         "default": False, "group": "Training"},

        # RT
        {"name": "use_rt", "flag": "--use-rt", "type": "bool",
         "default": False, "group": "RT"},
        {"name": "bpe_path", "flag": "--bpe-path", "type": "str",
         "default": "checkpoints/bpe_code.json", "group": "RT"},
        {"name": "n_concepts", "flag": "--n-concepts", "type": "int",
         "default": 12092, "group": "RT"},
        {"name": "concept_dim", "flag": "--concept-dim", "type": "int",
         "default": 132, "group": "RT"},
        {"name": "aux_loss_weight", "flag": "--aux-loss-weight",
         "type": "float", "default": 0.1, "group": "RT"},

        # LoRA
        {"name": "lora", "flag": "--lora", "type": "bool",
         "default": False, "group": "LoRA"},
        {"name": "lora_r", "flag": "--lora-r", "type": "int",
         "default": 8, "group": "LoRA"},
        {"name": "lora_alpha", "flag": "--lora-alpha", "type": "float",
         "default": 16.0, "group": "LoRA"},
        {"name": "lora_dropout", "flag": "--lora-dropout", "type": "float",
         "default": 0.0, "group": "LoRA"},
        {"name": "lora_targets", "flag": "--lora-targets", "type": "list",
         "default": None, "group": "LoRA"},
        {"name": "lora_out", "flag": "--lora-out", "type": "str",
         "default": None, "group": "LoRA"},

        # RAG
        {"name": "rag_kb", "flag": "--rag-kb", "type": "str",
         "default": None, "group": "RAG"},
        {"name": "rag_top_k", "flag": "--rag-top-k", "type": "int",
         "default": 4, "group": "RAG"},

        # Reasoning
        {"name": "no_reasoning", "flag": "--no-reasoning", "type": "bool",
         "default": False, "group": "Reasoning"},
        {"name": "reasoning_chunk_size", "flag": "--reasoning-chunk-size",
         "type": "int", "default": 500, "group": "Reasoning"},
        {"name": "reasoning_bias", "flag": "--reasoning-bias",
         "type": "bool", "default": False, "group": "Reasoning"},
        {"name": "reasoning_bias_strength", "flag": "--reasoning-bias-strength",
         "type": "float", "default": None, "group": "Reasoning"},
        {"name": "reasoning_bias_top_k", "flag": "--reasoning-bias-top-k",
         "type": "int", "default": None, "group": "Reasoning"},
        {"name": "reasoning_bias_remine_every",
         "flag": "--reasoning-bias-remine-every",
         "type": "int", "default": None, "group": "Reasoning"},
        {"name": "reasoning_bias_facts_window",
         "flag": "--reasoning-bias-facts-window",
         "type": "int", "default": None, "group": "Reasoning"},
        {"name": "reasoning_bias_min_confidence",
         "flag": "--reasoning-bias-min-confidence",
         "type": "float", "default": None, "group": "Reasoning"},
        {"name": "reasoning_mapper_path", "flag": "--reasoning-mapper-path",
         "type": "str", "default": None, "group": "Reasoning"},
        {"name": "reasoning_graph_path", "flag": "--reasoning-graph-path",
         "type": "str", "default": None, "group": "Reasoning"},

        # Output
        {"name": "out", "flag": "--out", "type": "str",
         "default": "checkpoints/run", "group": "Output"},
        {"name": "device", "flag": "--device", "type": "select",
         "options": DEVICES, "default": "auto", "group": "Output"},
        {"name": "config", "flag": "--config", "type": "str",
         "default": None, "group": "Output"},
        {"name": "resume", "flag": "--resume", "type": "str",
         "default": None, "group": "Output"},
    ],
}


LORA_SCHEMA = {
    "type": "lora",
    "title": "LoRA fine-tuning",
    "description": "Adapter LoRA sobre un checkpoint congelado.",
    "positional": [
        {"name": "data", "type": "paths", "required": True,
         "group": "Data"},
    ],
    "fields": [
        {"name": "base", "flag": "--base", "type": "str",
         "required": True, "group": "Input"},
        {"name": "out", "flag": "--out", "type": "str",
         "required": True, "group": "Output"},
        {"name": "r", "flag": "--r", "type": "int", "default": 8,
         "group": "LoRA"},
        {"name": "alpha", "flag": "--alpha", "type": "float",
         "default": 16.0, "group": "LoRA"},
        {"name": "dropout", "flag": "--dropout", "type": "float",
         "default": 0.0, "group": "LoRA"},
        {"name": "targets", "flag": "--targets", "type": "list",
         "default": None, "group": "LoRA"},
        {"name": "lr", "flag": "--lr", "type": "float",
         "default": 1e-3, "group": "Training"},
        {"name": "max_steps", "flag": "--max-steps", "type": "int",
         "default": 500, "group": "Training"},
        {"name": "batch_size", "flag": "--batch-size", "type": "int",
         "default": None, "group": "Training"},
        {"name": "grad_accum", "flag": "--grad-accum", "type": "int",
         "default": None, "group": "Training"},
        {"name": "seq_len", "flag": "--seq-len", "type": "int",
         "default": None, "group": "Training"},
        {"name": "warmup_steps", "flag": "--warmup-steps", "type": "int",
         "default": 100, "group": "Training"},
        {"name": "eval_every", "flag": "--eval-every", "type": "int",
         "default": 200, "group": "Training"},
        {"name": "eval_batches", "flag": "--eval-batches", "type": "int",
         "default": 20, "group": "Training"},
        {"name": "log_every", "flag": "--log-every", "type": "int",
         "default": 20, "group": "Training"},
        {"name": "ckpt_every", "flag": "--ckpt-every", "type": "int",
         "default": 500, "group": "Training"},
        {"name": "seed", "flag": "--seed", "type": "int",
         "default": 42, "group": "Training"},
        {"name": "device", "flag": "--device", "type": "select",
         "options": DEVICES, "default": "auto", "group": "Output"},
        {"name": "extensions", "flag": "--extensions", "type": "list",
         "default": [".txt"], "group": "Data"},
        {"name": "split_by", "flag": "--split-by", "type": "select",
         "options": SPLIT_BY, "default": "file", "group": "Data"},
        {"name": "code_corpus", "flag": "--code-corpus", "type": "bool",
         "default": False, "group": "Data"},
        {"name": "no_amp", "flag": "--no-amp", "type": "bool",
         "default": False, "group": "Training"},
        {"name": "num_workers", "flag": "--num-workers", "type": "int",
         "default": 0, "group": "Training"},
    ],
}


RAG_INDEX_SCHEMA = {
    "type": "rag_index",
    "title": "RAG index build",
    "description": "Construye o extiende un KnowledgeBase.",
    "positional": [],
    "fields": [
        {"name": "corpus", "flag": "--corpus", "type": "paths",
         "required": True, "group": "Input"},
        {"name": "out", "flag": "--out", "type": "str",
         "default": None, "group": "Output",
         "help": "Ruta de KB nueva (si no se usa --kb)."},
        {"name": "kb", "flag": "--kb", "type": "str",
         "default": None, "group": "Input",
         "help": "KB existente a extender."},
        {"name": "encoder", "flag": "--encoder", "type": "select",
         "options": ["hidden", "sentence_transformer:all-MiniLM-L6-v2"],
         "default": "hidden", "group": "Encoder"},
        {"name": "ckpt", "flag": "--ckpt", "type": "str",
         "default": None, "group": "Encoder"},
        {"name": "extensions", "flag": "--extensions", "type": "list",
         "default": [".txt"], "group": "Input"},
        {"name": "chunk_size", "flag": "--chunk-size", "type": "int",
         "default": 1000, "group": "Chunking"},
        {"name": "chunk_overlap", "flag": "--chunk-overlap", "type": "int",
         "default": 200, "group": "Chunking"},
        {"name": "max_docs", "flag": "--max-docs", "type": "int",
         "default": 100000, "group": "Limits"},
        {"name": "device", "flag": "--device", "type": "select",
         "options": DEVICES, "default": "auto", "group": "Encoder"},
    ],
}


REASONING_SCHEMA = {
    "type": "reasoning",
    "title": "Reasoning graph mining",
    "description": "Mina reglas del corpus y guarda el grafo.",
    "positional": [],
    "fields": [
        {"name": "corpus", "flag": "--corpus", "type": "paths",
         "required": True, "group": "Input"},
        {"name": "out", "flag": "--save", "type": "str",
         "required": True, "group": "Output",
         "help": "Ruta del grafo a guardar."},
        {"name": "chunk_size", "flag": "--chunk-size", "type": "int",
         "default": 500, "group": "Mining"},
    ],
}


BPE_SCHEMA = {
    "type": "bpe_train",
    "title": "BPE training",
    "description": "Entrena un tokenizer BPE byte-level.",
    "positional": [],
    "fields": [
        {"name": "corpus", "flag": "--corpus", "type": "paths",
         "required": True, "group": "Input"},
        {"name": "out", "flag": "--out", "type": "str",
         "required": True, "group": "Output"},
        {"name": "vocab_size", "flag": "--vocab-size", "type": "int",
         "default": 40000, "group": "BPE"},
        {"name": "min_frequency", "flag": "--min-frequency", "type": "int",
         "default": 2, "group": "BPE"},
    ],
}


JOB_SCHEMAS = {
    "train": TRAIN_SCHEMA,
    "lora": LORA_SCHEMA,
    "rag_index": RAG_INDEX_SCHEMA,
    "reasoning": REASONING_SCHEMA,
    "bpe_train": BPE_SCHEMA,
}


# ─────────────────────────────────────────────────────────────────────
# CLI arg building
# ─────────────────────────────────────────────────────────────────────

SCRIPT_MAP = {
    "train": "scripts/train.py",
    "lora": "scripts/train_lora.py",
    "rag_index": "scripts/add_knowledge.py",
    "reasoning": "scripts/reason.py",
    "bpe_train": "scripts/train_bpe.py",
}


def validate_config(job_type: str, cfg: dict) -> dict:
    """Valida contra el schema. Devuelve cfg normalizado. Raise ValueError."""
    schema = JOB_SCHEMAS.get(job_type)
    if schema is None:
        raise ValueError(f"Unknown job_type: {job_type!r}")

    out: dict = {}

    for pos in schema["positional"]:
        name = pos["name"]
        val = cfg.get(name)
        if val is None and pos.get("required"):
            raise ValueError(f"Missing required field: {name}")
        if val is not None:
            out[name] = val

    for field in schema["fields"]:
        name = field["name"]
        val = cfg.get(name, field.get("default"))
        if val is None:
            if field.get("required"):
                raise ValueError(f"Missing required field: {name}")
            continue
        ftype = field["type"]
        try:
            if ftype == "int" and val is not None:
                val = int(val)
            elif ftype == "float" and val is not None:
                val = float(val)
            elif ftype == "bool":
                val = bool(val)
            elif ftype == "list":
                if isinstance(val, str):
                    val = [v.strip() for v in val.split(",") if v.strip()]
                else:
                    val = list(val)
            elif ftype == "paths":
                if isinstance(val, str):
                    val = [v.strip() for v in val.split(",") if v.strip()]
                else:
                    val = list(val)
            elif ftype == "select":
                if val not in field["options"]:
                    raise ValueError(
                        f"{name}={val!r} not in {field['options']}"
                    )
            elif ftype == "str":
                val = str(val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid field {name}={val!r}: {e}")
        out[name] = val

    return out


def build_cli_cmd(job_type: str, cfg: dict) -> list[str]:
    """Convierte config validada en lista de argv para el script."""
    import sys

    schema = JOB_SCHEMAS[job_type]
    script = SCRIPT_MAP[job_type]
    cmd = [sys.executable, script]

    for pos in schema["positional"]:
        name = pos["name"]
        if name in cfg:
            val = cfg[name]
            if isinstance(val, (list, tuple)):
                cmd.extend(str(v) for v in val)
            else:
                cmd.append(str(val))

    for field in schema["fields"]:
        name = field["name"]
        if name not in cfg:
            continue
        val = cfg[name]
        flag = field["flag"]
        ftype = field["type"]

        if ftype == "bool":
            if val:
                cmd.append(flag)
            continue

        if ftype in ("list", "paths"):
            if val:
                cmd.append(flag)
                cmd.extend(str(v) for v in val)
            continue

        cmd.extend([flag, str(val)])

    return cmd
