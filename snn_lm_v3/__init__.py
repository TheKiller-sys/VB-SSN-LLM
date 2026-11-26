"""SNN-LM v4.3 — Transformer con core configurable (attention|temporal SSM),
FFN híbrido dense+spiking, memoria neural, abstracción geométrica,
autoconocimiento, razonamiento simbólico con bias en logits, RT y LoRA."""
from .config import (
    GPTConfig, resolve_device,
    SIZE_TINY, SIZE_SMALL, SIZE_MEDIUM, SIZE_LARGE, SIZES,
)
from .snn import (
    SurrogateSpike, spike_fn, LIFNeuron,
    SpikingActivation, SpikingFFN,
)
from .rope import precompute_rope, apply_rope, rotate_half
from .attention import MultiHeadAttention
from .temporal_core import TemporalCore
from .block import TransformerBlock
from .model import SNNLM
from .memory import (
    NeuralMemory, AbstractionHead, SelfKnowledgeHead, LatentReasoner,
)
from .dataset import (
    TextDataset, load_texts, load_texts_with_paths,
    make_dataloader, make_rt_dataloader, CODE_EXTENSIONS,
)
from .train import train, evaluate
from .generate import generate
from .checkpoint import save_checkpoint, load_checkpoint
from .analyzer import ContinuousAnalyzer

# RAG (v4.2)
from .rag import (
    KnowledgeBase,
    make_model_encoder, make_sentence_transformer_encoder,
    build_rag_prompt,
)

# LoRA (v4.2)
from .lora import (
    LoRALinear, apply_lora, freeze_base_for_lora,
    lora_parameters, count_lora_params,
    lora_state_dict, load_lora_state_dict,
    save_lora, load_lora, merge_lora, is_lora_active,
)

# Resonant Tokenizer (v4.1)
from .tokenizer import (
    ResonantVocab, RVConfig, RTModelWrapper,
    train_bpe, iter_corpus,
)

from .reasoning import (
    SignalProbe, InternalSignals,
    CorpusObserver, Observation,
    ReasoningGraph, Rule,
    ConsistencyVerifier, Verdict, Problem,
    save_graph, load_graph,
)

# Reasoning bias (v4.3)
from .symbol_mapping import TokenSymbolMapper
from .reasoning_bias import ReasoningBias, ReasoningBiasState

__version__ = "4.3.0"

__all__ = [
    # config
    "GPTConfig", "resolve_device",
    "SIZE_TINY", "SIZE_SMALL", "SIZE_MEDIUM", "SIZE_LARGE", "SIZES",
    # snn
    "SurrogateSpike", "spike_fn", "LIFNeuron",
    "SpikingActivation", "SpikingFFN",
    # rope
    "precompute_rope", "apply_rope", "rotate_half",
    # transformer
    "MultiHeadAttention", "TemporalCore", "TransformerBlock", "SNNLM",
    # memory
    "NeuralMemory", "AbstractionHead", "SelfKnowledgeHead", "LatentReasoner",
    # data
    "TextDataset", "load_texts", "load_texts_with_paths",
    "make_dataloader", "make_rt_dataloader", "CODE_EXTENSIONS",
    # training
    "train", "evaluate", "generate", "save_checkpoint", "load_checkpoint",
    # analyzer
    "ContinuousAnalyzer",
    # RAG
    "KnowledgeBase", "make_model_encoder", "make_sentence_transformer_encoder",
    "build_rag_prompt",
    # LoRA
    "LoRALinear", "apply_lora", "freeze_base_for_lora",
    "lora_parameters", "count_lora_params",
    "lora_state_dict", "load_lora_state_dict",
    "save_lora", "load_lora", "merge_lora", "is_lora_active",
    # RT
    "ResonantVocab", "RVConfig", "RTModelWrapper",
    "train_bpe", "iter_corpus",
    # reasoning
    "SignalProbe", "InternalSignals",
    "CorpusObserver", "Observation",
    "ReasoningGraph", "Rule",
    "ConsistencyVerifier", "Verdict", "Problem",
    "save_graph", "load_graph",
    # reasoning bias (v4.3)
    "TokenSymbolMapper", "ReasoningBias", "ReasoningBiasState",
]
