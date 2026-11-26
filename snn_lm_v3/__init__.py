"""SNN-LM v4.1 — Transformer causal con FFN de spikes, memoria neural,
abstracción geométrica, autoconocimiento, razonamiento simbólico y
Resonant Tokenizer (BPE + VQ)."""
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

__version__ = "4.1.0"

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
    "MultiHeadAttention", "TransformerBlock", "SNNLM",
    # memory
    "NeuralMemory", "AbstractionHead", "SelfKnowledgeHead", "LatentReasoner",
    # data
    "TextDataset", "load_texts", "load_texts_with_paths",
    "make_dataloader", "make_rt_dataloader", "CODE_EXTENSIONS",
    # training
    "train", "evaluate", "generate", "save_checkpoint", "load_checkpoint",
    # analyzer
    "ContinuousAnalyzer",
    # RT (v4.1)
    "ResonantVocab", "RVConfig", "RTModelWrapper",
    "train_bpe", "iter_corpus",
    # reasoning
    "SignalProbe", "InternalSignals",
    "CorpusObserver", "Observation",
    "ReasoningGraph", "Rule",
    "ConsistencyVerifier", "Verdict", "Problem",
    "save_graph", "load_graph",
]
