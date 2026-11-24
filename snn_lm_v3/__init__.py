"""SNN-LM v3.1 — Transformer causal con FFN de spikes (LIF real) y capa de razonamiento."""
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
from .dataset import TextDataset, load_texts, make_dataloader
from .train import train, evaluate
from .generate import generate
from .checkpoint import save_checkpoint, load_checkpoint

from .reasoning import (
    SignalProbe, InternalSignals,
    CorpusObserver, Observation,
    ReasoningGraph, Rule,
    ConsistencyVerifier, Verdict, Problem,
    save_graph, load_graph,
)

__version__ = "3.1.0"

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
    # data
    "TextDataset", "load_texts", "make_dataloader",
    # training
    "train", "evaluate", "generate", "save_checkpoint", "load_checkpoint",
    # reasoning
    "SignalProbe", "InternalSignals",
    "CorpusObserver", "Observation",
    "ReasoningGraph", "Rule",
    "ConsistencyVerifier", "Verdict", "Problem",
    "save_graph", "load_graph",
]
