"""Tokenizer Resonante: BPE entrenado + codebook VQ acoplado."""
from .resonant_vocab import ResonantVocab, RVConfig
from .rt_model import RTModelWrapper
from .bpe_trainer import train_bpe, iter_corpus

__all__ = [
    "ResonantVocab", "RVConfig", "RTModelWrapper",
    "train_bpe", "iter_corpus",
]
