"""Fixtures compartidos."""
import random

import numpy as np
import pytest
import torch

from snn_lm_v3 import GPTConfig, SNNLM


@pytest.fixture(scope="session")
def gpt2_enc():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return tiktoken.get_encoding("gpt2")
    except Exception:
        pytest.skip("gpt2 encoding not available")


@pytest.fixture(scope="session")
def cl100k_enc():
    tiktoken = pytest.importorskip("tiktoken")
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        pytest.skip("cl100k_base not available")


@pytest.fixture(scope="session")
def tiny_bpe_path(tmp_path_factory):
    """Entrena un BPE tiny sobre un corpus de código."""
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    corpus_dir = tmp_path_factory.mktemp("bpe_corpus")
    (corpus_dir / "a.txt").write_text(
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
        "def merge(a, b):\n    return sorted(a + b)\n" * 30
    )

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=256, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(
        [open(corpus_dir / "a.txt").read()], trainer=trainer,
    )
    path = tmp_path_factory.mktemp("bpe") / "bpe.json"
    tok.save(str(path))
    return str(path)


@pytest.fixture
def tiny_rv(tiny_bpe_path):
    """ResonantVocab tiny para tests."""
    pytest.importorskip("tokenizers")
    from snn_lm_v3.tokenizer import ResonantVocab, RVConfig

    return ResonantVocab.from_bpe(
        tiny_bpe_path,
        RVConfig(n_concepts=16, concept_dim=8, embedding_dim=16, bpe_vocab_size=256),
    )


@pytest.fixture
def set_seed():
    def _set(s: int = 0) -> None:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        random.seed(s)
        np.random.seed(s)
    return _set


@pytest.fixture
def tiny_cfg():
    return GPTConfig(
        n_layer=2, n_head=2, n_embd=32, seq_len=16, vocab_size=100,
        dropout=0.0, use_spiking=True, snn_mode="multi_step",
        snn_num_steps=2, snn_track_stats=True,
    )


@pytest.fixture
def tiny_model(tiny_cfg, set_seed):
    set_seed(0)
    return SNNLM(tiny_cfg)
