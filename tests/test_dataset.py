"""Tests del dataset y dataloader."""
import numpy as np
import pytest

from snn_lm_v3.dataset import TextDataset, make_dataloader


def test_dataset_len_and_content():
    tokens = np.arange(100, dtype=np.uint32)
    ds = TextDataset(tokens, seq_len=10)
    assert len(ds) == 100 - 10 - 1
    x, y = ds[0]
    assert x.shape == (10,)
    assert y.shape == (10,)
    assert (x.numpy() == tokens[0:10]).all()
    assert (y.numpy() == tokens[1:11]).all()


def test_dataset_empty_when_too_short():
    ds = TextDataset(np.arange(5, dtype=np.uint32), seq_len=10)
    assert len(ds) == 0


def test_dataset_invalid_seq_len():
    with pytest.raises(ValueError):
        TextDataset(np.arange(100, dtype=np.uint32), seq_len=0)


def test_make_dataloader_no_docs_raises(tmp_path):
    pytest.importorskip("tiktoken")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No .txt files"):
        make_dataloader([str(empty)], seq_len=8, batch_size=2)


def test_make_dataloader_single_short_doc_raises(tmp_path):
    pytest.importorskip("tiktoken")
    p = tmp_path / "a.txt"
    p.write_text("hi")
    with pytest.raises(ValueError):
        make_dataloader([str(p)], seq_len=64, batch_size=2)


def test_make_dataloader_works(tmp_path, gpt2_enc):
    p = tmp_path / "a.txt"
    p.write_text("The quick brown fox jumps over the lazy dog. " * 20)
    train_dl, val_dl, enc = make_dataloader(
        [str(p)], seq_len=16, batch_size=2, val_split=0.1
    )
    assert len(train_dl) > 0
    x, y = next(iter(train_dl))
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)


def test_make_dataloader_batch_too_large(tmp_path, gpt2_enc):
    p = tmp_path / "a.txt"
    p.write_text("Hello world. " * 10)
    with pytest.raises(ValueError, match="Train dataloader is empty"):
        make_dataloader([str(p)], seq_len=64, batch_size=1000)
