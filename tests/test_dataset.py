import numpy as np

from snn_lm_v3.dataset import TextDataset


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
    tokens = np.arange(5, dtype=np.uint32)
    ds = TextDataset(tokens, seq_len=10)
    assert len(ds) == 0
