"""Dataset y dataloaders para LM causal."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken


def load_texts(paths: Iterable[str]) -> list[str]:
    """Devuelve una lista de documentos (cada archivo .txt es uno)."""
    docs: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".txt"):
                    with open(os.path.join(p, f), "r", encoding="utf-8") as fh:
                        docs.append(fh.read())
        else:
            with open(p, "r", encoding="utf-8") as fh:
                docs.append(fh.read())
    return docs


class TextDataset(Dataset):
    """Ventanas deslizantes de (input, target) de longitud seq_len."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = max(0, len(tokens) - seq_len - 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        chunk = self.tokens[i : i + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def _encode(enc, docs: list[str]) -> np.ndarray:
    if not docs:
        return np.array([], dtype=np.uint32)
    return np.concatenate(
        [np.array(enc.encode_ordinary(d), dtype=np.uint32) for d in docs]
    )


def make_dataloader(
    paths,
    seq_len: int,
    batch_size: int,
    enc_name: str = "gpt2",
    shuffle: bool = True,
    num_workers: int = 0,
    val_split: float = 0.05,
    seed: int = 42,
    pin_memory: bool = True,
):
    enc = tiktoken.get_encoding(enc_name)
    docs = load_texts(paths)
    if not docs:
        raise ValueError(f"No .txt files found in {paths!r}")

    min_tokens = seq_len + 2

    if len(docs) < 2:
        tokens = np.array(enc.encode_ordinary(docs[0]), dtype=np.uint32)
        if len(tokens) < min_tokens:
            raise ValueError(
                f"Single document too short ({len(tokens)} tokens); "
                f"need at least {min_tokens}"
            )
        split = int(len(tokens) * (1 - val_split))
        split = max(split, min_tokens)
        split = min(split, len(tokens) - min_tokens)
        train_tokens = tokens[:split]
        val_tokens = tokens[split:]
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(docs))
        n_val = max(1, int(len(docs) * val_split))
        val_docs = [docs[i] for i in order[:n_val]]
        train_docs = [docs[i] for i in order[n_val:]]
        train_tokens = _encode(enc, train_docs)
        val_tokens = _encode(enc, val_docs)

    if len(val_tokens) < min_tokens:
        needed = min_tokens - len(val_tokens)
        if len(train_tokens) > needed + min_tokens:
            val_tokens = np.concatenate([train_tokens[-needed:], val_tokens])
            train_tokens = train_tokens[:-needed]
        else:
            raise ValueError(
                f"Not enough tokens for validation split "
                f"(train={len(train_tokens)}, val={len(val_tokens)}, need>={min_tokens})"
            )

    if len(train_tokens) < min_tokens:
        raise ValueError(
            f"Not enough training tokens: {len(train_tokens)} < {min_tokens}"
        )

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )
    if len(train_dl) == 0:
        raise ValueError("Train dataloader is empty (batch_size too large?).")
    return train_dl, val_dl, enc
