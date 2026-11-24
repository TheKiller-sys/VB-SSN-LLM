import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken


def load_text(paths):
    text = ""
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".txt"):
                    with open(os.path.join(p, f), "r", encoding="utf-8") as fh:
                        text += fh.read() + "\n"
        else:
            with open(p, "r", encoding="utf-8") as fh:
                text += fh.read() + "\n"
    return text


class TextDataset(Dataset):
    """Crea (input, target) de longitud seq_len+1 sobre el texto tokenizado."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = max(0, len(tokens) - seq_len - 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        chunk = self.tokens[i:i + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def make_dataloader(paths, seq_len, batch_size, enc_name="gpt2",
                    shuffle=True, num_workers=0, val_split=0.05, seed=42):
    enc = tiktoken.get_encoding(enc_name)
    text = load_text(paths)
    tokens = np.array(enc.encode_ordinary(text), dtype=np.uint32)

    rng = np.random.default_rng(seed)
    split = int(len(tokens) * (1 - val_split))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:] if split < len(tokens) else tokens[-seq_len - 2:]

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, drop_last=False)
    return train_dl, val_dl, enc
