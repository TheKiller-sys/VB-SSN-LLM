"""Dataset y dataloaders para LM causal. Soporta tiktoken, RT y split por repo."""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken


# Extensiones de código reconocidas (debe coincidir con bpe_trainer.CODE_EXTS)
CODE_EXTENSIONS = (
    ".txt", ".py", ".js", ".ts", ".go", ".rs", ".c", ".h", ".cpp",
    ".java", ".rb", ".md", ".kt", ".swift", ".scala", ".php", ".cs",
)


# ─────────────────────────────────────────────────────────────────────
# Helpers de carga
# ─────────────────────────────────────────────────────────────────────

def _walk_extensions(root: str, extensions: Sequence[str]) -> list[str]:
    """Devuelve todos los archivos bajo `root` con alguna de las extensiones."""
    exts = tuple(extensions)
    out: list[str] = []
    for r, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(exts):
                out.append(os.path.join(r, f))
    return out


def load_texts(paths: Iterable[str]) -> list[str]:
    """Compat: carga .txt desde paths (dirs o archivos). Devuelve contenidos."""
    docs: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            for f in _walk_extensions(p, (".txt",)):
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    docs.append(fh.read())
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                docs.append(fh.read())
    return docs


def load_texts_with_paths(
    paths: Iterable[str],
    extensions: Sequence[str] = (".txt",),
) -> list[tuple[str, str]]:
    """Carga (path, contenido) desde paths con las extensiones dadas.

    Usado por `scripts/train.py` para razonamiento y dataloaders.
    """
    exts = tuple(extensions)
    docs: list[tuple[str, str]] = []
    for p in paths:
        if os.path.isdir(p):
            for f in _walk_extensions(p, exts):
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    docs.append((f, fh.read()))
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                docs.append((p, fh.read()))
    return docs


def _split_docs(
    docs: list[tuple[str, str]],
    val_split: float,
    seed: int,
    split_by: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Divide docs en (train, val) según `split_by`.

    split_by:
      - "file": cada archivo va a train o val (puede haber leak entre archivos
                del mismo repo).
      - "repo": todos los archivos de un mismo repo van al mismo split
                (evita val leak en código).
    """
    if not docs:
        return [], []

    if split_by == "file":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(docs))
        n_val = max(1, int(len(docs) * val_split))
        val_docs = [docs[i] for i in order[:n_val]]
        train_docs = [docs[i] for i in order[n_val:]]
        return train_docs, val_docs

    if split_by == "repo":
        # Agrupar por repo (primer componente de la ruta relativa al root)
        groups: dict[str, list[tuple[str, str]]] = {}
        for path, content in docs:
            # Heurística: repo = directorio de nivel superior bajo el root común
            parts = os.path.normpath(path).split(os.sep)
            # Tomar los últimos 2 componentes si es posible, si no el archivo
            repo = os.sep.join(parts[:-1]) if len(parts) > 1 else path
            groups.setdefault(repo, []).append((path, content))

        keys = sorted(groups.keys())
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(keys))
        n_val = max(1, int(len(keys) * val_split))
        val_repos = {keys[i] for i in order[:n_val]}
        train_repos = {keys[i] for i in order[n_val:]}

        train_docs: list[tuple[str, str]] = []
        val_docs: list[tuple[str, str]] = []
        for repo, items in groups.items():
            if repo in val_repos:
                val_docs.extend(items)
            else:
                train_docs.extend(items)

        # Fallback: si un split quedó vacío, mover un archivo del otro
        if not val_docs and train_docs:
            val_docs.append(train_docs.pop())
        if not train_docs and val_docs:
            train_docs.append(val_docs.pop())

        return train_docs, val_docs

    raise ValueError(f"invalid split_by: {split_by!r}")


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
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


def _pack_or_fallback(
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    min_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Garantiza que ambos splits tengan al menos `min_tokens`."""
    if len(val_tokens) < min_tokens:
        needed = min_tokens - len(val_tokens)
        if len(train_tokens) > needed + min_tokens:
            val_tokens = np.concatenate([train_tokens[-needed:], val_tokens])
            train_tokens = train_tokens[:-needed]
        else:
            raise ValueError(
                f"Not enough tokens for val (train={len(train_tokens)}, "
                f"val={len(val_tokens)})"
            )
    if len(train_tokens) < min_tokens:
        raise ValueError(
            f"Not enough train tokens: {len(train_tokens)} < {min_tokens}"
        )
    return train_tokens, val_tokens


# ─────────────────────────────────────────────────────────────────────
# make_dataloader (tiktoken)
# ─────────────────────────────────────────────────────────────────────

def make_dataloader(
    paths,
    seq_len: int,
    batch_size: int,
    enc_name: str = "cl100k_base",
    shuffle: bool = True,
    num_workers: int = 0,
    val_split: float = 0.05,
    seed: int = 42,
    pin_memory: bool = True,
    split_by: str = "file",
    extensions: Sequence[str] = (".txt",),
):
    """Dataloader estándar con tiktoken, con split por file/repo."""
    enc = tiktoken.get_encoding(enc_name)
    docs = load_texts_with_paths(paths, extensions=extensions)
    if not docs:
        raise ValueError(
            f"No files found in {paths!r} with extensions {tuple(extensions)}"
        )

    min_tokens = seq_len + 2

    if len(docs) < 2:
        path, content = docs[0]
        tokens = np.array(enc.encode_ordinary(content), dtype=np.uint32)
        if len(tokens) < min_tokens:
            raise ValueError(f"Single doc too short ({len(tokens)}); need {min_tokens}")
        split = int(len(tokens) * (1 - val_split))
        split = max(split, min_tokens)
        split = min(split, len(tokens) - min_tokens)
        train_tokens, val_tokens = tokens[:split], tokens[split:]
    else:
        train_docs, val_docs = _split_docs(docs, val_split, seed, split_by)
        train_tokens = _encode(enc, [c for _, c in train_docs])
        val_tokens = _encode(enc, [c for _, c in val_docs])

    train_tokens, val_tokens = _pack_or_fallback(train_tokens, val_tokens, min_tokens)

    train_ds = TextDataset(train_tokens, seq_len)
    val_ds = TextDataset(val_tokens, seq_len)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True, pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin_memory,
    )
    if len(train_dl) == 0:
        raise ValueError("Train dataloader empty (batch_size too large?)")
    return train_dl, val_dl, enc


# ─────────────────────────────────────────────────────────────────────
# make_rt_dataloader (ResonantVocab)
# ─────────────────────────────────────────────────────────────────────

def make_rt_dataloader(
    paths,
    seq_len: int,
    batch_size: int,
    rv,
    shuffle: bool = True,
    num_workers: int = 0,
    val_split: float = 0.05,
    seed: int = 42,
    pin_memory: bool = True,
    split_by: str = "file",
    extensions: Sequence[str] = (".txt",),
):
    """Dataloader usando ResonantVocab.encode (BPE propio), con split por file/repo."""
    docs = load_texts_with_paths(paths, extensions=extensions)
    if not docs:
        raise ValueError(
            f"No files found in {paths!r} with extensions {tuple(extensions)}"
        )

    min_tokens = seq_len + 2

    if len(docs) < 2:
        _, content = docs[0]
        ids = np.array(rv.encode(content), dtype=np.int64)
        if len(ids) < min_tokens:
            raise ValueError(f"Single doc too short ({len(ids)}); need {min_tokens}")
        split = int(len(ids) * (1 - val_split))
        split = max(split, min_tokens)
        split = min(split, len(ids) - min_tokens)
        train_ids, val_ids = ids[:split], ids[split:]
    else:
        train_docs, val_docs = _split_docs(docs, val_split, seed, split_by)

        def _enc(ds: list[tuple[str, str]]) -> np.ndarray:
            all_ids: list[int] = []
            for _, content in ds:
                all_ids.extend(rv.encode(content))
            return np.array(all_ids, dtype=np.int64)

        train_ids = _enc(train_docs)
        val_ids = _enc(val_docs)

    train_ids, val_ids = _pack_or_fallback(train_ids, val_ids, min_tokens)

    train_ds = TextDataset(train_ids, seq_len)
    val_ds = TextDataset(val_ids, seq_len)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True, pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=pin_memory,
    )
    if len(train_dl) == 0:
        raise ValueError("Train dataloader empty (batch_size too large?)")
    return train_dl, val_dl
