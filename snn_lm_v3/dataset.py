"""Dataset y dataloaders para LM causal. Soporta tiktoken y RT, con split
por archivo o por repositorio (para evitar val leak)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken


# Extensiones por defecto para corpus de código
CODE_EXTENSIONS = (
    ".txt", ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".kt", ".swift",
    ".rb", ".php", ".cs", ".scala", ".md",
)


# ─────────────────────────────────────────────────────────────
# Carga de textos
# ─────────────────────────────────────────────────────────────

def load_texts_with_paths(
    paths: Iterable[str],
    extensions: Sequence[str] = (".txt",),
) -> list[tuple[str, str]]:
    """Devuelve (path_relativo, contenido) para cada archivo válido.

    - Si `paths[i]` es directorio: rglob recursivo, ruta relativa al directorio.
    - Si es archivo: (nombre_base, contenido).
    - Solo extensiones en `extensions`.
    """
    out: list[tuple[str, str]] = []
    for p in paths:
        p_obj = Path(p)
        if p_obj.is_dir():
            root = p_obj
            for f in sorted(root.rglob("*")):
                if not f.is_file():
                    continue
                if not f.name.endswith(tuple(extensions)):
                    continue
                try:
                    rel = f.relative_to(root).as_posix()
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        out.append((rel, fh.read()))
                except OSError:
                    continue
        elif p_obj.is_file():
            try:
                with open(p_obj, "r", encoding="utf-8", errors="ignore") as fh:
                    out.append((p_obj.name, fh.read()))
            except OSError:
                continue
    return out


def load_texts(paths: Iterable[str]) -> list[str]:
    """Compat: lista de contenidos (sin paths). Usa .txt por defecto."""
    return [c for _, c in load_texts_with_paths(paths, extensions=(".txt",))]


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Helpers de split
# ─────────────────────────────────────────────────────────────

def _split_docs_by_file(
    docs: list[tuple[str, str]], val_split: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split clásico: barajar archivos y separar."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))
    n_val = max(1, int(len(docs) * val_split))
    if n_val >= len(docs):
        n_val = len(docs) - 1
    val_docs = [docs[i][1] for i in order[:n_val]]
    train_docs = [docs[i][1] for i in order[n_val:]]
    return train_docs, val_docs


def _extract_repo(rel_path: str) -> str:
    """Extrae el repo del path relativo: 'repo/sub/file.py' -> 'repo'."""
    parts = rel_path.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "_root"


def _split_docs_by_repo(
    docs: list[tuple[str, str]], val_split: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split por repo: agrupa por directorio raíz y separa enteros.

    Evita que archivos del mismo proyecto aparezcan en train y val a la vez.
    Fallback a split por archivo si hay < 2 repos.
    """
    repos: dict[str, list[str]] = {}
    for path, content in docs:
        repo = _extract_repo(path)
        repos.setdefault(repo, []).append(content)

    if len(repos) < 2:
        return _split_docs_by_file(docs, val_split, seed)

    rng = np.random.default_rng(seed)
    repo_names = sorted(repos.keys())
    order = rng.permutation(len(repo_names))
    n_val = max(1, int(len(repo_names) * val_split))
    if n_val >= len(repo_names):
        n_val = len(repo_names) - 1

    val_docs: list[str] = []
    train_docs: list[str] = []
    for idx in order[:n_val]:
        val_docs.extend(repos[repo_names[idx]])
    for idx in order[n_val:]:
        train_docs.extend(repos[repo_names[idx]])
    return train_docs, val_docs


def _encode_tiktoken(enc, docs: list[str]) -> np.ndarray:
    if not docs:
        return np.array([], dtype=np.uint32)
    return np.concatenate(
        [np.array(enc.encode_ordinary(d), dtype=np.uint32) for d in docs]
    )


def _finalize_tokens(
    train_tokens: np.ndarray, val_tokens: np.ndarray, seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Asegura que ambos splits tienen suficientes tokens."""
    min_tokens = seq_len + 2
    if len(val_tokens) < min_tokens:
        needed = min_tokens - len(val_tokens)
        if len(train_tokens) > needed + min_tokens:
            val_tokens = np.concatenate([train_tokens[-needed:], val_tokens])
            train_tokens = train_tokens[:-needed]
        else:
            raise ValueError(
                f"Not enough tokens for val "
                f"(train={len(train_tokens)}, val={len(val_tokens)}, need={min_tokens})"
            )
    if len(train_tokens) < min_tokens:
        raise ValueError(f"Not enough train tokens: {len(train_tokens)} < {min_tokens}")
    return train_tokens, val_tokens


def _build_loaders(
    train_tokens: np.ndarray, val_tokens: np.ndarray,
    seq_len: int, batch_size: int, shuffle: bool,
    num_workers: int, pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
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
    return train_dl, val_dl


# ─────────────────────────────────────────────────────────────
# Dataloaders públicos
# ─────────────────────────────────────────────────────────────

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
    """Dataloader estándar con tiktoken.

    split_by: "file" (baraja archivos) o "repo" (baraja repos completos).
    extensions: extensiones a incluir (por defecto .txt).
    """
    if split_by not in ("file", "repo"):
        raise ValueError(f"split_by must be 'file' or 'repo', got {split_by!r}")

    enc = tiktoken.get_encoding(enc_name)
    docs = load_texts_with_paths(paths, extensions=extensions)
    if not docs:
        raise ValueError(f"No files found in {paths!r} (ext={extensions})")

    if split_by == "repo":
        train_docs, val_docs = _split_docs_by_repo(docs, val_split, seed)
    else:
        train_docs, val_docs = _split_docs_by_file(docs, val_split, seed)

    train_tokens = _encode_tiktoken(enc, train_docs)
    val_tokens = _encode_tiktoken(enc, val_docs)
    train_tokens, val_tokens = _finalize_tokens(train_tokens, val_tokens, seq_len)

    train_dl, val_dl = _build_loaders(
        train_tokens, val_tokens, seq_len, batch_size,
        shuffle, num_workers, pin_memory,
    )
    return train_dl, val_dl, enc


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
    """Dataloader usando ResonantVocab.encode (BPE propio)."""
    if split_by not in ("file", "repo"):
        raise ValueError(f"split_by must be 'file' or 'repo', got {split_by!r}")

    docs = load_texts_with_paths(paths, extensions=extensions)
    if not docs:
        raise ValueError(f"No files found in {paths!r} (ext={extensions})")

    if split_by == "repo":
        train_docs, val_docs = _split_docs_by_repo(docs, val_split, seed)
    else:
        train_docs, val_docs = _split_docs_by_file(docs, val_split, seed)

    def _encode_rv(doc_list: list[str]) -> np.ndarray:
        all_ids: list[int] = []
        for d in doc_list:
            all_ids.extend(rv.encode(d))
        return np.array(all_ids, dtype=np.int64)

    train_ids = _encode_rv(train_docs)
    val_ids = _encode_rv(val_docs)
    train_ids, val_ids = _finalize_tokens(train_ids, val_ids, seq_len)

    train_dl, val_dl = _build_loaders(
        train_ids, val_ids, seq_len, batch_size,
        shuffle, num_workers, pin_memory,
    )
    return train_dl, val_dl
