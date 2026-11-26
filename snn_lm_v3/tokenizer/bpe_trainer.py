"""Entrena un BPE sobre un corpus de texto/código. Reversible, byte-level.

FIX v4.2.1:
  - Importa `CODE_EXTENSIONS` desde `dataset` (fuente única de verdad).
  - Ignora `.git`, `node_modules`, `__pycache__`, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator


# Carpetas ignoradas en `rglob("*")`
_IGNORE_DIR_PARTS = frozenset({
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
    ".ipynb_checkpoints", ".cache",
})


def _code_extensions() -> tuple[str, ...]:
    """Importa `CODE_EXTENSIONS` del paquete padre, evitando ciclos."""
    from ..dataset import CODE_EXTENSIONS
    return tuple(CODE_EXTENSIONS)


def _is_ignored_path(p: Path) -> bool:
    return any(part in _IGNORE_DIR_PARTS for part in p.parts)


def iter_corpus(paths: list[str], chunk_bytes: int = 10_000_000) -> Iterator[str]:
    """Itera chunks de texto desde archivos o directorios."""
    exts = _code_extensions()
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files = [
                f for f in path.rglob("*")
                if f.is_file()
                and f.suffix.lower() in exts
                and not _is_ignored_path(f)
            ]
        else:
            files = [path] if path.is_file() else []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    while True:
                        chunk = fh.read(chunk_bytes)
                        if not chunk:
                            break
                        yield chunk
            except OSError:
                continue


def train_bpe(
    corpus_paths: list[str],
    vocab_size: int = 40_000,
    out_path: str = "checkpoints/bpe_code.json",
    min_frequency: int = 2,
    special_tokens: list[str] | None = None,
) -> str:
    """Entrena BPE byte-level. Devuelve la ruta al archivo .json."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    special_tokens = special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter_corpus(corpus_paths), trainer=trainer)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_path)
    return out_path
