"""Entrena un BPE sobre un corpus de texto/código. Reversible, byte-level."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

CODE_EXTS = {
    ".txt", ".py", ".js", ".ts", ".go", ".rs", ".c", ".h", ".cpp",
    ".java", ".rb", ".md", ".kt", ".swift", ".scala", ".php", ".cs",
}


def iter_corpus(paths: list[str], chunk_bytes: int = 10_000_000) -> Iterator[str]:
    """Itera chunks de texto desde archivos o directorios."""
    for p in paths:
        path = Path(p)
        files = list(path.rglob("*")) if path.is_dir() else [path]
        for f in files:
            if not f.is_file() or f.suffix.lower() not in CODE_EXTS:
                continue
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
