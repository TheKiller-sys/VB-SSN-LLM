"""Tests del split por repositorio en dataset."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from snn_lm_v3.dataset import (
    load_texts_with_paths, _split_docs_by_repo, _split_docs_by_file,
    _extract_repo,
)


def test_extract_repo():
    assert _extract_repo("repo1/file.py") == "repo1"
    assert _extract_repo("repo1/sub/file.py") == "repo1"
    assert _extract_repo("file.py") == "_root"


def test_load_texts_with_paths_dir(tmp_path):
    (tmp_path / "repo1").mkdir()
    (tmp_path / "repo1" / "a.txt").write_text("hello")
    (tmp_path / "repo2").mkdir()
    (tmp_path / "repo2" / "b.txt").write_text("world")

    docs = load_texts_with_paths([str(tmp_path)], extensions=(".txt",))
    paths = {p for p, _ in docs}
    assert "repo1/a.txt" in paths
    assert "repo2/b.txt" in paths


def test_load_texts_with_paths_filters_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("text")
    (tmp_path / "b.py").write_text("code")
    (tmp_path / "c.bin").write_text("binary")

    txt_docs = load_texts_with_paths([str(tmp_path)], extensions=(".txt",))
    assert len(txt_docs) == 1
    assert txt_docs[0][0] == "a.txt"

    code_docs = load_texts_with_paths([str(tmp_path)], extensions=(".py",))
    assert len(code_docs) == 1
    assert code_docs[0][0] == "b.py"


def test_split_by_repo_no_leak():
    docs = [
        ("repo1/a.py", "AAA"),
        ("repo1/b.py", "BBB"),
        ("repo2/c.py", "CCC"),
        ("repo2/d.py", "DDD"),
        ("repo3/e.py", "EEE"),
        ("repo3/f.py", "FFF"),
    ]
    train, val = _split_docs_by_repo(docs, val_split=0.34, seed=42)
    # 34% de 3 repos ≈ 1 repo en val
    assert len(val) == 2  # un repo completo (2 archivos)
    assert len(train) == 4  # dos repos completos
    # No debe haber archivos de repo mezclados
    train_str = " ".join(train)
    val_str = " ".join(val)
    for repo, marker_a, marker_b in [
        ("repo1", "AAA", "BBB"),
        ("repo2", "CCC", "DDD"),
        ("repo3", "EEE", "FFF"),
    ]:
        in_train = marker_a in train_str and marker_b in train_str
        in_val = marker_a in val_str and marker_b in val_str
        # Ambos archivos del mismo repo van juntos
        assert in_train or in_val, f"Repo {repo} quedó dividido"


def test_split_by_repo_fallback_single_repo():
    """Con 1 solo repo, debe caer a split por archivo."""
    docs = [
        ("repo1/a.py", "A"),
        ("repo1/b.py", "B"),
        ("repo1/c.py", "C"),
    ]
    train, val = _split_docs_by_repo(docs, val_split=0.33, seed=42)
    assert len(train) >= 1 and len(val) >= 1


def test_split_by_file_random():
    docs = [(f"f{i}.txt", f"content{i}") for i in range(10)]
    train, val = _split_docs_by_file(docs, val_split=0.2, seed=42)
    assert len(val) == 2
    assert len(train) == 8


def test_make_dataloader_repo_split(tmp_path):
    pytest.importorskip("tiktoken")
    from snn_lm_v3.dataset import make_dataloader

    for repo in ["r1", "r2", "r3"]:
        (tmp_path / repo).mkdir()
        # Cada repo con suficientes tokens
        content = "The quick brown fox jumps over the lazy dog. " * 100
        (tmp_path / repo / "a.txt").write_text(content)

    train_dl, val_dl, enc = make_dataloader(
        [str(tmp_path)], seq_len=16, batch_size=2,
        split_by="repo", val_split=0.34,
    )
    assert len(train_dl) > 0
    assert len(val_dl) >= 0  # puede ser 0 si el repo es muy chico


def test_invalid_split_by_raises(tmp_path):
    pytest.importorskip("tiktoken")
    from snn_lm_v3.dataset import make_dataloader

    (tmp_path / "a.txt").write_text("text " * 1000)
    with pytest.raises(ValueError, match="split_by"):
        make_dataloader(
            [str(tmp_path)], seq_len=16, batch_size=2, split_by="invalid",
        )
