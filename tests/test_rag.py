"""Tests para RAG: KnowledgeBase y build_rag_prompt."""
import numpy as np
import pytest
import torch

from snn_lm_v3.rag import KnowledgeBase, build_rag_prompt


def dummy_encoder(dim: int = 8):
    """Encoder determinista: suma de char codes → vector normalizado."""
    def encode(text: str) -> np.ndarray:
        v = np.zeros(dim, dtype=np.float32)
        for i, c in enumerate(text):
            v[i % dim] += float(ord(c))
        return v
    return encode


def test_kb_empty_search():
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    assert kb.search("anything", k=4) == []
    assert len(kb) == 0


def test_kb_add_and_search():
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    docs = ["the quick brown fox", "the lazy dog sleeps", "quantum mechanics"]
    added = kb.add(docs)
    assert added == 3
    assert len(kb) == 3

    results = kb.search("the quick brown fox", k=2)
    assert len(results) == 2
    # El primero debe ser idéntico al query
    assert results[0][0] == "the quick brown fox"
    assert results[0][1] > 0.99


def test_kb_skips_empty_and_invalid():
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    added = kb.add(["", "   ", "valid doc"])
    assert added == 1
    assert len(kb) == 1


def test_kb_requires_encoder_for_add():
    kb = KnowledgeBase(dim=8, encoder_fn=None)
    with pytest.raises(RuntimeError):
        kb.add(["some doc"])


def test_kb_save_load(tmp_path):
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    kb.add(["doc one", "doc two", "doc three"])
    path = str(tmp_path / "kb.pt")
    kb.save(path)

    kb2 = KnowledgeBase.load(path, encoder_fn=dummy_encoder())
    assert len(kb2) == len(kb)
    assert kb2.dim == kb.dim
    np.testing.assert_array_equal(kb2.keys, kb.keys)
    assert kb2.texts == kb.texts

    # Search funciona tras load
    results = kb2.search("doc one", k=1)
    assert results[0][0] == "doc one"


def test_kb_max_docs_fifo():
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder(), max_docs=3)
    kb.add(["a", "b", "c", "d", "e"])
    assert len(kb) == 3
    assert kb.texts == ["c", "d", "e"]


def test_kb_shape_validation(tmp_path):
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    kb.add(["x"])
    path = str(tmp_path / "kb.pt")
    kb.save(path)

    # Corromper: dim distinta
    payload = torch.load(path, weights_only=False)
    payload["dim"] = 16
    torch.save(payload, path)
    with pytest.raises(ValueError):
        KnowledgeBase.load(path, encoder_fn=dummy_encoder())


def test_build_rag_prompt_no_kb():
    assert build_rag_prompt("hello", None) == "hello"
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    assert build_rag_prompt("hello", kb) == "hello"


def test_build_rag_prompt_with_context():
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    kb.add(["Paris is the capital of France"])
    out = build_rag_prompt("capital of France?", kb, k=1)
    assert "Paris" in out
    assert "capital of France?" in out
    assert out.index("Paris") < out.index("capital of France?")


def test_build_rag_prompt_truncation():
    kb = KnowledgeBase(dim=8, encoder_fn=dummy_encoder())
    kb.add(["x" * 5000])
    out = build_rag_prompt("query", kb, k=1, max_context_chars=100)
    # El contexto se trunca a 100 chars; el prompt ("query") se preserva
    assert len(out) < 200
    assert "query" in out


def test_kb_encoder_dim_mismatch():
    kb = KnowledgeBase(dim=8, encoder_fn=lambda t: np.zeros(16, dtype=np.float32))
    with pytest.raises(ValueError):
        kb.add(["doc"])
