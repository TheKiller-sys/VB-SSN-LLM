"""Tests de integración: corpus → grafo → modelo → generación → verificación."""
import pytest
import torch

from snn_lm_v3 import (
    GPTConfig, SNNLM, generate,
    CorpusObserver, ReasoningGraph, ConsistencyVerifier,
    SignalProbe,
    save_graph, load_graph,
)


CORPUS = [
    "The clever fox quickly jumped over the brown lazy dog.",
    "The clever fox quickly jumped over the brown lazy dog.",
    "The clever fox quickly jumped over the brown lazy dog.",
]


def _build_graph():
    obs = CorpusObserver()
    for text in CORPUS:
        obs.observe(text)
    graph = ReasoningGraph(cooldown_steps=2)
    graph.mine_from_corpus(obs)
    return obs, graph


def _tiny_model():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=32, vocab_size=50257,
        snn_mode="multi_step", snn_num_steps=2, snn_track_stats=True,
        dropout=0.0,
    )
    torch.manual_seed(0)
    return SNNLM(cfg)


def test_full_pipeline_corpus_to_graph():
    obs, graph = _build_graph()
    assert obs.get_state()["vocabulary_size"] > 0
    assert graph.get_state()["rules_count"] >= 1


def test_full_pipeline_generate_and_verify(gpt2_enc):
    obs, graph = _build_graph()
    model = _tiny_model()
    model.eval()

    text = generate(
        model, prompt="The clever", max_new_tokens=10,
        temperature=0.8, top_k=40, top_p=0.95, seed=1,
    )
    assert isinstance(text, str)

    verifier = ConsistencyVerifier()
    problems = verifier.verify_against_corpus(text, graph, obs)
    verdict = verifier.verdict(problems)
    # Aceptable cualquier veredicto, pero no debe crashear
    assert verdict in {
        "valid", "valid_with_warnings", "suspect", "invalid",
        "insufficient_evidence",
    }


def test_signal_probe_returns_signals(gpt2_enc):
    model = _tiny_model()
    probe = SignalProbe(model)
    signals = probe.probe("The clever fox")
    assert signals.n_tokens > 0
    assert 0.0 <= signals.mean_spike_rate <= 1.0
    assert signals.output_entropy >= 0.0


def test_signal_probe_feeds_verifier(gpt2_enc):
    model = _tiny_model()
    probe = SignalProbe(model)
    signals = probe.probe("The clever fox")
    verifier = ConsistencyVerifier()
    problems = verifier.verify("estoy en calma", signals)
    # No assert on count — solo que no crashea y la lógica se ejecuta
    assert isinstance(problems, list)


def test_persist_and_reload_graph_then_apply():
    obs, graph = _build_graph()
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "graph.pt")
        save_graph(graph, path)
        graph2 = load_graph(path)
    assert len(graph2.rules) == len(graph.rules)
    top = graph2.get_state()["top_rules"]
    if top:
        r = top[0]
        consequences = graph2.apply([r["antecedent"]])
        assert isinstance(consequences, list)


def test_end_to_end_with_cache_and_no_cache(gpt2_enc):
    model = _tiny_model()
    model.eval()
    text1 = generate(
        model, prompt="The clever", max_new_tokens=5, seed=7, use_cache=True
    )
    text2 = generate(
        model, prompt="The clever", max_new_tokens=5, seed=7, use_cache=False
    )
    # Misma semilla → misma salida
    assert text1 == text2
