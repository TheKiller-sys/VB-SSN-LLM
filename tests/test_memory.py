"""Tests de memoria neural, abstracción y autoconocimiento."""
import pytest
import torch

from snn_lm_v3 import (
    GPTConfig, SNNLM,
    NeuralMemory, AbstractionHead, SelfKnowledgeHead,
)


def test_neural_memory_shapes():
    mem = NeuralMemory(dim=32, max_size=100, k=4)
    x = torch.randn(2, 8, 32)
    out = mem(x, update=False)
    assert out.shape == (2, 8, 32)


def test_neural_memory_append_grows_size():
    mem = NeuralMemory(dim=32, max_size=100, k=4)
    x = torch.randn(2, 8, 32)
    mem(x, update=True)
    assert int(mem.size.item()) == 16


def test_neural_memory_ring_buffer():
    mem = NeuralMemory(dim=16, max_size=10, k=2)
    for _ in range(5):
        mem(torch.randn(1, 4, 16), update=True)
    assert int(mem.size.item()) == 10


def test_neural_memory_knn_lookup_uses_memory():
    mem = NeuralMemory(dim=16, max_size=100, k=4)
    torch.manual_seed(0)
    # Insertar vectores conocidos
    mem(torch.randn(1, 8, 16), update=True)
    # Ahora un vector similar debe recuperar algo
    x = torch.randn(1, 1, 16)
    out = mem(x, update=False)
    assert out.shape == (1, 1, 16)
    assert not torch.allclose(out, x)


def test_neural_memory_hebbian_does_not_change_transformer():
    """Hebbian update NO debe tocar parámetros de otros módulos."""
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=8, vocab_size=100,
        neural_memory_enabled=True,
        abstraction_enabled=False,
        self_knowledge_enabled=False,
        snn_track_stats=False,
    )
    model = SNNLM(cfg)
    model.eval()

    block_params_before = {
        n: p.clone() for n, p in model.blocks.named_parameters()
    }

    x = torch.randint(0, 100, (1, 8))
    model(x, update_memory=True)

    for n, p in model.blocks.named_parameters():
        assert torch.equal(p, block_params_before[n]), \
            f"Transformer param {n} cambió tras Hebbian"


def test_neural_memory_reset():
    mem = NeuralMemory(dim=8, max_size=100, k=2)
    mem(torch.randn(1, 4, 8), update=True)
    assert int(mem.size.item()) > 0
    mem.reset()
    assert int(mem.size.item()) == 0


def test_abstraction_head_shapes():
    head = AbstractionHead(dim=32, n_codes=16)
    x = torch.randn(2, 8, 32)
    quantized, idx = head(x, update=False)
    assert quantized.shape == (2, 8, 32)
    assert idx.shape == (2, 8)
    assert idx.min() >= 0 and idx.max() < 16


def test_abstraction_head_ema_update():
    head = AbstractionHead(dim=16, n_codes=8, ema_decay=0.9)
    x = torch.randn(2, 8, 16)
    head(x, update=True)
    assert head.ema_counts.sum().item() > 0


def test_abstraction_head_codebook_does_not_get_gradients():
    """El codebook es buffer, no parámetro."""
    head = AbstractionHead(dim=16, n_codes=8)
    params = list(head.parameters())
    codebook_in_params = any(
        p is head.codebook for p in params
    )
    assert not codebook_in_params


def test_self_knowledge_head_shapes():
    head = SelfKnowledgeHead(dim=32)
    hidden = torch.randn(2, 8, 32)
    concept = torch.randn(2, 8, 32)
    confidence = head(hidden, concept)
    assert confidence.shape == (2, 8)
    assert (confidence >= 0).all() and (confidence <= 1).all()


def test_self_knowledge_loss_computes():
    head = SelfKnowledgeHead(dim=16)
    hidden = torch.randn(2, 8, 16)
    concept = torch.randn(2, 8, 16)
    conf = head(hidden, concept)
    logits = torch.randn(2, 8, 100)
    targets = torch.randint(0, 100, (2, 8))
    loss = SelfKnowledgeHead.compute_loss(conf, logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_full_model_with_memory_forward():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=8, vocab_size=100,
        neural_memory_enabled=True,
        abstraction_enabled=True,
        self_knowledge_enabled=True,
        snn_mode="multi_step",
        snn_num_steps=2,
        dropout=0.0,
    )
    model = SNNLM(cfg)
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    logits, loss = model(x, y)
    assert logits.shape == (2, 8, 100)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_full_model_hebbian_during_eval_updates_memory():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=8, vocab_size=100,
        neural_memory_enabled=True,
        abstraction_enabled=False,
        self_knowledge_enabled=False,
    )
    model = SNNLM(cfg)
    model.eval()
    size_before = int(model.neural_memory.size.item())
    x = torch.randint(0, 100, (1, 8))
    model(x, update_memory=True)
    size_after = int(model.neural_memory.size.item())
    assert size_after > size_before


def test_full_model_no_memory_update_during_train():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=8, vocab_size=100,
        neural_memory_enabled=True,
        abstraction_enabled=False,
        self_knowledge_enabled=False,
    )
    model = SNNLM(cfg)
    model.train()
    x = torch.randint(0, 100, (1, 8))
    y = torch.randint(0, 100, (1, 8))
    model(x, y, update_memory=False)
    assert int(model.neural_memory.size.item()) == 0


def test_memory_state_checkpoint_roundtrip():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=8, vocab_size=100,
        neural_memory_enabled=True,
        abstraction_enabled=True,
        self_knowledge_enabled=False,
    )
    model = SNNLM(cfg)
    model.eval()
    model(torch.randint(0, 100, (1, 8)), update_memory=True)

    state = model.memory_state_for_checkpoint()
    assert "neural_memory" in state
    assert "abstraction" in state

    model2 = SNNLM(cfg)
    model2.load_memory_state_from_checkpoint(state, "cpu")
    assert int(model2.neural_memory.size.item()) == int(
        model.neural_memory.size.item()
    )
