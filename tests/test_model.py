import torch

from snn_lm_v3 import GPTConfig, SNNLM


def _cfg(**kw):
    base = dict(n_layer=2, n_head=2, n_embd=32, seq_len=16, vocab_size=100)
    base.update(kw)
    return GPTConfig(**base)


def test_forward_shapes():
    model = SNNLM(_cfg())
    x = torch.randint(0, 100, (2, 16))
    logits, loss = model(x)
    assert logits.shape == (2, 16, 100)
    assert loss is None


def test_forward_loss():
    model = SNNLM(_cfg())
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 100)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_generate_shape():
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 5))
    out = model.generate(idx, max_new_tokens=10, temperature=1.0)
    assert out.shape == (1, 15)


def test_generate_with_and_without_cache():
    model = SNNLM(_cfg())
    model.eval()
    torch.manual_seed(0)
    idx = torch.randint(0, 100, (1, 5))
    out_a = model.generate(idx.clone(), 10, temperature=1.0, use_cache=False)
    torch.manual_seed(0)
    out_b = model.generate(idx.clone(), 10, temperature=1.0, use_cache=True)
    assert out_a.shape == out_b.shape


def test_top_p_extreme_does_not_crash():
    model = SNNLM(_cfg())
    model.eval()
    idx = torch.randint(0, 100, (1, 5))
    # top_p muy bajo: debe conservar al menos 1 token
    out = model.generate(idx, 5, temperature=1.0, top_p=1e-6)
    assert out.shape == (1, 10)


def test_spiking_toggle():
    model = SNNLM(_cfg(use_spiking=False))
    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))
    _, loss = model(x, y)
    assert loss.item() > 0
