from snn_lm_v3 import GPTConfig, SNNLM, generate


def test_generate_wrapper_returns_str():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=16, vocab_size=50257
    )
    model = SNNLM(cfg)
    out = generate(model, prompt="hello", max_new_tokens=3, temperature=1.0)
    assert isinstance(out, str)


def test_generate_empty_prompt_uses_eot():
    cfg = GPTConfig(
        n_layer=1, n_head=1, n_embd=16, seq_len=16, vocab_size=50257
    )
    model = SNNLM(cfg)
    out = generate(model, prompt="", max_new_tokens=3, temperature=1.0)
    assert isinstance(out, str)
