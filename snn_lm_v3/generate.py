import torch

import tiktoken


@torch.no_grad()
def generate(model, prompt="", max_new_tokens=200, temperature=1.0,
             top_k=None, top_p=None, seed=None, enc_name="gpt2",
             device=None):
    enc = tiktoken.get_encoding(enc_name)
    device = device or next(model.parameters()).device
    model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    if prompt:
        ids = enc.encode_ordinary(prompt)
    else:
        ids = [enc.eot_token]

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature, top_k, top_p)
    return enc.decode(out[0].tolist())
