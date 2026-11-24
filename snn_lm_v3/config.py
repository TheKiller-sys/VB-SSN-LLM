from dataclasses import dataclass, asdict
import json
import os


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    seq_len: int = 512
    dropout: float = 0.1
    spiking_threshold: float = 1.0
    spiking_alpha: float = 2.0

    # training
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 100_000
    grad_clip: float = 1.0
    use_amp: bool = True

    # system
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 500
    out_dir: str = "checkpoints"
    device: str = "cuda"
    compile: bool = False

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


SIZE_TINY = dict(n_layer=6, n_head=8, n_embd=256)
SIZE_SMALL = dict(n_layer=8, n_head=8, n_embd=512)
SIZE_MEDIUM = dict(n_layer=12, n_head=12, n_embd=768)
