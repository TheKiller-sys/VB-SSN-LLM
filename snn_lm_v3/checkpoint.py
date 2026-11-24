import os

import torch


def save_checkpoint(model, optimizer, config, step, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "config": config.__dict__,
        "step": step,
    }, path)


def load_checkpoint(path, model_class, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    from .config import GPTConfig
    config = GPTConfig(**{k: v for k, v in ckpt["config"].items()
                          if k in GPTConfig.__dataclass_fields__})
    model = model_class(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, config, ckpt.get("step", 0)
