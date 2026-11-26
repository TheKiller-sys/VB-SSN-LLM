"""LoRA (Low-Rank Adaptation) para SNN-LM v4.3.

v4.3:
  - `apply_lora` con targets que cubren attention (`qkv`, `proj`), temporal
    (`in_proj`, `out_proj`), dense FFN (`fc1`, `fc2`) y hybrid (`fc1_spike`).
  - `_detect_default_targets(model)` inspecciona el modelo y devuelve los
    targets relevantes para su core_type/ffn_mode.
  - `_unwrap(model)` en todas las funciones públicas (soporta `torch.compile`).
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _unwrap(model):
    seen: set[int] = set()
    while hasattr(model, "_orig_mod") and id(model) not in seen:
        seen.add(id(model))
        model = model._orig_mod
    return model


class LoRALinear(nn.Module):
    """Linear con adaptador LoRA. El base queda congelado."""

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"r must be > 0, got {r}")
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.empty(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_A @ self.lora_B) * self.scaling
        return self.base.weight + delta.T

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────
# Targets
# ─────────────────────────────────────────────────────────────────────

# Defaults v4.3: cubren attention (qkv, proj), temporal (in_proj, out_proj),
# dense FFN (fc1, fc2) y hybrid (fc1_spike).
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "qkv", "proj",
    "in_proj", "out_proj",
    "fc1", "fc2", "fc1_spike",
)


def _detect_default_targets(model) -> tuple[str, ...]:
    """Devuelve los targets presentes en el modelo (no falla si alguno no existe)."""
    model = _unwrap(model)
    found: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
            short = name.rsplit(".", 1)[-1]
            if short in DEFAULT_LORA_TARGETS:
                found.add(short)
    return tuple(sorted(found))


# ─────────────────────────────────────────────────────────────────────
# Helpers de reemplazo
# ─────────────────────────────────────────────────────────────────────

def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    if "." in dotted_name:
        parent_name, _, attr = dotted_name.rpartition(".")
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        attr = dotted_name
    setattr(parent, attr, new_module)


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Sequence[str] | None = None,
    skip_prefixes: Sequence[str] = ("rv.", "tok_emb."),
) -> int:
    """Envuelve cada `nn.Linear` cuyo último componente esté en `targets`.

    Si `targets=None`, se autodetectan con `_detect_default_targets`.
    """
    model = _unwrap(model)

    if targets is None:
        targets = _detect_default_targets(model)
        if not targets:
            raise ValueError(
                "apply_lora found 0 Linear modules matching DEFAULT_LORA_TARGETS. "
                "Model has no known target layers."
            )
    if isinstance(targets, str):
        targets = (targets,)
    targets_set = set(targets)
    skip = tuple(skip_prefixes)

    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if isinstance(module, LoRALinear):
            continue
        if any(name.startswith(p) for p in skip):
            continue
        short = name.rsplit(".", 1)[-1]
        if short in targets_set:
            to_replace.append((name, module))

    for name, module in to_replace:
        wrapper = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, wrapper)

    if not to_replace:
        raise ValueError(
            f"apply_lora found 0 modules matching targets={targets}. "
            f"Model may not have the expected Linear layers."
        )

    return len(to_replace)


def freeze_base_for_lora(model: nn.Module) -> int:
    model = _unwrap(model)
    for p in model.parameters():
        p.requires_grad = False
    n_trainable = 0
    for name, p in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            p.requires_grad = True
            n_trainable += 1
    return n_trainable


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    model = _unwrap(model)
    for name, p in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            yield p


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def lora_state_dict(model: nn.Module) -> dict:
    model = _unwrap(model)
    out: dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[name] = {
                "A": module.lora_A.detach().cpu().clone(),
                "B": module.lora_B.detach().cpu().clone(),
                "r": module.r,
                "alpha": module.alpha,
            }
    return out


def load_lora_state_dict(model: nn.Module, state: dict) -> int:
    model = _unwrap(model)
    model_lora = {
        n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)
    }
    if not model_lora:
        raise ValueError(
            "load_lora_state_dict: model has no LoRALinear modules. "
            "Call apply_lora first."
        )

    missing = set(model_lora.keys()) - set(state.keys())
    extra = set(state.keys()) - set(model_lora.keys())
    if missing:
        raise ValueError(f"LoRA state missing modules: {sorted(missing)[:5]}")
    if extra:
        raise ValueError(f"LoRA state has extra modules: {sorted(extra)[:5]}")

    for name, module in model_lora.items():
        entry = state[name]
        A = entry["A"]
        B = entry["B"]
        if A.shape != module.lora_A.shape:
            raise ValueError(
                f"{name}: A shape mismatch {tuple(A.shape)} vs "
                f"{tuple(module.lora_A.shape)}. "
                f"r mismatch (loaded r={A.shape[1]}, model r={module.r})."
            )
        if B.shape != module.lora_B.shape:
            raise ValueError(
                f"{name}: B shape mismatch {tuple(B.shape)} vs "
                f"{tuple(module.lora_B.shape)}"
            )
        module.lora_A.data.copy_(A.to(module.lora_A.device))
        module.lora_B.data.copy_(B.to(module.lora_B.device))

    return len(model_lora)


def save_lora(model: nn.Module, path: str, extra: dict | None = None) -> None:
    model = _unwrap(model)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = lora_state_dict(model)
    if not state:
        raise ValueError("save_lora: model has no LoRALinear modules.")
    payload = {
        "version": 2,  # bumped: v4.3 targets
        "lora_state": state,
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_lora(model: nn.Module, path: str) -> dict:
    model = _unwrap(model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version", 0)
    if version not in (1, 2):
        raise ValueError(f"LoRA format version {version} not supported (1, 2)")
    load_lora_state_dict(model, payload["lora_state"])
    return payload.get("extra", {})


def merge_lora(model: nn.Module) -> int:
    model = _unwrap(model)
    to_merge: list[tuple[str, LoRALinear]] = [
        (n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)
    ]
    for name, module in to_merge:
        with torch.no_grad():
            module.base.weight.data.copy_(module.merged_weight())
        _set_submodule(model, name, module.base)
    return len(to_merge)


def is_lora_active(model: nn.Module) -> bool:
    model = _unwrap(model)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            return True
    return False
