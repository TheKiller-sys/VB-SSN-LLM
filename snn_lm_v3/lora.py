"""LoRA (Low-Rank Adaptation) para SNN-LM.

FIX v4.2.1:
  - Todas las funciones públicas hacen `_unwrap(model)` al entrar, para
    soportar modelos compilados con `torch.compile` (`OptimizedModule`).
  - `load_lora` avisa si el `r` del adapter no coincide con el del modelo.
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

        # A: (in, r), B: (r, out). W_eff = W + (A @ B) * scaling
        self.lora_A = nn.Parameter(torch.empty(in_f, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_f))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B = 0 → forward inicial == base

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    def merged_weight(self) -> torch.Tensor:
        """W + (A @ B) * scaling. No muta el módulo."""
        delta = (self.lora_A @ self.lora_B) * self.scaling
        return self.base.weight + delta.T

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers de reemplazo
# ─────────────────────────────────────────────────────────────────────

def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Reemplaza root.<dotted_name> por new_module."""
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
    targets: Sequence[str] = ("qkv", "proj", "fc1", "fc2"),
    skip_prefixes: Sequence[str] = ("rv.", "tok_emb."),
) -> int:
    """Envuelve cada `nn.Linear` cuyo último componente del nombre esté en `targets`.

    FIX v4.2.1: hace `_unwrap(model)` al entrar, para soportar `OptimizedModule`.
    El modelo se modifica in-place, por lo que la referencia original sigue siendo válida.
    """
    model = _unwrap(model)
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
            f"Check that the model has Linear layers named qkv/proj/fc1/fc2."
        )

    return len(to_replace)


def freeze_base_for_lora(model: nn.Module) -> int:
    """Congela todo y reactiva solo `lora_A` y `lora_B`."""
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
    """Itera solo los parámetros LoRA (A, B)."""
    model = _unwrap(model)
    for name, p in model.named_parameters():
        if ".lora_A" in name or ".lora_B" in name:
            yield p


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for p in lora_parameters(model))


def lora_state_dict(model: nn.Module) -> dict:
    """Devuelve {nombre_módulo: {A, B, r, alpha}} para todos los LoRALinear.

    FIX v4.2.1: hace `_unwrap(model)` para que los nombres no lleven `_orig_mod.`.
    """
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
    """Carga A, B desde `lora_state_dict`. Devuelve número de módulos cargados.

    FIX v4.2.1: hace `_unwrap(model)` al entrar.
    """
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
    """Guarda solo A, B + metadata. No guarda pesos base.

    FIX v4.2.1: hace `_unwrap(model)` para que los nombres no lleven `_orig_mod.`.
    """
    model = _unwrap(model)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = lora_state_dict(model)
    if not state:
        raise ValueError("save_lora: model has no LoRALinear modules.")
    payload = {
        "version": 1,
        "lora_state": state,
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_lora(model: nn.Module, path: str) -> dict:
    """Carga A, B en un modelo que YA tiene LoRA aplicado.

    Returns:
        Metadata `extra` guardada con `save_lora`.
    """
    model = _unwrap(model)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version", 0)
    if version != 1:
        raise ValueError(f"LoRA format version {version} != 1")
    load_lora_state_dict(model, payload["lora_state"])
    return payload.get("extra", {})


def merge_lora(model: nn.Module) -> int:
    """Funde A @ B en los pesos base. Destructivo: reemplaza LoRALinear por Linear.

    FIX v4.2.1: hace `_unwrap(model)` al entrar.
    """
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
    """True si el modelo tiene al menos un LoRALinear.

    FIX v4.2.1: hace `_unwrap(model)` para mirar el módulo real.
    """
    model = _unwrap(model)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            return True
    return False
