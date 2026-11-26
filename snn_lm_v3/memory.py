"""Memoria neural, abstracción geométrica y autoconocimiento para SNN-LM v4.2.1.

Thread safety:
  - `NeuralMemory` y `AbstractionHead` usan `threading.RLock` interno.
  - El `ContinuousAnalyzer` (hilo daemon) puede llamar `prune()` y
    `reinit_dead_codes()` de forma segura mientras el modelo entrene/gene.

Semántica de `update` (FIX v4.2):
  - `update=None` (default) -> `self.training`. La memoria se ACTUALIZA
    durante el entrenamiento y NO se toca en eval/inference.
  - `update=True` -> fuerza actualización (continual learning / generation).
  - `update=False` -> solo retrieval, sin escritura.

FIX v4.2.1 (Hebbian numerics):
  - `flat` y `delta` se clampean ANTES de `W.add_` para evitar overflow
    con `float16` bajo AMP.
"""
from __future__ import annotations

import math
import threading
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# NEURAL MEMORY (kNN + fast weights)
# ============================================================
class NeuralMemory(nn.Module):
    """Memoria episódica con kNN lookup y fast-weights Hebbian.

    - `forward(x)`: kNN lookup sobre `keys` y mezcla `values` en `x`.
    - Si `update=True`: apila `x` en ring-buffer y aplica Heb sobre `W_mem`.
    - `update=None` -> `self.training` (FIX v4.2).
    - `prune()`: elimina NaN/inf de keys/values (thread-safe).
    """

    def __init__(
        self,
        dim: int,
        max_size: int = 10_000,
        k: int = 8,
        hebbian_lr: float = 1e-3,
        hebbian_decay: float = 0.999,
        norm_clamp: float = 100.0,
        use_fast_weights: bool = True,
        use_knn: bool = True,
        append_batch_size: int = 4096,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if max_size <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        self.dim = dim
        self.max_size = max_size
        self.k = k
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.norm_clamp = norm_clamp
        self.use_fast_weights = use_fast_weights
        self.use_knn = use_knn
        self.append_batch_size = append_batch_size

        if use_fast_weights:
            self.W_mem = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.W_mem.weight)
        else:
            self.W_mem = None

        self.register_buffer(
            "keys", torch.zeros(max_size, dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "values", torch.zeros(max_size, dim, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ptr", torch.zeros(1, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "size", torch.zeros(1, dtype=torch.long), persistent=False,
        )

        self._lock = threading.RLock()

    # ------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        update: Optional[bool] = None,
    ) -> torch.Tensor:
        # ── FIX v4.2: actualizar en TRAIN, no en EVAL ──
        if update is None:
            update = self.training
        if not self.use_fast_weights and not self.use_knn:
            return x

        orig_dtype = x.dtype
        orig_shape = x.shape
        x32 = x.float()

        out = x32

        if self.use_fast_weights and self.W_mem is not None:
            out = out + self.W_mem(x32)

        if self.use_knn and self.size.item() > 0:
            retrieved = self._knn_lookup(x32)
            out = out + retrieved

        if update:
            with torch.no_grad():
                self._append(x32.detach())
                if self.use_fast_weights and self.W_mem is not None:
                    self._hebbian_update(x32.detach())

        return out.to(orig_dtype).reshape(orig_shape)

    # ------------------------------------------------------- kNN
    def _knn_lookup(self, x: torch.Tensor) -> torch.Tensor:
        with self._lock:
            B, T, D = x.shape
            flat = x.reshape(-1, D)
            n_active = int(self.size.item())

            if n_active == 0:
                return torch.zeros_like(x)

            if n_active == self.max_size:
                active_keys = self.keys
                active_values = self.values
            else:
                active_keys = self.keys[:n_active]
                active_values = self.values[:n_active]

            k = min(self.k, n_active)
            chunk = self.append_batch_size

            results = []
            for i in range(0, flat.shape[0], chunk):
                q = flat[i:i + chunk]
                sims = q @ active_keys.T
                top_vals, top_idx = sims.topk(k, dim=-1)
                weights = torch.softmax(top_vals, dim=-1)
                neighbor_vals = active_values[top_idx]
                retrieved = (weights.unsqueeze(-1) * neighbor_vals).sum(dim=1)
                results.append(retrieved)

            retrieved = torch.cat(results, dim=0)
            return retrieved.reshape(B, T, D)

    # ------------------------------------------------------- append
    @torch.no_grad()
    def _append(self, x: torch.Tensor) -> None:
        with self._lock:
            flat = x.reshape(-1, self.dim)
            valid = torch.isfinite(flat).all(dim=-1)
            flat = flat[valid]
            n = flat.shape[0]
            if n == 0:
                return

            norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            flat = flat / norms

            if n >= self.max_size:
                self.keys.copy_(flat[-self.max_size:])
                self.values.copy_(flat[-self.max_size:])
                self.ptr.fill_(0)
                self.size.fill_(self.max_size)
                return

            ptr = int(self.ptr.item())
            size = int(self.size.item())
            end = ptr + n

            if end <= self.max_size:
                self.keys[ptr:end] = flat
                self.values[ptr:end] = flat
            else:
                first = self.max_size - ptr
                self.keys[ptr:] = flat[:first]
                self.values[ptr:] = flat[:first]
                rest = n - first
                self.keys[:rest] = flat[first:]
                self.values[:rest] = flat[first:]

            self.ptr.fill_((ptr + n) % self.max_size)
            self.size.fill_(min(self.max_size, size + n))

    # ------------------------------------------------------- hebbian
    @torch.no_grad()
    def _hebbian_update(self, x: torch.Tensor, lr: float | None = None) -> None:
        with self._lock:
            lr = self.hebbian_lr if lr is None else lr
            if lr <= 0 or self.W_mem is None:
                return

            flat = x.reshape(-1, self.dim)
            valid = torch.isfinite(flat).all(dim=-1)
            flat = flat[valid]
            n = flat.shape[0]
            if n < 2:
                return

            # ── FIX v4.2.1: clampear flat antes del outer product ──
            flat = flat.clamp(-self.norm_clamp, self.norm_clamp)

            delta = (flat.T @ flat) / n

            # ── FIX v4.2.1: clampear delta ANTES de W.add_ ──
            delta_norm = delta.norm()
            if delta_norm > self.norm_clamp:
                delta = delta * (self.norm_clamp / delta_norm)

            W = self.W_mem.weight.data
            W.mul_(self.hebbian_decay)
            W.add_(delta, alpha=lr)

            norm = W.norm()
            if norm > self.norm_clamp:
                W.mul_(self.norm_clamp / norm)

    # ------------------------------------------------------- prune
    @torch.no_grad()
    def prune(self) -> int:
        """Elimina entradas con NaN/inf en keys/values. Thread-safe."""
        with self._lock:
            n = int(self.size.item())
            if n == 0:
                return 0
            active_keys = self.keys if n == self.max_size else self.keys[:n]
            active_values = self.values if n == self.max_size else self.values[:n]

            finite_mask = torch.isfinite(active_keys).all(dim=-1)
            n_invalid = int((~finite_mask).sum().item())
            if n_invalid == 0:
                return 0

            keep_keys = active_keys[finite_mask]
            keep_values = active_values[finite_mask]
            n_keep = keep_keys.shape[0]

            self.keys[:n_keep] = keep_keys
            self.values[:n_keep] = keep_values
            if n_keep < self.max_size:
                self.keys[n_keep:].zero_()
                self.values[n_keep:].zero_()

            self.size.fill_(n_keep)
            self.ptr.fill_(0 if n_keep == self.max_size else n_keep)
            return n_invalid

    # ------------------------------------------------------- api
    def get_keys(self) -> torch.Tensor:
        with self._lock:
            n = int(self.size.item())
            if n == 0:
                return self.keys.new_zeros(0, self.dim)
            if n == self.max_size:
                return self.keys.clone()
            return self.keys[:n].clone()

    def reset(self) -> None:
        with self._lock:
            self.keys.zero_()
            self.values.zero_()
            self.ptr.zero_()
            self.size.zero_()
            if self.W_mem is not None:
                nn.init.zeros_(self.W_mem.weight)

    def state_for_checkpoint(self) -> dict:
        with self._lock:
            return {
                "keys": self.keys.cpu().clone(),
                "values": self.values.cpu().clone(),
                "ptr": self.ptr.cpu().clone(),
                "size": self.size.cpu().clone(),
            }

    def load_state_from_checkpoint(self, data: dict, device) -> None:
        with self._lock:
            if "keys" in data:
                self.keys.copy_(data["keys"].to(device))
            if "values" in data:
                self.values.copy_(data["values"].to(device))
            if "ptr" in data:
                self.ptr.copy_(data["ptr"].to(device))
            if "size" in data:
                self.size.copy_(data["size"].to(device))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, max_size={self.max_size}, k={self.k}, "
            f"size={int(self.size.item())}, "
            f"fast_weights={self.use_fast_weights}, knn={self.use_knn}"
        )


# ============================================================
# ABSTRACTION HEAD (VQ-VAE ligero con EMA)
# ============================================================
class AbstractionHead(nn.Module):
    """Proyecta embeddings a un codebook discreto con EMA.

    `update=None` -> `self.training` (FIX v4.2). El codebook se actualiza
    durante el entrenamiento y se congela en eval.
    """

    def __init__(
        self,
        dim: int,
        n_codes: int = 256,
        ema_decay: float = 0.99,
        min_count_for_update: float = 0.1,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")
        if n_codes < 2:
            raise ValueError(f"n_codes must be >= 2, got {n_codes}")

        self.dim = dim
        self.n_codes = n_codes
        self.ema_decay = ema_decay
        self.min_count_for_update = min_count_for_update

        self.proj = nn.Linear(dim, dim)

        self.register_buffer(
            "codebook", torch.randn(n_codes, dim) * 0.02, persistent=True,
        )
        self.register_buffer(
            "ema_counts", torch.zeros(n_codes), persistent=True,
        )
        self.register_buffer(
            "ema_sums", torch.zeros(n_codes, dim), persistent=True,
        )

        self._lock = threading.RLock()

    def forward(
        self,
        x: torch.Tensor,
        update: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ── FIX v4.2: actualizar en TRAIN, no en EVAL ──
        if update is None:
            update = self.training

        orig_dtype = x.dtype
        z = self.proj(x.float())
        B, T, D = z.shape

        z_flat = z.reshape(-1, D)
        dists = torch.cdist(z_flat, self.codebook)
        idx = dists.argmin(dim=-1)
        quantized = self.codebook[idx].reshape(B, T, D)

        quantized_st = z + (quantized - z).detach()
        code_idx = idx.reshape(B, T)

        if update:
            with torch.no_grad():
                self._ema_update(z_flat.detach(), idx.detach())

        return quantized_st.to(orig_dtype), code_idx

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, idx: torch.Tensor) -> None:
        with self._lock:
            n = z.shape[0]
            if n == 0:
                return
            one_hot = F.one_hot(idx, self.n_codes).float()
            counts = one_hot.sum(dim=0)
            sums = one_hot.T @ z

            decay = self.ema_decay
            self.ema_counts.mul_(decay).add_(counts, alpha=1 - decay)
            self.ema_sums.mul_(decay).add_(sums, alpha=1 - decay)

            n_eff = self.ema_counts.clamp(min=1e-6).unsqueeze(-1)
            new_codes = self.ema_sums / n_eff
            mask = (self.ema_counts > self.min_count_for_update).unsqueeze(-1)
            self.codebook.copy_(torch.where(mask, new_codes, self.codebook))

    @torch.no_grad()
    def reinit_dead_codes(self, min_count: float = 0.01) -> int:
        with self._lock:
            dead = (self.ema_counts < min_count).nonzero(as_tuple=True)[0]
            n_dead = dead.shape[0]
            if n_dead == 0:
                return 0
            live = (self.ema_counts >= min_count).nonzero(as_tuple=True)[0]
            if live.shape[0] == 0:
                self.codebook[dead] = torch.randn_like(self.codebook[dead]) * 0.02
            else:
                src = live[
                    torch.randint(0, live.shape[0], (n_dead,), device=live.device)
                ]
                noise = torch.randn_like(self.codebook[dead]) * 0.01
                self.codebook[dead] = self.codebook[src] + noise
            self.ema_counts[dead] = 0.0
            self.ema_sums[dead] = 0.0
            return int(n_dead)

    def get_code_for(self, emb: torch.Tensor) -> torch.Tensor:
        orig_shape = emb.shape[:-1]
        z = self.proj(emb.float())
        flat = z.reshape(-1, self.dim)
        dists = torch.cdist(flat, self.codebook)
        idx = dists.argmin(dim=-1)
        return idx.reshape(orig_shape)

    def reset(self) -> None:
        with self._lock:
            nn.init.normal_(self.codebook, std=0.02)
            self.ema_counts.zero_()
            self.ema_sums.zero_()


# ============================================================
# SELF-KNOWLEDGE HEAD
# ============================================================
class SelfKnowledgeHead(nn.Module):
    """Predice la confianza del modelo por token."""

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden if hidden and hidden > 0 else dim
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        concept_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([hidden, concept_emb], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

    @staticmethod
    def compute_loss(
        confidence: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            correct = (logits.argmax(dim=-1) == targets).float()
        return F.binary_cross_entropy(
            confidence.clamp(min=1e-6, max=1.0 - 1e-6),
            correct,
        )


# ============================================================
# LATENT REASONER
# ============================================================
class LatentReasoner(nn.Module):
    """Cross-attention sobre claves de memoria."""

    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by n_heads ({n_heads})"
            )
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=n_heads,
                dim_feedforward=dim * 4,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.gate = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        mem_keys: torch.Tensor | None,
    ) -> torch.Tensor:
        if mem_keys is None or mem_keys.shape[0] == 0:
            return x

        B = x.shape[0]
        mem = mem_keys.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, mem)

        gate = torch.sigmoid(self.gate(x))
        return x * gate
