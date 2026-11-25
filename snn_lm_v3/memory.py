"""Memoria neural, abstracción geométrica y autoconocimiento para SNN-LM v4.0.

Contiene cuatro componentes que se integran en `SNNLM`:

  - `NeuralMemory`: kNN + fast weights con actualización Hebbian (sin backprop).
  - `AbstractionHead`: VQ codebook con EMA (esquemas emergentes del latente).
  - `SelfKnowledgeHead`: predice confianza del modelo por token (entrenado).
  - `LatentReasoner`: cross-attention sobre `NeuralMemory` para razonamiento
    latente (opcional, desactivado por defecto).

Todas las actualizaciones "sin backprop" ocurren dentro de `torch.no_grad()`
y NO tocan el grafo de cómputo del transformer. El transformer queda congelado
durante generación/análisis; solo se actualizan buffers y fast-weights.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# NEURAL MEMORY (kNN + fast weights)
# ============================================================
class NeuralMemory(nn.Module):
    """Memoria episódica con kNN lookup y fast-weights Hebbian.

    Comportamiento:
      - `forward(x)` hace lookup kNN sobre `keys` y mezcla `values` en `x`.
      - Si `update=True`, además apila `x` en el buffer (ring buffer FIFO)
        y aplica regla de Hebb sobre `W_mem` (con decay y clamp de norma).

    El buffer es circular (`ptr`), de tamaño fijo `max_size`.
    Las claves/valores se guardan en `float32` siempre, para estabilidad.

    IMPORTANTE:
      - `W_mem` es un `nn.Linear(dim, dim, bias=False)` entrenable por backprop
        durante el entrenamiento normal. El update Hebbian (eval) es
        adicional y NO compite con el optimizador porque sólo se aplica
        cuando `model.eval()`.
      - Para evitar OOM con `max_size` grande, el lookup kNN se hace en
        chunks de `append_batch_size`.
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

        # Fast weights (entrenable)
        if use_fast_weights:
            self.W_mem = nn.Linear(dim, dim, bias=False)
            nn.init.zeros_(self.W_mem.weight)
        else:
            self.W_mem = None

        # Ring buffer (no persistente en checkpoint por tamaño, se guarda aparte)
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

    # ------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        update: Optional[bool] = None,
    ) -> torch.Tensor:
        """x: (B, T, D). Devuelve: (B, T, D) con memoria mezclada."""
        if update is None:
            update = not self.training
        if not self.use_fast_weights and not self.use_knn:
            return x

        orig_dtype = x.dtype
        orig_shape = x.shape
        x32 = x.float()

        out = x32

        # 1. fast weights (backprop-friendly)
        if self.use_fast_weights and self.W_mem is not None:
            out = out + self.W_mem(x32)

        # 2. kNN retrieval
        if self.use_knn and self.size.item() > 0:
            retrieved = self._knn_lookup(x32)
            out = out + retrieved

        # 3. update (Hebbian + buffer append)
        if update:
            with torch.no_grad():
                self._append(x32.detach())
                if self.use_fast_weights and self.W_mem is not None:
                    self._hebbian_update(x32.detach())

        return out.to(orig_dtype).reshape(orig_shape)

    # ------------------------------------------------------- kNN
    def _knn_lookup(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) fp32. Devuelve: (B, T, D) fp32."""
        B, T, D = x.shape
        flat = x.reshape(-1, D)  # (B*T, D)
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
            q = flat[i:i + chunk]  # (c, D)
            # similitud coseno (keys están normalizadas al apilar)
            sims = q @ active_keys.T  # (c, N)
            top_vals, top_idx = sims.topk(k, dim=-1)  # (c, k)
            weights = torch.softmax(top_vals, dim=-1)  # (c, k)
            neighbor_vals = active_values[top_idx]  # (c, k, D)
            retrieved = (weights.unsqueeze(-1) * neighbor_vals).sum(dim=1)
            results.append(retrieved)

        retrieved = torch.cat(results, dim=0)
        return retrieved.reshape(B, T, D)

    # ------------------------------------------------------- append
    @torch.no_grad()
    def _append(self, x: torch.Tensor) -> None:
        """x: (B, T, D) fp32. Apila al ring buffer con normalización L2."""
        flat = x.reshape(-1, self.dim)
        # Filtrar NaN/inf
        valid = torch.isfinite(flat).all(dim=-1)
        flat = flat[valid]
        n = flat.shape[0]
        if n == 0:
            return

        # Normalizar para similitud coseno
        norms = flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        flat = flat / norms

        if n >= self.max_size:
            # Sobrescribe todo con los últimos max_size
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
        """Regla de Hebb con decay y clamp de norma.

        W = (1 - decay) * W + lr * (x^T @ x) / n
        W = W * min(1, norm_clamp / ||W||)
        """
        lr = self.hebbian_lr if lr is None else lr
        if lr <= 0 or self.W_mem is None:
            return

        flat = x.reshape(-1, self.dim)
        valid = torch.isfinite(flat).all(dim=-1)
        flat = flat[valid]
        n = flat.shape[0]
        if n < 2:
            return

        delta = (flat.T @ flat) / n  # (D, D)

        W = self.W_mem.weight.data
        W.mul_(self.hebbian_decay)
        W.add_(delta, alpha=lr)

        norm = W.norm()
        if norm > self.norm_clamp:
            W.mul_(self.norm_clamp / norm)

    # ------------------------------------------------------- api
    def get_keys(self) -> torch.Tensor:
        """Devuelve las claves activas (para LatentReasoner)."""
        n = int(self.size.item())
        if n == 0:
            return self.keys.new_zeros(0, self.dim)
        if n == self.max_size:
            return self.keys
        return self.keys[:n]

    def reset(self) -> None:
        self.keys.zero_()
        self.values.zero_()
        self.ptr.zero_()
        self.size.zero_()
        if self.W_mem is not None:
            nn.init.zeros_(self.W_mem.weight)

    def state_for_checkpoint(self) -> dict:
        """Estado serializable (buffers incluidos)."""
        return {
            "keys": self.keys.cpu().clone(),
            "values": self.values.cpu().clone(),
            "ptr": self.ptr.cpu().clone(),
            "size": self.size.cpu().clone(),
        }

    def load_state_from_checkpoint(self, data: dict, device) -> None:
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
    """Proyecta embeddings a un codebook discreto (esquemas emergentes).

    El codebook NO se entrena por backprop: se actualiza con EMA durante
    eval, de modo que los códigos convergen a los centroides de los
    embeddings vistos. La proyección `proj` SÍ se entrena por backprop
    con straight-through estimator.

    Salida:
      - `quantized`: (B, T, D) con straight-through
      - `code_idx`: (B, T) índice discreto
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

        # codebook como buffer (no entrenable)
        self.register_buffer(
            "codebook", torch.randn(n_codes, dim) * 0.02, persistent=True,
        )
        self.register_buffer(
            "ema_counts", torch.zeros(n_codes), persistent=True,
        )
        self.register_buffer(
            "ema_sums", torch.zeros(n_codes, dim), persistent=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        update: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, T, D). Devuelve (quantized_st, code_idx)."""
        if update is None:
            update = not self.training

        orig_dtype = x.dtype
        z = self.proj(x.float())  # (B, T, D)
        B, T, D = z.shape

        # distancias
        z_flat = z.reshape(-1, D)
        dists = torch.cdist(z_flat, self.codebook)  # (B*T, K)
        idx = dists.argmin(dim=-1)  # (B*T,)
        quantized = self.codebook[idx].reshape(B, T, D)

        # straight-through
        quantized_st = z + (quantized - z).detach()
        code_idx = idx.reshape(B, T)

        if update:
            with torch.no_grad():
                self._ema_update(z_flat.detach(), idx.detach())

        return quantized_st.to(orig_dtype), code_idx

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, idx: torch.Tensor) -> None:
        """z: (N, D) fp32, idx: (N,) long."""
        n = z.shape[0]
        if n == 0:
            return

        # Acumular por código
        one_hot = F.one_hot(idx, self.n_codes).float()  # (N, K)
        counts = one_hot.sum(dim=0)  # (K,)
        sums = one_hot.T @ z  # (K, D)

        decay = self.ema_decay
        self.ema_counts.mul_(decay).add_(counts, alpha=1 - decay)
        self.ema_sums.mul_(decay).add_(sums, alpha=1 - decay)

        # Actualizar sólo códigos con suficiente evidencia
        n_eff = self.ema_counts.clamp(min=1e-6).unsqueeze(-1)
        new_codes = self.ema_sums / n_eff
        mask = (self.ema_counts > self.min_count_for_update).unsqueeze(-1)
        self.codebook.copy_(torch.where(mask, new_codes, self.codebook))

    @torch.no_grad()
    def reinit_dead_codes(self, min_count: float = 0.01) -> int:
        """Reinicializa códigos muertos con muestras recientes (si hay)."""
        dead = (self.ema_counts < min_count).nonzero(as_tuple=True)[0]
        n_dead = dead.shape[0]
        if n_dead == 0:
            return 0
        # Reinicializar con perturbaciones de códigos vivos
        live = (self.ema_counts >= min_count).nonzero(as_tuple=True)[0]
        if live.shape[0] == 0:
            self.codebook[dead] = torch.randn_like(self.codebook[dead]) * 0.02
        else:
            src = live[torch.randint(0, live.shape[0], (n_dead,), device=live.device)]
            noise = torch.randn_like(self.codebook[dead]) * 0.01
            self.codebook[dead] = self.codebook[src] + noise
        self.ema_counts[dead] = 0.0
        self.ema_sums[dead] = 0.0
        return int(n_dead)

    def get_code_for(self, emb: torch.Tensor) -> torch.Tensor:
        """emb: (..., D). Devuelve code_idx (...)."""
        orig_shape = emb.shape[:-1]
        z = self.proj(emb.float())
        flat = z.reshape(-1, self.dim)
        dists = torch.cdist(flat, self.codebook)
        idx = dists.argmin(dim=-1)
        return idx.reshape(orig_shape)

    def reset(self) -> None:
        nn.init.normal_(self.codebook, std=0.02)
        self.ema_counts.zero_()
        self.ema_sums.zero_()


# ============================================================
# SELF-KNOWLEDGE HEAD
# ============================================================
class SelfKnowledgeHead(nn.Module):
    """Predice la confianza del modelo por token (entrenable).

    Entrada: hidden state (B, T, D) + concepto (tok_emb, B, T, D).
    Salida: confidence (B, T) en [0, 1].

    Auto-supervisión: confidence_target = (argmax logits == target).
    Se entrena con BCE.
    """

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
        """BCE entre confidence y acierto del modelo."""
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
    """Cross-attention sobre claves de memoria (razonamiento latente).

    Opcional. Añade latencia. Desactivado por defecto.
    """

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

        # mem: (B, N, D) — expandir batch
        B = x.shape[0]
        mem = mem_keys.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, mem)

        gate = torch.sigmoid(self.gate(x))
        return x * gate
