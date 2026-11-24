"""Componentes de SNN: neurona LIF + FFN con modos configurables.

Modos del FFN:
  * static     -> surrogate spike por token (sin membrana)
  * multi_step -> LIF con N pasos internos por token (paralelo en T, cache OK)
  * temporal   -> LIF a lo largo de la secuencia (BPTT, sin cache)
  * gelu       -> FFN estándar GPT-2
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- surrogate
class SurrogateSpike(torch.autograd.Function):
    """Forward: Heaviside. Backward: triángulo centrado (surrogate)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = torch.clamp(1.0 - x.abs() / alpha, min=0.0) / alpha
        return grad_out * grad, None


def spike_fn(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return SurrogateSpike.apply(x, alpha)


# ---------------------------------------------------------------- LIF neuron
class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire.

    Dinámica por paso:
        v_t = decay * v_{t-1} + I_t
        s_t = H(v_t - threshold)
        v_t = v_t - s_t * threshold        (soft reset)
        v_t = v_t * (1 - s_t)              (hard reset)
    """

    def __init__(
        self,
        threshold: float = 1.0,
        decay: float = 0.9,
        alpha: float = 2.0,
        reset_mode: str = "soft",
        learnable_threshold: bool = False,
        learnable_decay: bool = False,
    ):
        super().__init__()
        if reset_mode not in ("soft", "hard"):
            raise ValueError(f"reset_mode must be 'soft'|'hard', got {reset_mode!r}")
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"decay must be in (0,1), got {decay}")

        self.alpha = float(alpha)
        self.reset_mode = reset_mode

        if learnable_threshold:
            self.threshold = nn.Parameter(torch.tensor(float(threshold)))
        else:
            self.register_buffer(
                "threshold", torch.tensor(float(threshold)), persistent=False
            )

        if learnable_decay:
            logit = torch.logit(torch.tensor(float(decay)).clamp(1e-3, 1 - 1e-3))
            self.decay_logit = nn.Parameter(logit)
        else:
            self.register_buffer(
                "decay", torch.tensor(float(decay)), persistent=False
            )

    def _decay(self) -> torch.Tensor:
        if hasattr(self, "decay_logit"):
            return torch.sigmoid(self.decay_logit)
        return self.decay

    def _threshold(self) -> torch.Tensor:
        return self.threshold

    # ----------- variante 1: pasos internos por posición (paralelo en T)
    def forward(self, x: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """x: (B, T, C). Cada posición tiene su propia membrana independiente."""
        B, T, C = x.shape
        v = torch.zeros_like(x)
        total = torch.zeros_like(x)
        decay = self._decay().to(x.dtype)
        threshold = self._threshold().to(x.dtype)

        for _ in range(num_steps):
            v = decay * v + x
            s = spike_fn(v - threshold, self.alpha)
            if self.reset_mode == "soft":
                v = v - s * threshold
            else:
                v = v * (1 - s)
            total = total + s

        return total

    # ----------- variante 2: la secuencia ES el tiempo (BPTT real)
    def forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C). T se interpreta como tiempo. BPTT canónico."""
        B, T, C = x.shape
        v = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        decay = self._decay().to(x.dtype)
        threshold = self._threshold().to(x.dtype)

        spikes = []
        for t in range(T):
            v = decay * v + x[:, t]
            s = spike_fn(v - threshold, self.alpha)
            if self.reset_mode == "soft":
                v = v - s * threshold
            else:
                v = v * (1 - s)
            spikes.append(s)

        return torch.stack(spikes, dim=1)


# ---------------------------------------------------------------- spiking FFN
class SpikingFFN(nn.Module):
    """FFN con activación SNN configurable.

    `use_spiking=False` fuerza modo GELU. En otro caso se usa `mode`.
    """

    def __init__(
        self,
        n_embd: int,
        mult: int = 4,
        threshold: float = 1.0,
        alpha: float = 2.0,
        dropout: float = 0.0,
        use_spiking: bool = True,
        residual: bool = True,
        mode: str = "multi_step",
        decay: float = 0.9,
        reset_mode: str = "soft",
        num_steps: int = 4,
        learnable_threshold: bool = False,
        learnable_decay: bool = False,
        track_stats: bool = False,
    ):
        super().__init__()
        if mode not in ("static", "multi_step", "temporal", "gelu"):
            raise ValueError(f"invalid mode: {mode!r}")

        self.fc1 = nn.Linear(n_embd, n_embd * mult, bias=False)
        self.fc2 = nn.Linear(n_embd * mult, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

        self.use_spiking = use_spiking
        self.mode = "gelu" if not use_spiking else mode
        self.residual = residual
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.track_stats = bool(track_stats)

        if self.mode in ("multi_step", "temporal"):
            self.lif = LIFNeuron(
                threshold=threshold,
                decay=decay,
                alpha=alpha,
                reset_mode=reset_mode,
                learnable_threshold=learnable_threshold,
                learnable_decay=learnable_decay,
            )

        if self.track_stats:
            self.register_buffer(
                "last_firing_rate", torch.tensor(0.0), persistent=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)

        if self.mode == "gelu":
            h = F.gelu(h)

        elif self.mode == "static":
            s = spike_fn(h - self.threshold, self.alpha)
            if self.track_stats:
                self.last_firing_rate.copy_(s.detach().mean())
            h = s * h if self.residual else s

        elif self.mode == "multi_step":
            s = self.lif(h, self.num_steps)
            if self.track_stats:
                self.last_firing_rate.copy_(s.detach().mean() / self.num_steps)
            h = s * h if self.residual else s

        elif self.mode == "temporal":
            s = self.lif.forward_temporal(h)
            if self.track_stats:
                self.last_firing_rate.copy_(s.detach().mean())
            h = s * h if self.residual else s

        else:
            raise RuntimeError(f"unreachable mode {self.mode}")

        return self.drop(self.fc2(h))


# ---------------------------------------------------------------- alias legacy
class SpikingActivation(nn.Module):
    """Compat: activación puntual (sin membrana)."""

    def __init__(
        self, threshold: float = 1.0, alpha: float = 2.0, residual: bool = True
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = spike_fn(x - self.threshold, self.alpha)
        return s * x if self.residual else s
