"""Componentes de SNN: neurona LIF + FFN con modos configurables.

v4.3: añade `mode="hybrid"` (dense + spiking con gate aprendido).

Modos del FFN:
  * dense      -> FFN estándar GELU (sin LIF, sin spikes)
  * static     -> spike binario por token (sin membrana)
  * multi_step -> LIF con N pasos internos; salida BINARIA con STE
  * temporal   -> LIF a lo largo de la secuencia (BPTT, sin cache)
  * hybrid     -> gate * gelu(fc1_dense) + (1-gate) * lif(fc1_spike)
  * gelu       -> alias legacy de dense
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
    """Leaky Integrate-and-Fire con salida binaria y STE."""

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

    def forward(self, x: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        B, T, C = x.shape
        v = torch.zeros_like(x)
        fired_count = torch.zeros_like(x)
        decay = self._decay().to(x.dtype)
        threshold = self._threshold().to(x.dtype)

        for _ in range(num_steps):
            v = decay * v + x
            s = spike_fn(v - threshold, self.alpha)
            if self.reset_mode == "soft":
                v = v - s * threshold
            else:
                v = v * (1 - s)
            fired_count = fired_count + s

        binary = (fired_count > 0).to(x.dtype)
        return fired_count + (binary - fired_count).detach()

    def forward_temporal(self, x: torch.Tensor) -> torch.Tensor:
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
    """FFN con activación configurable: dense, spiking, o hybrid.

    `ffn_mode`:
      - "dense": fc1 -> GELU -> fc2 (GPT-2 style)
      - "spiking" / "static" / "multi_step" / "temporal": FFN con LIF
      - "hybrid": gate * dense + (1-gate) * spiking
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
        ffn_mode: str | None = None,
        hybrid_gate_init: float = 0.5,
    ):
        super().__init__()

        # Compatibilidad: si ffn_mode se pasa, gana sobre `mode`.
        if ffn_mode is None:
            if not use_spiking:
                ffn_mode = "dense"
            elif mode == "gelu":
                ffn_mode = "dense"
            else:
                ffn_mode = "spiking"
        if ffn_mode not in ("dense", "spiking", "hybrid"):
            raise ValueError(f"invalid ffn_mode: {ffn_mode!r}")
        if ffn_mode == "spiking" and mode not in (
            "static", "multi_step", "temporal"
        ):
            raise ValueError(f"invalid spiking mode: {mode!r}")

        self.ffn_mode = ffn_mode
        self.mode = mode
        self.residual = residual
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.alpha = float(alpha)
        self.track_stats = bool(track_stats)

        # Dense branches
        self.fc1 = nn.Linear(n_embd, n_embd * mult, bias=False)
        self.fc2 = nn.Linear(n_embd * mult, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

        # Spiking branch (solo si aplica)
        if ffn_mode == "spiking":
            self._init_spiking_branch(n_embd, mult, decay, reset_mode,
                                     learnable_threshold, learnable_decay)
        elif ffn_mode == "hybrid":
            self.fc1_spike = nn.Linear(n_embd, n_embd * mult, bias=False)
            self._init_spiking_branch(n_embd, mult, decay, reset_mode,
                                     learnable_threshold, learnable_decay,
                                     lif_attr="_lif")
            # gate aprendido: (n_embd,) -> (n_embd * mult,) logits de mezcla
            # Usamos un escalar por canal para eficiencia.
            gate_init = max(1e-4, min(1.0 - 1e-4, float(hybrid_gate_init)))
            gate_logit = float(torch.logit(torch.tensor(gate_init)))
            self.hybrid_gate_logit = nn.Parameter(
                torch.full((n_embd * mult,), gate_logit)
            )

        if self.track_stats:
            self.register_buffer(
                "last_firing_rate", torch.tensor(0.0), persistent=False
            )

    def _init_spiking_branch(
        self,
        n_embd: int,
        mult: int,
        decay: float,
        reset_mode: str,
        learnable_threshold: bool,
        learnable_decay: bool,
        lif_attr: str = "lif",
    ) -> None:
        if self.mode in ("multi_step", "temporal"):
            lif = LIFNeuron(
                threshold=self.threshold,
                decay=decay,
                alpha=self.alpha,
                reset_mode=reset_mode,
                learnable_threshold=learnable_threshold,
                learnable_decay=learnable_decay,
            )
            setattr(self, lif_attr, lif)

    def _dense_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.fc1(x))

    def _spiking_forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "static":
            s = spike_fn(h - self.threshold, self.alpha)
        elif self.mode == "multi_step":
            s = self.lif(h, self.num_steps)
        elif self.mode == "temporal":
            s = self.lif.forward_temporal(h)
        else:
            raise RuntimeError(f"unreachable spiking mode {self.mode}")

        if self.track_stats:
            self.last_firing_rate.copy_(s.detach().mean())
        return s * h if self.residual else s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_mode == "dense":
            h = self._dense_forward(x)
        elif self.ffn_mode == "spiking":
            h = self._spiking_forward(self.fc1(x))
        else:  # hybrid
            h_dense = self._dense_forward(x)
            h_spike = self._spiking_forward(self.fc1_spike(x))
            gate = torch.sigmoid(self.hybrid_gate_logit)  # (mult*n_embd,)
            h = gate.unsqueeze(0).unsqueeze(0) * h_dense + (
                1.0 - gate
            ).unsqueeze(0).unsqueeze(0) * h_spike
            if self.track_stats:
                # Reportar el gate medio como proxy de "cuánto SNN se usa"
                self.last_firing_rate.copy_(
                    (1.0 - gate.mean()).detach()
                )

        return self.drop(self.fc2(h))

    def extra_repr(self) -> str:
        return (
            f"mode={self.ffn_mode} (spiking={self.mode}), "
            f"residual={self.residual}, "
            f"mult={self.fc1.out_features // self.fc1.in_features}"
        )


# ---------------------------------------------------------------- alias legacy
class SpikingActivation(nn.Module):
    """Compat: activación puntual (sin membrana). Salida binaria."""

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
