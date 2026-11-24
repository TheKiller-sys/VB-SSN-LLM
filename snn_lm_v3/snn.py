import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    """Forward: Heaviside. Backward: triángulo centrado (surrogate gradient)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = torch.clamp(1.0 - x.abs() / alpha, min=0.0) / alpha
        return grad_out * grad, None


def spike_fn(x, alpha=2.0):
    return SurrogateSpike.apply(x, alpha)


class SpikingActivation(nn.Module):
    def __init__(self, threshold=1.0, alpha=2.0):
        super().__init__()
        self.threshold = float(threshold)
        self.alpha = float(alpha)

    def forward(self, x):
        return spike_fn(x - self.threshold, self.alpha)


class SpikingFFN(nn.Module):
    """FFN cuya no-linealidad es un disparo de spike con surrogate gradient.

    Inferencia: salida binaria (0/1). Entrenamiento: gradiente aproximado.
    """

    def __init__(self, n_embd, mult=4, threshold=1.0, alpha=2.0, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, n_embd * mult, bias=False)
        self.act = SpikingActivation(threshold, alpha)
        self.fc2 = nn.Linear(n_embd * mult, n_embd, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        s = self.act(h)
        s = s * h  # spikes × magnitud (residual sobre el disparo)
        return self.drop(self.fc2(s))
