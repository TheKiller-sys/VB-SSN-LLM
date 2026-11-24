# SNN-LM v3 — Spiking Language Model con surrogate gradients

Transformer estilo GPT-2 donde la FFN dispara spikes.
Entrena con backprop + surrogate gradient, infiere con spikes binarios reales.

## Instalación

    pip install -r requirements.txt

## Demo

    python main.py

## Entrenar

    # tiny (~15M params), RTX 3060, 1-2 horas en corpus de 10 MB
    python scripts/train.py data/ --size tiny --device cuda

    # small (~50M params), RTX 3090, 8-12 horas
    python scripts/train.py data/ --size small --device cuda

    # medium (~124M params), A100, 1-2 días
    python scripts/train.py data/ --size medium --device cuda

## Continuar entrenamiento

    python scripts/train.py data/ --resume checkpoints/latest.pt

## Generar

    python scripts/generate.py --ckpt checkpoints/latest.pt \
        --prompt "The quick brown" --max-tokens 200 --temperature 0.8

## Arquitectura

    token_id → Embedding(50257, d) → weight-tied con head
       ↓
    Block × N:
       ├─ LayerNorm → MultiHeadAttention(RoPE, causal) → residual
       └─ LayerNorm → SpikingFFN(surrogate gradient) → residual
       ↓
    LayerNorm → Head → softmax

- **Attention**: multi-head estándar, RoPE, causal masking.
- **FFN**: spiking (surrogate triángulo en backward, step en forward).
- **Entrenamiento**: backprop normal con surrogate, AdamW, cosine schedule,
  gradient clipping, mixed precision.
- **Inferencia**: spikes binarios (0/1), listos para hardware neuromórfico.

## Tamaños

| Size | Capas | Heads | d_model | Params | GPU |
|---|---|---|---|---|---|
| tiny | 6 | 8 | 256 | ~15M | 1× 3060 |
| small | 8 | 8 | 512 | ~50M | 1× 3090 |
| medium | 12 | 12 | 768 | ~124M | 1× A100 |

## Por qué surrogate gradients

La función `spike(x) = 1[x > 0]` tiene derivada 0 casi en todo punto.
Sin un surrogate, el gradiente no fluye y el modelo no entrena.
Usamos `∂spike/∂x ≈ (1/α)·max(0, 1 − |x|/α)` (triángulo) — estándar
en SNN modernas (STBP, SuperSpike, Spikformer).

## Limitaciones

- No es un LLM general. Es un LM entrenado sobre tu corpus.
- 124M params ≈ GPT-2 small. Perplejidad en texto natural será ~20-40,
  no ~15 como GPT-2 (por la spiking FFN).
- Corpus de 10 MB → coherencia local; 1 GB → coherencia de párrafo.
- Necesita GPU para tamaños medium.

## Estructura

    snn_lm_v3/       paquete
    scripts/         train.py, generate.py
    data/            tus .txt
    checkpoints/     config.json + latest.pt + step_N.pt
