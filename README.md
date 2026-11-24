# SNN-LM v3.1

Transformer causal (estilo GPT-2, RoPE, pre-LN) cuyo FFN usa una **neurona
LIF real con surrogate gradient**, más una **capa de razonamiento simbólico
sobre el corpus** portada de conceptos de SAL/VerifierCritic.

## Qué incluye

- **Modelo** (`snn_lm_v3/`):
  - `GPTConfig` con validación.
  - RoPE, atención causal con KV-cache real.
  - `LIFNeuron` con membrana, umbral, reset (soft/hard), decay.
  - 4 modos de FFN: `static`, `multi_step`, `temporal`, `gelu`.
  - Entrenamiento con AMP CUDA (fp16) / CPU (bf16), grad-accum, resume
    bit-reproducible, `best.pt`, checkpoints atómicos.
- **Razonamiento** (`snn_lm_v3/reasoning/`):
  - `CorpusObserver`: extrae tokens, transiciones y secuencias del corpus.
  - `ReasoningGraph`: motor de reglas con anti-ciclo, anti-tautología,
    cooldown por paso, pruning por confianza×uso×recencia, decay.
  - `ConsistencyVerifier`: verifica texto generado contra corpus y contra
    señales internas (firing rate, entropy, perplexity).
  - `SignalProbe`: extrae señales internas del modelo.
  - Persistencia (`save_graph` / `load_graph`).
- **Scripts** (`scripts/`, `main.py`): demo, train, generate, reason.
- **Tests** (`tests/`): cobertura de todos los módulos.

## Instalación

```bash
pip install -r requirements.txt
# desarrollo:
pip install -r requirements-dev.txt
```

## Demo end-to-end

```bash
python main.py
python main.py --size small --steps 1000 --device cuda
python main.py --size tiny --steps 100 --no-reasoning
```

## Entrenamiento

```bash
# básico
python scripts/train.py data/ --size tiny

# con razonamiento y verificación final
python scripts/train.py data/ --size tiny --snn-track-stats

# resume
python scripts/train.py data/ --resume checkpoints/latest.pt

# config completo
python scripts/train.py data/ --config cfg.json

# modo temporal (SNN canónica, lento)
python scripts/train.py data/ --size tiny --snn-mode temporal
```

## Generación

```bash
# simple
python scripts/generate.py --ckpt checkpoints/best.pt --prompt "Once upon a time"

# con verificación contra corpus
python scripts/generate.py --ckpt checkpoints/best.pt \
    --prompt "The quick brown" --corpus data/ --verify
```

## Razonamiento sobre corpus

```bash
# construir grafo y guardarlo
python scripts/reason.py --corpus data/ --mine --save checkpoints/graph.pt

# consultar estadísticas
python scripts/reason.py --graph checkpoints/graph.pt --stats

# aplicar facts
python scripts/reason.py --graph checkpoints/graph.pt --apply quick fox lazy
```

## API mínima

```python
from snn_lm_v3 import (
    GPTConfig, SNNLM, make_dataloader, train, generate,
    CorpusObserver, ReasoningGraph, ConsistencyVerifier,
)

cfg = GPTConfig(n_layer=6, n_head=8, n_embd=256, seq_len=256)
model = SNNLM(cfg)

train_dl, val_dl, enc = make_dataloader(["data/"], cfg.seq_len, cfg.batch_size)
model = train(model, train_dl, val_dl, cfg)

text = generate(model, prompt="Hello", max_new_tokens=50)
print(text)

# Razonamiento
observer = CorpusObserver()
for chunk in ["The quick brown fox jumps", "The lazy dog sleeps"]:
    observer.observe(chunk)
graph = ReasoningGraph()
graph.mine_from_corpus(observer)

verifier = ConsistencyVerifier()
problems = verifier.verify_against_corpus(text, graph, observer)
print(verifier.verdict(problems))
```

## Tests

```bash
python -m pytest tests/
```

## Modos SNN

| Modo | Dinámica | Cache | `torch.compile` | Uso |
|---|---|---|---|---|
| `static` | Spike por token (sin membrana) | ✅ | ✅ | Baseline mínimo |
| `multi_step` | LIF con N pasos internos por token | ✅ | ✅ | **Producción** |
| `temporal` | LIF sobre secuencia (BPTT) | ❌ | ❌ | Investigación |
| `gelu` | FFN GPT-2 estándar | ✅ | ✅ | Ablación |

## Advertencias honestas

- `multi_step` y `temporal` son **SNNs reales** (membrana, umbral, reset,
  integración con fugas). No son SNN de hardware neuromórfico: corren en
  GPU/CPU estándar. `temporal` es la SNN canónica donde la secuencia *es*
  el tiempo; `multi_step` es el LIF aplicado por token, que en la práctica
  es lo que hacen SpikeGPT/Spikingformer para mantener paralelismo.
- La **capa de razonamiento** mina co-ocurrencias, no razonamiento lógico.
  Es útil para auditar generación y detectar OOV/degeneración. No infiere
  causalidad ni transitividad.
- `temporal` es 5–10× más lento que `multi_step`. Usar solo con `seq_len ≤ 256`.
- El sistema **no incluye** DDP/FSDP, atención Flash explícita, ni
  tokenizadores alternativos. Ver sección "Proyecciones" abajo.

## Proyecciones (no incluidas)

Si quieres llevar el sistema más lejos, en orden de coste/beneficio:

1. **DDP** (~30 líneas): escalar a multi-GPU.
2. **`attn_backend`** (~50 líneas): xFormers / FlashAttention-2.
3. **Constrained decoding** (~80 líneas): enmascarar tokens que violen
   reglas del grafo en `generate()`. Convierte el grafo de auditor en guía.
4. **Few-shot dinámico** (~40 líneas): inyectar top-K facts del grafo al
   prompt antes de generar.
5. **FSDP + sharded checkpoints** (~150 líneas): >1B params.
6. **Tokenizer plug-and-play** (~80 líneas): Llama-3, custom.

## Licencia

MIT.
