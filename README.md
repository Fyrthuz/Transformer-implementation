# Transformer Experiment

Implementación modular de un **Transformer decoder-only** (estilo GPT) con 9 mecanismos de atención, 4 variantes de FFN, 3 codificaciones posicionales, múltiples datasets, loss functions customizables, **CLI completa** y **grid search automático** con métricas vía TensorBoard.

## Instalación

```bash
# Con uv (recomendado)
uv sync

# Con pip
pip install -r requirements.txt
```

## Uso rápido

```bash
# Script todo-en-uno (TensorBoard automático)
./run_experiment.sh train -c configs/gqa_swiglu.yaml

# Sin script
python run.py train -c configs/gqa_swiglu.yaml

# Sweep (grid search automático)
python run.py sweep -c configs/sweep_example.yaml

# Generar texto con modelo entrenado
python run.py generate -m best_model.pt -p "ROMEO:\n" -t 0.7

# Listar componentes disponibles
python run.py list

# TensorBoard
tensorboard --logdir=runs
```

---

## Componentes

### 9 Mecanismos de Atención

| Nombre | CLI | Paper | Descripción |
|--------|-----|-------|-------------|
| Multi-Head Attention | `mha` | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Atención estándar con Q, K, V por head |
| Multi-Query Attention | `mqa` | [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) | Un solo K,V compartido entre todos los heads |
| Grouped Query Attention | `gqa` | [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) | K,V agrupados, intermedio entre MHA y MQA |
| Linear Attention | `linear` | [Transformers are RNNs](https://arxiv.org/abs/2006.16236) | Kernel ELU, complejidad O(N) lineal |
| Window Attention | `window` | [Longformer](https://arxiv.org/abs/2004.05150) | Atención restringida a una ventana local |
| Dilated Attention | `dilated` | [Longformer](https://arxiv.org/abs/2004.05150) | Atención con patrón de dilatación periódico |
| Global-Local Attention | `global_local` | [BigBird](https://arxiv.org/abs/2007.14062) | Tokens globales + atención local por ventana |
| Mamba | `mamba` | [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) | SSM selectivo simplificado (S6) |
| SSM | `ssm` | [Structured State Spaces](https://arxiv.org/abs/2111.00396) | State Space Model básico |

### 4 Variantes de FFN

| Nombre | CLI | Descripción |
|--------|-----|-------------|
| Standard | `standard` | `Linear → GELU/ReLU → Linear` |
| SwiGLU | `swiglu` | `(SiLU(xW₁) ⊗ xW₃)W₂` (PaLM, Llama) |
| Gated | `gated` | `(GELU(xW₁) ⊗ σ(xW_gate))W₂` |
| Mixture of Experts | `moe` | Router top-k con N expertos + load balancing loss |

### 3 Codificaciones Posicionales

| Nombre | CLI | Descripción |
|--------|-----|-------------|
| Sinusoidal | `sinusoidal` | Senos/cosenos del paper original |
| RoPE | `rope` | Rotary Position Embedding (Llama, GPT-NeoX) |
| None | `none` | Sin PE explícita (para Mamba/SSM) |

### 3 Datasets

| Nombre | CLI | Tokenización | Tamaño |
|--------|-----|-------------|--------|
| Tiny Shakespeare | `tinyshakespeare` | char o BPE | ~1MB |
| WikiText-2 | `wikitext2` | BPE | ~10MB |
| WikiText-103 | `wikitext103` | BPE | ~180MB |

### 4 Loss Functions

| Nombre | CLI | Descripción |
|--------|-----|-------------|
| Cross-Entropy | `cross_entropy` | Con label smoothing configurable |
| NLL Loss | `nll` | Negative Log Likelihood |
| MSE Loss | `mse` | Mean Squared Error |
| Focal Loss | `focal` | Focal loss con γ configurable |

---

## Sistema de Configuración

Toda la configuración se define en YAML. Ejemplo completo:

```yaml
name: mi_experimento

model:
  d_model: 256
  num_layers: 4

  # Regularización — dropout por componente + stochastic depth
  dropout: 0.2                    # fallback global para todos los dropouts
  attention_dropout: 0.2          # dropout interno de atención (si null, usa dropout)
  ffn_dropout: 0.2                # dropout interno de FFN (si null, usa dropout)
  embedding_dropout: 0.1          # dropout en embedding + posición (si null, usa dropout)
  stochastic_depth: 0.1           # prob. de saltar capas (LayerDrop). 0 = desactivado

  attention:
    type: gqa               # mha | mqa | gqa | linear | window | dilated | global_local | mamba | ssm
    num_heads: 8
    num_kv_heads: 2          # solo para gqa
    window_size: 64          # solo para window / dilated
    dilation: 4              # solo para dilated
    d_state: 16              # solo para mamba / ssm
    expand_factor: 2         # solo para mamba
    d_conv: 4                # solo para mamba

  ffn:
    type: swiglu             # standard | swiglu | gated | moe
    d_ff: 1024
    activation: gelu         # solo para standard
    num_experts: 8           # solo para moe
    top_k: 2                 # solo para moe

  position:
    type: rope               # sinusoidal | rope | none
    max_len: 5000
    rope_theta: 10000.0      # solo para rope

dataset:
  name: wikitext2
  tokenization: bpe          # char | bpe
  vocab_size: 4096
  max_seq_len: 128
  train_stride: 16
  cache_dir: ./data
  max_train_chunks: 35000
  max_test_chunks: 2000

training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.0005
  weight_decay: 0.1          # regularización L2
  grad_clip: 1.0
  scheduler: cosine          # cosine | linear | none
  early_stop_patience: 5     # 0 = desactivado. Detiene training si test loss no mejora

  loss:
    type: cross_entropy      # cross_entropy | nll | mse | focal
    label_smoothing: 0.1     # suaviza targets, reduce overfitting
    ignore_index: auto
    moe_load_balance_coef: 0.01   # peso auxiliar para MoE
    focal_gamma: 2.0              # solo para focal
```

### CLI Overrides

Cualquier campo del YAML se sobreescribe desde la CLI:

```bash
python run.py train -c configs/gqa_swiglu.yaml \
  model.attention.type=gqa \
  model.attention.num_kv_heads=2 \
  model.ffn.type=swiglu \
  model.position.type=rope \
  training.num_epochs=50 \
  dataset.name=wikitext2
```

---

## CLI Completa

### `train` — Entrenar un modelo

```bash
# Desde config YAML
python run.py train -c configs/gqa_swiglu.yaml

# Config inline (sin YAML)
python run.py train \
  model.attention.type=mamba \
  model.attention.d_state=32 \
  model.ffn.type=swiglu \
  model.position.type=none \
  dataset.name=tinyshakespeare \
  dataset.tokenization=char \
  training.num_epochs=30 \
  name=mamba_test
```

### `generate` — Generar texto

```bash
python run.py generate \
  -m best_model.pt \
  -p "ROMEO:\n" \
  -t 0.7 \
  -n 600 \
  -c configs/gqa_swiglu.yaml
```

### `sweep` — Grid Search automático

```yaml
# configs/sweep_example.yaml
sweep:
  strategy: grid
  max_combinations: 12
  repetitions: 1

  model.attention.type: [mha, mqa, gqa, linear]
  model.attention.num_heads: [8]
  model.ffn.type: [swiglu]
  training.num_epochs: [50]
  dataset.name: [wikitext2]
```

```bash
python run.py sweep -c configs/sweep_example.yaml
```

Cada combinación se entrena secuencialmente y al final se imprime una tabla con resultados ordenados por loss.

### Grid Search completo en TinyShakespeare (108 combinaciones)

```bash
./run_experiment.sh sweep -c configs/sweep_shakespeare.yaml

# O directamente:
python run.py sweep -c configs/sweep_shakespeare.yaml

# Con TensorBoard aparte:
tensorboard --logdir=runs --port=6006
```

Barre las 108 combinaciones (9 atenciones × 4 FFN × 3 posiciones) con modelos pequeños (d_model=64, 2 capas) en TinyShakespeare. Cada combinación entrena 20 epochs y genera su propia carpeta en `runs/`. Al final imprime un ranking top-30 por perplexity y exporta un JSON con todos los resultados.

### `list` — Listar componentes

```bash
python run.py list
```

Muestra todos los mecanismos de atención, FFNs, posiciones y datasets disponibles.

---

## Script todo-en-uno

`run_experiment.sh` lanza TensorBoard automáticamente junto con el experimento:

```bash
# Entrenar con TensorBoard incluido
./run_experiment.sh train -c configs/gqa_swiglu.yaml

# Con overrides
./run_experiment.sh train -c configs/gqa_swiglu.yaml model.attention.type=gqa

# Grid search
./run_experiment.sh sweep -c configs/sweep_example.yaml

# Solo TensorBoard
./run_experiment.sh tensorboard-only

# Sin TensorBoard
NO_TENSORBOARD=1 ./run_experiment.sh train -c configs/gqa_swiglu.yaml
```

Variables de entorno:

| Variable | Default | Descripción |
|----------|---------|-------------|
| `TENSORBOARD_PORT` | `6006` | Puerto de TensorBoard |
| `TENSORBOARD_DIR` | `./runs` | Directorio de logs |
| `NO_TENSORBOARD` | — | `=1` desactiva TensorBoard automático |

---

## TensorBoard

```bash
tensorboard --logdir=runs --port=6006
# Abre http://localhost:6006
```

Cada experimento genera `runs/{experiment_name}/` con las siguientes métricas en **paneles separados**:

| Panel | Métrica | Frecuencia |
|-------|---------|------------|
| `Train/Loss_step` | Loss por paso | cada batch |
| `Train/Loss` | Loss media de entrenamiento | cada epoch |
| `Test/Loss` | Loss de validación | cada epoch |
| `Train/Perplexity` | Perplexity de entrenamiento | cada epoch |
| `Test/Perplexity` | Perplexity de validación | cada epoch |
| `Params/Learning_Rate` | Tasa de aprendizaje | cada epoch |
| `Samples/Prediction_Test` | Texto generado de ejemplo | cada 5 epochs |

Además, la pestaña **HPARAMS** permite comparar experimentos lado a lado con tabla de hiperparámetros.

### Comparar múltiples experimentos

En TensorBoard:
1. Abre la pestaña **Scalars**
2. En el panel izquierdo, selecciona los experimentos que quieras comparar
3. Las curvas se superponen automáticamente

### Comparar resultados del sweep en TensorBoard

Tras ejecutar `run.py sweep -c configs/sweep_shakespeare.yaml`, cada combinación genera su propia carpeta en `runs/`:

```bash
tensorboard --logdir=runs --port=6006
```

Usa la pestaña **HPARAMS** de TensorBoard para comparar todas las combinaciones lado a lado: ordena por `test_perplexity` para ver las configuraciones ganadoras, o filtra por tipo de atención/FFN/posición para encontrar patrones. Además, el sweep exporta un JSON con el ranking completo.

---

## Regularización contra Overfitting

El sistema incluye varias técnicas de regularización configurables:

| Técnica | Parámetro | Default | Efecto |
|---------|-----------|---------|--------|
| **Stochastic Depth** | `model.stochastic_depth` | `0.0` | Saltea capas al azar durante training (decaimiento lineal: primeras capas menos, últimas más). Scaling en inferencia. |
| **Dropout en atención** | `model.attention_dropout` | = `dropout` | Dropout interno en cada head de atención |
| **Dropout en FFN** | `model.ffn_dropout` | = `dropout` | Dropout entre capas del FFN |
| **Dropout en embedding** | `model.embedding_dropout` | = `dropout` | Dropout en embedding + posición |
| **Weight Decay** | `training.weight_decay` | `0.01` | Regularización L2 en AdamW |
| **Label Smoothing** | `training.loss.label_smoothing` | `0.025` | Suaviza las targets, evita confianza excesiva |
| **Early Stopping** | `training.early_stop_patience` | `0` | Detiene training si test loss no mejora en N epochs |
| **Gradient Clipping** | `training.grad_clip` | `1.0` | Norma máxima del gradiente |

### Stochastic Depth

Capa por capa, la probabilidad de dropout se escala linealmente:

```
drop_prob(layer) = start + (end - start) * layer / (num_layers - 1)
```

Donde `start = stochastic_depth / 2` y `end = stochastic_depth`. Las primeras capas se saltan menos, las últimas más. Durante inferencia todas las capas se ejecutan con scaling `1 / (1 - drop_prob)`.

### Recomendación para WikiText-2

```yaml
model:
  dropout: 0.2
  attention_dropout: 0.2
  ffn_dropout: 0.2
  embedding_dropout: 0.1
  stochastic_depth: 0.1

training:
  weight_decay: 0.1
  early_stop_patience: 5

  loss:
    label_smoothing: 0.1
```

## Perplexity

Se calcula como `exp(loss)` al final de cada epoch:

```
Train Perplexity = exp(Train Loss)
Test Perplexity  = exp(Test Loss)
```

- Se loguea a TensorBoard como `Train/Perplexity` y `Test/Perplexity`
- Se muestra en consola junto al loss
- Es el criterio para guardar el mejor checkpoint (`best_model.pt`)
- Se registra en la tabla HPARAMS de TensorBoard

---

## Configuraciones de ejemplo

| Archivo | Atención | FFN | Posición | Dataset |
|---------|----------|-----|----------|---------|
| `configs/gqa_swiglu.yaml` | MHA | Standard | RoPE | WikiText-2 |
| `configs/gqa_swiglu.yaml` | GQA (2 KV heads) | SwiGLU | RoPE | WikiText-2 |
| `configs/mamba_bpe.yaml` | Mamba | Standard | None | WikiText-2 |
| `configs/moe_linear.yaml` | Linear | MoE (8 experts) | Sinusoidal | WikiText-2 |
| `configs/wikitext_mqa_rope.yaml` | MQA | SwiGLU | RoPE | WikiText-2 |
| `configs/sweep_example.yaml` | Grid search comparando MHA, MQA, GQA, Linear | — | — | WikiText-2 |
| `configs/sweep_shakespeare.yaml` | **108 combos**: todas las atenciones × FFN × posiciones | — | — | TinyShakespeare |

---

## Estructura del Proyecto

```
transformer_impl/              # Paquete principal
  ├── __init__.py              # Export público
  ├── config.py                # Config: dataclasses + YAML loader + CLI parser
  ├── model.py                 # Transformer: embedding + N capas + output
  ├── embedding.py             # Token embedding + posición
  ├── train.py                 # Training loop + TensorBoard + perplexity + checkpoint
  ├── generate.py              # Generación con sampling por temperatura
  │
  ├── attention/               # 9 mecanismos de atención
  │   ├── __init__.py          # Registry: mha, mqa, gqa, linear, window, dilated, global_local, mamba, ssm
  │   ├── base.py              # Clase abstracta BaseAttention
  │   ├── mha.py               # Multi-Head Attention
  │   ├── mqa.py               # Multi-Query Attention
  │   ├── gqa.py               # Grouped Query Attention
  │   ├── linear_attn.py       # Linear Attention (kernel ELU)
  │   ├── window_attn.py       # Sliding Window Attention
  │   ├── dilated_attn.py      # Dilated Attention
  │   ├── global_local.py      # Global-Local Attention
  │   ├── mamba.py             # Mamba SSM simplificado
  │   └── ssm.py               # State Space Model
  │
  ├── ffn/                     # 4 variantes de FFN
  │   ├── __init__.py          # Registry: standard, swiglu, gated, moe
  │   ├── base.py              # Clase abstracta BaseFFN
  │   ├── standard.py          # Linear → GELU/ReLU → Linear
  │   ├── swiglu.py            # SwiGLU (SiLU gate)
  │   ├── gated.py             # Gated FFN (sigmoid gate)
  │   └── moe.py               # Mixture of Experts con load balancing loss
  │
  ├── position/                # 3 codificaciones posicionales
  │   ├── __init__.py          # Registry: sinusoidal, rope, none
  │   ├── base.py
  │   ├── sinusoidal.py        # Senos/cosenos (Vaswani et al.)
  │   ├── rope.py              # Rotary Position Embedding
  │   └── none.py              # Identity (sin PE)
  │
  ├── blocks/
  │   ├── __init__.py
  │   └── transformer_block.py # Pre-LN block: norm → attn → residuo → norm → ffn → residuo
  │
  └── datasets/                # Cargadores de datasets
      ├── __init__.py          # Registry: tinyshakespeare, wikitext2, wikitext103
      ├── base.py              # Dataset preparer abstracto
      ├── tinyshakespeare.py   # Tiny Shakespeare (char o BPE)
      ├── wikitext.py          # WikiText-2 y WikiText-103
      └── tokenizer.py         # CharTokenizer + BPETokenizer

configs/                       # 6 configuraciones YAML de ejemplo
run.py                         # CLI principal (argparse, 4 subcomandos)
run_experiment.sh              # Script bash con TensorBoard automático
```

---

## Añadir un nuevo componente

### Nueva atención

```python
# transformer_impl/attention/mi_atencion.py
import torch
from . import register_attention

@register_attention("mi_atencion")
class MiAtencion(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        # ...

    def forward(self, x, mask=None):
        # ...
        return x
```

### Nuevo FFN

```python
# transformer_impl/ffn/mi_ffn.py
from . import register_ffn

@register_ffn("mi_ffn")
class MiFFN(BaseFFN):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # ...

    def forward(self, x):
        return x
```

### Nuevo dataset

```python
# transformer_impl/datasets/mi_dataset.py
from . import register_dataset

@register_dataset("mi_dataset")
class MiDataset(BaseDatasetPreparer):
    def prepare(self, cfg):
        # ...
        return DatasetOutput(...)
```

Los decoradores `@register_*` registran automáticamente el componente y estará disponible vía CLI y YAML.

---

## Dependencias

```
torch>=2.1.0
tokenizers>=0.13.3    → BPE tokenizer
pyyaml>=6.0           → Configuración YAML
tqdm>=4.65.0          → Progress bars
tensorboard>=2.13.0   → Métricas y visualización
```
