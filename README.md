# Transformer Experiment — LLM Training Pipeline

Modular **decoder-only Transformer** (GPT-style) with 9 attention mechanisms, 4 FFN variants, 3 positional encodings, CLI, grid search, and **complete LLM training pipeline**: pre-training, SFT, DPO, PPO, and GRPO.

## Quick Start

```bash
uv sync                        # or: pip install -r requirements.txt
./run_experiment.sh list       # disponibles
./run_experiment.sh train -c configs/pretrain_tinystories.yaml
./run_experiment.sh sweep -c configs/sweep_shakespeare.yaml   # 108 combos
```

## CLI

| Subcommand | Description |
|------------|-------------|
| `train -c <yaml> [overrides...]` | Train model (original) |
| `generate -c <yaml> -m <model.pt> -p <prompt>` | Generate text |
| `sweep -c <sweep.yaml>` | Grid search |
| `test [pytest_args...]` | Run 50 tests with pytest |
| `list` | List components |
| `pretrain -c <yaml>` | Pre-training from scratch (MHA+SwiGLU default) |
| `sft -c <yaml> -m <model.pt>` | Supervised fine-tuning |
| `dpo -c <yaml> -m <model.pt>` | Direct Preference Optimization |
| `ppo -c <yaml> -m <model.pt>` | PPO alignment |
| `grpo -c <yaml> -m <model.pt>` | GRPO reasoning fine-tuning |
| `pipeline -c <pipeline.yaml>` | Multi-stage pipeline (encadena stages, sin grad accum) |

CLI overrides: `python run.py train model.attention.type=mha model.ffn.type=swiglu`

## Pipeline Completo

```
Pretraining ──► SFT ──► DPO/PPO ──► GRPO
(from scratch)  (instruct)  (alignment)  (reasoning)
```

### Comando único

```bash
python run.py pipeline -c configs/pipeline_full.yaml \
  model.d_model=256 model.num_layers=6 training.batch_size=32
```

Ejecuta **pretrain → SFT → DPO** (o + GRPO) secuencialmente, pasando el mejor checkpoint de cada etapa a la siguiente.

```bash
# 3 etapas (rápido, recomendado):
./run_experiment.sh pipeline -c configs/pipeline_sft_dpo.yaml

# 4 etapas (incluye GRPO, lento):
./run_experiment.sh pipeline -c configs/pipeline_full.yaml
```

> **Arquitectura por defecto**: MHA + SwiGLU + sin posición (rápida, estable en memoria).  
> **TensorBoard**: cada run tiene timestamp único en `runs/`. Los logs incluyen scalars (loss, LR, grad norm, perplexity), histogramas de gradientes, y ejemplos de texto generado.

### Etapa por etapa

```bash
# 1. Pre-training
python run.py pretrain -c configs/pretrain_tinystories.yaml

# 2. SFT
python run.py sft -c configs/sft_alpaca.yaml -m best_model.pt

# 3. DPO Alignment
python run.py dpo -c configs/dpo_ultrafeedback.yaml -m best_model.pt

# 4. GRPO Reasoning
python run.py grpo -c configs/grpo_gsm8k.yaml -m best_model.pt
```

## Fine-tuning con Configuración Diferente

Puedes cargar un checkpoint entrenado con una arquitectura y fine-tunearlo con **otra distinta**:

```bash
# Pre-entrenar con modelo pequeño (6 layers, d_model=256)
python run.py pretrain -c configs/pretrain_tinystories.yaml \
  model.num_layers=6 model.d_model=256

# SFT con modelo más grande, cargando checkpoint parcial
python run.py sft -c configs/sft_alpaca.yaml \
  model.num_layers=8 model.d_model=512 \
  -m best_model.pt
```

El sistema adapta automáticamente:

| Cambio | Comportamiento |
|--------|----------------|
| `vocab_size` diferente | Redimensiona embedding + output layer |
| `num_layers` diferente | Carga capas comunes, resto aleatorio |
| `d_model` / `num_heads` diferente | Omite capas incompatibles |

## Datasets Open-Source (Auto-download)

| Etapa | Dataset | Ejemplos | Licencia |
|-------|---------|----------|----------|
| Pre-training | TinyStories | 2.1M cuentos | CDLA-Sharing |
| Pre-training | FineWeb | ~10B tokens sample | ODC-BY |
| SFT | Alpaca Cleaned | 52K instruct | CC BY NC 4.0 |
| SFT | OpenAssistant | 88K conv. | Apache 2.0 |
| SFT | Dolly | 15K instruct | CC BY-SA 3.0 |
| DPO/PPO | UltraFeedback | 61K pares | MIT |
| DPO/PPO | HH-RLHF | 170K pares | MIT |
| GRPO | GSM8K | 8.5K math | MIT |
| GRPO | MATH | 12.5K math | MIT |
| GRPO | MBPP | 974 code | CC BY 4.0 |

Ver [`docs/datasets_new.md`](docs/datasets_new.md) para detalles completos.

## Results — TinyShakespeare Sweep (108 combos)

Grid search over 9 attention × 4 FFN × 3 positions on TinyShakespeare (char, d_model=64, 2 layers, 20 epochs). All 108 completed successfully.

```bash
python run.py sweep -c configs/sweep_shakespeare.yaml
```

### Top 30

| Rank | Attention | FFN | Position | PPL | Params |
|------|-----------|-----|----------|-----|--------|
| 1 | mamba | moe | none | **9.76** | 264K |
| 2 | mamba | swiglu | none | 9.85 | 180K |
| 3 | mamba | gated | none | 10.00 | 180K |
| 4 | mamba | standard | none | 10.01 | 164K |
| 5 | mamba | moe | rope | 10.60 | 264K |
| 6 | mamba | gated | rope | 10.71 | 180K |
| 7 | mamba | standard | rope | 10.88 | 164K |
| 8 | mamba | swiglu | rope | 10.94 | 180K |
| 9 | mamba | gated | sinusoidal | 12.87 | 180K |
| 10 | mamba | swiglu | sinusoidal | 12.91 | 180K |
| 11 | mamba | moe | sinusoidal | 12.97 | 264K |
| 12 | mamba | standard | sinusoidal | 13.02 | 164K |
| 13 | global_local | moe | none | 15.21 | 176K |
| 14 | dilated | swiglu | none | 15.28 | 91K |
| 15 | dilated | moe | none | 15.29 | 175K |
| 16 | dilated | gated | none | 15.30 | 91K |
| 17 | global_local | standard | none | 15.36 | 76K |
| 18 | global_local | gated | none | 15.41 | 92K |
| 19 | global_local | swiglu | none | 15.45 | 92K |
| 20 | window | moe | none | 15.48 | 175K |
| 21 | dilated | standard | none | 15.59 | 75K |
| 22 | window | swiglu | none | 15.60 | 91K |
| 23 | dilated | moe | rope | 15.61 | 175K |
| 24 | window | standard | none | 15.64 | 75K |
| 25 | ssm | moe | none | 15.72 | 171K |
| 26 | ssm | swiglu | none | 15.74 | 87K |
| 27 | ssm | gated | none | 15.78 | 87K |
| 28 | gqa | swiglu | none | 15.79 | 79K |
| 29 | mha | moe | none | 15.80 | 175K |
| 30 | mqa | swiglu | none | 15.83 | 79K |

**Mamba + position=none** domina el top-12 completo.

---

## Documentation

| File | Contents |
|------|----------|
| [Architecture](docs/architecture.md) | Model, TransformerBlock, Embedding, Config system, Weight init, CLI |
| [Attention](docs/attention.md) | All 9 mechanisms: MHA, MQA, GQA, Linear, Window, Dilated, Global-Local, Mamba, SSM |
| [FFN](docs/ffn.md) | All 4 variants: Standard, SwiGLU, Gated, Mixture-of-Experts |
| [Position](docs/position.md) | Sinusoidal, RoPE, None |
| [Datasets](docs/datasets.md) | Legacy datasets, tokenizers, chunk caching |
| [Training](docs/training.md) | Training loop, losses, schedulers, TensorBoard, generation, sweep, checkpoint adaptation |
| [Pre-training](docs/pretraining.md) | Pre-training from scratch with large corpus |
| [SFT](docs/sft.md) | Supervised fine-tuning with instruction data |
| [DPO](docs/dpo.md) | Direct Preference Optimization alignment |
| [PPO](docs/ppo.md) | PPO alignment with reward model |
| [GRPO](docs/grpo.md) | GRPO reasoning fine-tuning |
| [Datasets Catalog](docs/datasets_new.md) | All open-source datasets with auto-download |
| [Training Pipeline](docs/training_pipeline.md) | Full LLM training pipeline overview |
| [Future Work](docs/future_work.md) | Features not yet implemented |

## Regularization

| Technique | Param | Default |
|-----------|-------|---------|
| Stochastic Depth | `model.stochastic_depth` | 0.0 |
| Attention Dropout | `model.attention_dropout` | = `dropout` |
| FFN Dropout | `model.ffn_dropout` | = `dropout` |
| Embedding Dropout | `model.embedding_dropout` | = `dropout` |
| Weight Decay | `training.weight_decay` | 0.01 |
| Label Smoothing | `training.loss.label_smoothing` | 0.025 |
| Early Stopping | `training.early_stop_patience` | 0 |
| Gradient Clipping | `training.grad_clip` | 1.0 |

## Config YAML

See [configs/pretrain_tinystories.yaml](configs/pretrain_tinystories.yaml) or any file in [`configs/`](configs/). All fields overridable via CLI.

## Adding Components

```python
@register_attention("my_attn")
class MyAttention(nn.Module):
    def forward(self, x, mask=None): ...

@register_ffn("my_ffn")
class MyFFN(nn.Module):
    def forward(self, x): ...
```

Available immediately via YAML and CLI.
