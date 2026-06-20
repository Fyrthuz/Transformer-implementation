# Transformer Experiment

Modular **decoder-only Transformer** (GPT-style) with 9 attention mechanisms, 4 FFN variants, 3 positional encodings, CLI, and grid search with TensorBoard logging.

## Quick Start

```bash
uv sync                        # or: pip install -r requirements.txt
./run_experiment.sh list       # disponibles
./run_experiment.sh train -c configs/mamba_swiglu_none.yaml
./run_experiment.sh sweep -c configs/sweep_shakespeare.yaml   # 108 combos
```

## CLI

| Subcommand | Description |
|------------|-------------|
| `train -c <yaml> [overrides...]` | Train model |
| `generate -c <yaml> -m <model.pt> -p <prompt>` | Generate text |
| `sweep -c <sweep.yaml>` | Grid search |
| `list` | List components |

CLI overrides: `python run.py train model.attention.type=mamba model.ffn.type=swiglu`

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

**Mamba + position=none** domina el top-12 completo. El ganador claro es **mamba + moe/swiglu + none**.

---

## Documentation

| File | Contents |
|------|----------|
| [Architecture](docs/architecture.md) | Model, TransformerBlock, Embedding, Config system, Weight init, CLI |
| [Attention](docs/attention.md) | All 9 mechanisms: MHA, MQA, GQA, Linear, Window, Dilated, Global-Local, Mamba, SSM |
| [FFN](docs/ffn.md) | All 4 variants: Standard, SwiGLU, Gated, Mixture-of-Experts |
| [Position](docs/position.md) | Sinusoidal, RoPE, None |
| [Datasets](docs/datasets.md) | TinyShakespeare, WikiText-2/103, tokenizers, chunk caching |
| [Training](docs/training.md) | Training loop, losses, schedulers, TensorBoard, generation, sweep |

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

See [configs/mamba_swiglu_none.yaml](configs/mamba_swiglu_none.yaml) or any file in [`configs/`](configs/). All fields overridable via CLI.

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
