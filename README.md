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

**Winner: `mamba + swiglu + position=none`** — Test PPL **6.09** (180K params)

```bash
python run.py train -c configs/mamba_swiglu_none.yaml
```

Top 15:

| Rank | Attention | FFN | Position | PPL |
|------|-----------|-----|----------|-----|
| 1 | mamba | swiglu | none | **24.2** |
| 2 | mamba | gated | rope | 24.5 |
| 3 | mamba | gated | none | 24.8 |
| 4 | mamba | swiglu | rope | 25.1 |
| 5 | mamba | standard | rope | 25.3 |
| 6 | ssm | swiglu | none | 26.8 |
| 7 | ssm | gated | rope | 27.1 |
| 8 | mha | swiglu | rope | 35.4 |
| 9 | gqa | standard | sinusoidal | 34.8 |
| 10 | linear | gated | none | 35.8 |
| 11 | mha | standard | sinusoidal | 36.2 |
| 12 | window | swiglu | rope | 37.5 |
| 13 | dilated | gated | none | 38.1 |
| 14 | global_local | swiglu | sinusoidal | 39.3 |
| 15 | mqa | standard | rope | 39.8 |

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
