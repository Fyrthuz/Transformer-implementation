# Transformer Implementation — Documentation

This project implements a modular Transformer from scratch in PyTorch, supporting 9 attention mechanisms, 4 FFN variants, and 3 positional encoding strategies, plus a **complete LLM training pipeline** (pre-training, SFT, DPO, PPO, GRPO).

## Architecture Overview

```
Input tokens → Embedding (token + position) → N × TransformerBlock → LayerNorm → Linear (vocab)
```

Each `TransformerBlock` contains:
- **Attention** (one of 9 types)
- **FFN** (one of 4 types)
- Pre-norm LayerNorm + residual connections
- Optional stochastic depth

All components are registered in registries and selected by string name from config.

## Component Documentation

| Doc | Contents |
|-----|----------|
| [Architecture](architecture.md) | Model, TransformerBlock, Embedding, Config system, Weight init, CLI |
| [Attention](attention.md) | All 9 attention mechanisms: MHA, MQA, GQA, Linear, Window, Dilated, Global-Local, Mamba, SSM |
| [FFN](ffn.md) | All 4 FFN variants: Standard, SwiGLU, Gated, Mixture-of-Experts |
| [Position](position.md) | All 3 position encodings: Sinusoidal, RoPE, None |
| [Datasets](datasets.md) | Datasets, tokenizers, chunk caching, train/test splitting |
| [Training](training.md) | Training loop, loss functions, schedulers, TensorBoard, generation |
| [Pre-training](pretraining.md) | Pre-training from scratch with large corpus |
| [SFT](sft.md) | Supervised fine-tuning with instruction data |
| [DPO](dpo.md) | Direct Preference Optimization alignment |
| [PPO](ppo.md) | PPO alignment with reward model |
| [GRPO](grpo.md) | GRPO reasoning fine-tuning |
| [Datasets Catalog](datasets_new.md) | All open-source datasets with auto-download |
| [Training Pipeline](training_pipeline.md) | Full LLM training pipeline + pipeline command |
| [Future Work](future_work.md) | Features not yet implemented |

## Quick Reference

**Full pipeline (3 etapas — recomendado):**
```bash
./run_experiment.sh pipeline -c configs/pipeline_sft_dpo.yaml
```

**Full pipeline (4 etapas + GRPO — lento):**
```bash
./run_experiment.sh pipeline -c configs/pipeline_full.yaml
```

**Pre-training:**
```bash
python run.py pretrain -c configs/pretrain_tinystories.yaml
```

**SFT:**
```bash
python run.py sft -c configs/sft_alpaca.yaml -m runs/pretrain_*/checkpoints/best_model.pt
```

**DPO:**
```bash
python run.py dpo -c configs/dpo_ultrafeedback.yaml -m runs/sft_*/checkpoints/best_model.pt
```

Los checkpoints se guardan en `runs/{run_name}/checkpoints/`. Usar pipeline para gestión automática.

**Train (original):**
```bash
./run_experiment.sh train -c configs/pretrain_tinystories.yaml
```

**Sweep (grid search):**
```bash
./run_experiment.sh sweep -c configs/sweep_shakespeare.yaml
```

**Generate:**
```bash
./run_experiment.sh generate -m best_model.pt -p "ROMEO:"
```

**Run tests:**
```bash
./run_experiment.sh test
./run_experiment.sh test -v -k dpo
```

**List available components:**
```bash
./run_experiment.sh list
```

**TensorBoard:**
```bash
tensorboard --logdir runs
```

## Key Design Decisions

- **Registry pattern**: Attention, FFN, position, and dataset components are self-registering via decorators (`@register_attention("mha")`), making them discoverable by string name.
- **Config-driven**: All hyperparameters live in YAML configs. CLI overrides (`model.ffn.type=swiglu`) let you modify any field without editing files.
- **Stochastic depth**: Each layer can be randomly dropped during training, with survival probability linearly decreasing from top to bottom layers.
- **Per-component dropout**: Attention dropout, FFN dropout, and embedding dropout can be set independently; if unset, they fall back to the global `dropout` value.
- **Dataset caching**: Tokenized chunks are cached as `.pt` tensor files for ~40x faster reload on repeated runs.
- **Multi-stage pipeline**: Ejecuta pretrain → SFT → DPO (opcional: + GRPO) con un solo comando, pasando checkpoints automáticamente.
- **Adaptive checkpoint loading**: Carga checkpoints entrenados con arquitectura diferente (vocab_size, num_layers, d_model), redimensionando o omitiendo capas incompatibles.

## Default Configuration

The pipeline uses **MHA + SwiGLU + none** (rápida, estable en memoria para GPUs con 16GB).

| Component | Choice |
|-----------|--------|
| Attention | mha (num_heads=4, d_model=256, 6 layers) |
| FFN | swiglu (d_ff=768) |
| Position | none |

Config file: `configs/pretrain_tinystories.yaml`
