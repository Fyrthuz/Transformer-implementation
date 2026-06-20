# Transformer Implementation — Documentation

This project implements a modular Transformer from scratch in PyTorch, supporting 9 attention mechanisms, 4 FFN variants, and 3 positional encoding strategies — all interchangeable via a YAML config system.

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

## Quick Reference

**Train:**
```bash
./run_experiment.sh train -c configs/mamba_swiglu_none.yaml
```

**Sweep (grid search):**
```bash
./run_experiment.sh sweep -c configs/sweep_shakespeare.yaml
```

**Generate:**
```bash
./run_experiment.sh generate -m best_model.pt -p "ROMEO:"
```

**List available components:**
```bash
./run_experiment.sh list
```

## Key Design Decisions

- **Registry pattern**: Attention, FFN, position, and dataset components are self-registering via decorators (`@register_attention("mha")`), making them discoverable by string name.
- **Config-driven**: All hyperparameters live in YAML configs. CLI overrides (`model.attention.type=mamba`) let you modify any field without editing files.
- **Stochastic depth**: Each layer can be randomly dropped during training, with survival probability linearly decreasing from top to bottom layers.
- **Per-component dropout**: Attention dropout, FFN dropout, and embedding dropout can be set independently; if unset, they fall back to the global `dropout` value.
- **Dataset caching**: Tokenized chunks are cached as `.pt` tensor files for ~40x faster reload on repeated runs.

## Winning Configuration

On TinyShakespeare (character-level), the best combination across all 108 sweep combos is:

| Component | Choice | Test Perplexity |
|-----------|--------|----------------|
| Attention | mamba | 6.09 |
| FFN | swiglu | |
| Position | none | |
| Params | 180K | |

Config file: `configs/mamba_swiglu_none.yaml`
