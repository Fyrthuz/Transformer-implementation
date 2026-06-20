# Architecture

## Model

`transformer_impl/model.py` — `Transformer`

The top-level model, receiving token IDs and producing logits over the vocabulary:

```
x = Embedding(x)
for each layer:
    x = TransformerBlock(x, mask)
x = LayerNorm(x)
return Linear(x)          # → vocab_size logits
```

Key features:
- `auxiliary_losses()` collects per-layer auxiliary losses (e.g., MoE load-balancing loss) for the training loop.
- `generate_causal_mask(seq_len, device)` creates a standard triangular causal mask.

## TransformerBlock

`transformer_impl/blocks/transformer_block.py` — `TransformerBlock`

A pre-norm residual block with configurable attention and FFN:

```python
x = x + Attention(Norm1(x), mask)
x = x + FFN(Norm2(x))
```

**Stochastic depth integration:**
```python
if stochastic_depth and training and rand < drop_prob:
    return x  # skip entire block
x = x + Attention(Norm1(x)) / keep_prob
x = x + FFN(Norm2(x)) / keep_prob
```

- `drop_prob` interpolates linearly from `start = stoch_depth / 2` (layer 0) to `end = stoch_depth` (last layer).
- `keep_prob = 1 - drop_prob` scales residual outputs to preserve expected magnitude.
- During evaluation, stochastic depth is disabled entirely.

**Auxiliary losses:** If the FFN module exposes `auxiliary_losses()`, those losses are collected and returned.

## Embedding

`transformer_impl/embedding.py` — `TransformerEmbedding`

```python
def forward(self, x):
    return self.position_encoding(self.token_embedding(x))
```

- `token_embedding`: `nn.Embedding(vocab_size, d_model)`.
- `position_encoding`: one of sinusoidal, RoPE, or none (identity), selected by config string.
- Embedding dropout defaults to the global `model.dropout` unless `model.embedding_dropout` is explicitly set.

## Config System

`transformer_impl/config.py`

Six dataclasses nested under `ExperimentConfig`:

```
ExperimentConfig
├── name: str
├── seed: int
├── model: ModelConfig
│   ├── d_model, num_layers, dropout
│   ├── attention_dropout / ffn_dropout / embedding_dropout (optional per-component)
│   ├── stochastic_depth
│   ├── attention: AttentionConfig (type, num_heads, num_kv_heads, window_size, dilation, d_state, expand_factor, d_conv)
│   ├── ffn: FFNConfig (type, d_ff, activation, num_experts, top_k)
│   └── position: PositionConfig (type, max_len, rope_theta)
├── dataset: DatasetConfig (name, path, tokenization, vocab_size, max_seq_len, train_stride, cache_dir, max_train/test_chunks)
└── training: TrainConfig (batch_size, num_epochs, learning_rate, weight_decay, grad_clip, scheduler, early_stop_patience, loss)
    └── loss: LossConfig (type, label_smoothing, moe_load_balance_coef, focal_gamma)
```

**Loading configs:**

```python
# From YAML only:
cfg = config_from_yaml("configs/mha_rope.yaml")

# From YAML + CLI overrides:
cfg = config_from_cli("configs/mha_rope.yaml", {"model.attention.type": "gqa"})
```

`deep_merge` recursively merges override dicts into base dicts. `parse_cli_overrides` converts `model.attention.type=gqa` strings into nested dicts.

`flatten_config` / `flatten_dict` produce flat key-value maps for logging hyperparameters to TensorBoard.

## Weight Initialization

`run.py:initialize_weights`

- `nn.Linear`: Xavier uniform, zero biases.
- `nn.LayerNorm`: weight=1, bias=0.
- `nn.Embedding`: Normal(0, d_model^-0.5) — scaled inversely with model dimension.

## CLI Entry Points

`run.py` provides four subcommands:

| Command | Description |
|---------|-------------|
| `train -c <yaml> [overrides...]` | Train a single model |
| `generate -c <yaml> -m <model.pt> -p <prompt>` | Generate text |
| `sweep -c <sweep.yaml>` | Grid search over hyperparameter combinations |
| `list` | List all registered components |

`run_experiment.sh` wraps `run.py` and automatically starts TensorBoard on port 6006 before training/sweep commands. Set `NO_TENSORBOARD=1` to skip.
