# Training

## Training Loop

**File:** `transformer_impl/train.py:train_model`

The training loop follows a standard supervised language modeling setup:

1. **Data**: `TextDataset` wraps each tokenized example, pads to `max_seq_len`, and yields via DataLoader.
2. **Forward**: Inputs = `batch[:, :-1]`, targets = `batch[:, 1:]`. Causal mask is generated and passed to the model.
3. **Loss**: `main_loss` (per-token cross-entropy over vocabulary) + `sum(auxiliary_losses) * moe_load_balance_coef`.
4. **Backward**: Gradient clipping at `grad_clip` norm.
5. **Scheduler**: Cosine annealing over `num_epochs` (or constant LR if not cosine).
6. **Evaluation**: Full test set pass after each epoch, no gradient computation.
7. **Checkpointing**: When test loss improves, `best_model.pt` is saved with `model_state_dict`, `config`, `vocab_size`, `test_loss`, and `test_perplexity`.
8. **Early stopping**: If `early_stop_patience > 0`, stops after that many epochs without test loss improvement.

## Loss Functions

Configured via `training.loss`:

| `loss.type` | Class | Details |
|-------------|-------|---------|
| `cross_entropy` | `nn.CrossEntropyLoss` | Default. Supports `label_smoothing`. |
| `nll` | `nn.NLLLoss` | Negative log-likelihood. |
| `mse` | `nn.MSELoss` | Mean squared error. |
| `focal` | Custom `FocalLoss` | `(1 - pt)^γ * CE` with configurable `focal_gamma`. |

`ignore_index` defaults to `pad_token_id` (set to `"auto"`). Padding tokens are excluded from loss computation.

### Label Smoothing

When `training.loss.label_smoothing > 0`, the cross-entropy loss replaces hard targets with a uniform mixture:

```
target = (1 - ε) * one_hot(y) + ε / vocab_size
```

Applied by `nn.CrossEntropyLoss` natively when `label_smoothing` is set.

## Optimizer & Scheduler

- **Optimizer**: `AdamW` with `learning_rate` and `weight_decay`.
- **Scheduler**: `CosineAnnealingLR` with `T_max = num_epochs`. Falls back to constant LR if `scheduler != "cosine"`.

## TensorBoard Logging

**File:** `train.py` — uses `torch.utils.tensorboard.SummaryWriter`

Per epoch:

| Scalar | Description |
|--------|-------------|
| `Train/Loss` | Average training loss |
| `Train/Perplexity` | exp(Train/Loss) |
| `Test/Loss` | Average test loss |
| `Test/Perplexity` | exp(Test/Loss) |
| `Params/Learning_Rate` | Current LR |
| `Time/Epoch_seconds` | Wall time per epoch |
| `Time/Inference_ms_per_sample` | Inference latency |

Per step:
| Scalar | Description |
|--------|-------------|
| `Train/Loss_step` | Loss at each optimizer step |

Per 5 epochs:
| Text | Description |
|------|-------------|
| `Samples/Prediction_Test` | Input, target, and predicted text for a test batch |

GPU metrics (when CUDA is available):
| Scalar | Description |
|--------|-------------|
| `GPU/Memory_allocated_GB` | Allocated memory |
| `GPU/Memory_reserved_GB` | Reserved memory |

At end of training:
| Scalar | Description |
|--------|-------------|
| `Time/Total_experiment_seconds` | Total wall time |
| `Time/Avg_inference_ms_per_sample` | Average inference time per sample |

Hyperparameter logging:
```python
writer.add_hparams(hparam_dict, {'hparam/test_loss': best_loss, 'hparam/test_perplexity': best_ppl})
```

The `runs/` directory contains all TensorBoard logs. Launch with:
```bash
tensorboard --logdir=runs --port=6006
```

## Run Naming

Each run creates a directory in `runs/` with a descriptive name:

```
{prefix}_{attention}_{ffn}_{position}_d{d_model}l{num_layers}_{timestamp}
```

Examples:

| Run | Directory |
|-----|-----------|
| `./run_experiment.sh pipeline` | `runs/pipeline_pretrain_mha_swiglu_none_d256l6_20260626_202617/` |
| `./run_experiment.sh pretrain` | `runs/pretrain_mha_swiglu_none_d256l6_20260626_202617/` |
| `./run_experiment.sh train` | `runs/train_mha_standard_rope_d256l6_20260626_202617/` |

Override with `name=custom_name` in CLI overrides:

```bash
python run.py pretrain -c configs/pretrain_tinystories.yaml name=mi_experimento
```

## Generation

**File:** `transformer_impl/generate.py:generate_text`

Autoregressive text generation with temperature sampling:

```python
for _ in range(max_chars):
    logits = model(context)[:, -1, :] / temperature
    probs = softmax(logits)
    next_id = multinomial(probs)
    context = concat(context, next_id)
```

- Context is truncated to the last 128 tokens if it exceeds `max_seq_len`.
- Temperature: lower values (e.g., 0.5) produce more deterministic output; higher values (e.g., 1.0) increase diversity.
- Token decoding uses the dataset's tokenizer (char-level for TinyShakespeare char, BPE otherwise).

## Fine-tuning con Checkpoint Adaptativo

**File:** `transformer_impl/utils/checkpointing.py:load_checkpoint_with_adaptation`

Al cargar un checkpoint pre-entrenado con un modelo de **arquitectura diferente**, el sistema adapta automáticamente los pesos compatibles:

| Cambio en la config | Comportamiento |
|---------------------|----------------|
| `vocab_size` diferente | **Redimensiona** embedding y output layer — copia pesos del vocabulario compartido, inicializa el resto aleatoriamente |
| `num_layers` diferente | **Carga capas comunes** — si el checkpoint tiene 6 capas y el nuevo modelo 8, carga las primeras 6, las 2 restantes se inicializan aleatoriamente |
| `d_model` diferente | **Omite** todas las capas con shape mismatch (embedding, atención, FFN, LayerNorm). Solo carga `output_layer.bias` y capas sin dependencia de `d_model` |
| `num_heads` diferente | **Omite** los pesos de atención (shape mismatch), preserva el resto |
| Misma arquitectura | **Carga completa** — todos los pesos coinciden |

Ejemplo — preentrenar con modelo pequeño y fine-tunear con uno más grande:

```bash
# 1. Pre-training con modelo pequeño (6 layers, d_model=256)
python run.py pretrain -c configs/pretrain_tinystories.yaml \
  model.num_layers=6 model.d_model=256

# 2. SFT con modelo grande, cargando checkpoint parcial
python run.py sft -c configs/sft_alpaca.yaml \
  model.num_layers=8 model.d_model=512 \
  -m best_model.pt
```

En este caso, el sistema reportará:
```
  Checkpoint: loaded 3 layers matched (ln_f, output_layer.bias)
  Checkpoint: resized 2 layers (vocab size differs)
  Checkpoint: skipped 24 mismatched layers (architecture differs)
```

Esto permite **transfer learning entre arquitecturas distintas** sin necesidad de reentrenar desde cero.

## Grid Search (Sweep)

**File:** `run.py:cmd_sweep`

Reads a sweep YAML config with `sweep` key containing parameter lists:

```yaml
sweep:
  name: shakespeare_example
  model.attention.type: [mha, mqa, gqa, linear, window, dilated, global_local, mamba, ssm]
  model.ffn.type: [standard, swiglu, gated, moe]
  model.position.type: [sinusoidal, rope, none]
```

- Generates all combinations via `itertools.product` (108 for the above grid).
- Each combination is trained independently and ranked by test perplexity.
- Top 30 configurations are printed as a table.
- Full results (including failures) are exported as a timestamped JSON file.

## Experiment Shell Script

**File:** `run_experiment.sh`

Wraps `run.py` with:
- Automatic TensorBoard launch on port 6006 (configurable via `TENSORBOARD_PORT`).
- Virtual environment detection (uses `.venv/bin/python3` if it exists).
- Cleanup of TensorBoard process on exit.

```bash
./run_experiment.sh pipeline -c configs/pipeline_full.yaml
./run_experiment.sh train -c configs/pretrain_tinystories.yaml
./run_experiment.sh sweep -c configs/sweep_shakespeare.yaml
./run_experiment.sh generate -m best_model.pt -p "ROMEO:" -n 500 -t 0.7
./run_experiment.sh test                  # run all tests
./run_experiment.sh test -v -k dpo        # run only dpo tests
./run_experiment.sh list
NO_TENSORBOARD=1 ./run_experiment.sh train -c configs/pretrain_tinystories.yaml
```
