# Pre-training

Entrenamiento desde cero con corpus grande usando next-token prediction.

## CLI

```bash
python run.py pretrain -c configs/pretrain_tinystories.yaml
```

## Datasets disponibles

| Dataset | Tokens | Auto-download |
|---------|--------|--------------|
| `tinystories` | ~500M | Sí (HF) |
| `fineweb` | ~10B (sample) | Sí (HF) |
| `wikitext2` | ~2M | Sí (GitHub) |
| `tinyshakespeare` | ~1M | Sí (GitHub) |

## Config

```yaml
model:
  d_model: 256
  num_layers: 6
  attention:
    type: mha
    num_heads: 4
  ffn:
    type: swiglu
    d_ff: 768
  position:
    type: none
pretrain:
  enabled: true
  warmup_steps: 1000
  max_steps: 50000
  gradient_accumulation_steps: 4
  mixed_precision: bf16
  logging: tensorboard
  save_steps: 1000
  eval_steps: 500
training:
  batch_size: 16
  learning_rate: 0.0003
  scheduler: warmup_cosine
```

## Características

- Arquitectura **MHA + SwiGLU + sin posicional**
- Gradient accumulation (4 steps)
- AMP (mixed precision fp16/bf16)
- Warmup + cosine LR scheduler
- Checkpointing automático cada N steps
- Evaluación periódica de perplexity
- TensorBoard / WandB logging
- Resume desde checkpoint
