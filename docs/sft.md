# Supervised Fine-Tuning (SFT)

Ajuste fino supervisado con datos de instrucciones.

## CLI

```bash
python run.py sft -c configs/sft_alpaca.yaml -m pretrained_model.pt
```

## Datasets disponibles

| Dataset | Ejemplos | Formato | Auto-download |
|---------|----------|---------|--------------|
| `alpaca_cleaned` | 52K | instruction/input/output | Sí (HF) |
| `oasst1` | 88K | conversaciones multi-turno | Sí (HF) |
| `dolly` | 15K | instruction/context/response | Sí (HF) |

## Config

```yaml
model:
  attention:
    type: mha
    num_heads: 4
  ffn:
    type: swiglu
    d_ff: 768
  position:
    type: none
sft:
  enabled: true
  loss_on_response_only: true
  packing: true
  logging: tensorboard
  log_interval: 10
  gradient_accumulation_steps: 2
training:
  batch_size: 8
  learning_rate: 0.00002
  scheduler: warmup_cosine
```

## Características

- Arquitectura **MHA + SwiGLU** (rápida y estable en memoria)
- **Loss masking**: solo entrena sobre tokens de respuesta del asistente
- **Formatos**: Alpaca, OASST, Dolly, ShareGPT
- **Chat template**: formateo automático de conversaciones
- **Packing**: múltiples ejemplos por secuencia
