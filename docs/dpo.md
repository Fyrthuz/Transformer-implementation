# Direct Preference Optimization (DPO)

Alineación del modelo usando pares de respuestas preferidas/rechazadas sin reward model separado.

## CLI

```bash
python run.py dpo -c configs/dpo_ultrafeedback.yaml -m sft_model.pt
```

## Datasets disponibles

| Dataset | Pares | Auto-download |
|---------|-------|--------------|
| `ultrafeedback` | 61K | Sí (HF) |
| `hh_rlhf` | 170K | Sí (HF) |

## Loss

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    pi_diff = policy_chosen_logps - policy_rejected_logps
    ref_diff = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_diff - ref_diff)
    return -F.logsigmoid(logits).mean()
```

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
dpo:
  enabled: true
  beta: 0.1
  logging: tensorboard
  log_interval: 10
  gradient_accumulation_steps: 2
training:
  batch_size: 4
  learning_rate: 0.000001
  scheduler: warmup_cosine
```

## Características

- Reference model congelado (copia del policy)
- Log-probabilidades para chosen y rejected
- Beta controla la fuerza de la preferencia
- Gradient accumulation (2 steps)
