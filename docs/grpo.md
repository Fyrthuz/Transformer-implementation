# Group Relative Policy Optimization (GRPO)

Fine-tuning de razonamiento sin critic network. Usa grupos de respuestas para calcular ventajas relativas.

## CLI

```bash
python run.py grpo -c configs/grpo_gsm8k.yaml -m sft_model.pt
```

## Datasets disponibles

| Dataset | Ejemplos | Tipo | Auto-download |
|---------|----------|------|--------------|
| `gsm8k` | 8.5K | Matemáticas escolares | Sí (HF) |
| `math` | 12.5K | Matemáticas competitivas | Sí (HF) |
| `mbpp` | 974 | Programación Python | Sí (HF) |

## Loss

```python
def grpo_loss(policy_logps, ref_logps, rewards, epsilon=0.2, kl_coef=0.01):
    adv_std = rewards.std(dim=-1, keepdim=True)
    adv_mean = rewards.mean(dim=-1, keepdim=True)
    advantages = torch.where(
        adv_std > 1e-6,
        (rewards - adv_mean) / (adv_std + 1e-8),
        torch.zeros_like(rewards),
    )
    ratio = torch.exp(policy_logps - ref_logps.detach())
    ratio = torch.clamp(ratio, max=100.0)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    pg_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    kl_loss = (torch.exp(ref_logps - policy_logps) -
               (ref_logps - policy_logps) - 1).mean()
    return pg_loss + kl_coef * kl_loss
```

## Reward Functions

| Reward | Descripción |
|--------|-------------|
| `exact_match` | Compara respuesta extraída con ground truth |
| `format` | Verifica formato CoT con `<thinking>` tags |

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
grpo:
  enabled: true
  group_size: 4
  epsilon: 0.2
  kl_coef: 0.01
  max_gen_length: 1024
  reward_fn: exact_match
  logging: tensorboard
  log_interval: 10
  gradient_accumulation_steps: 2
training:
  batch_size: 4
  learning_rate: 0.000001
  scheduler: warmup_cosine
```

## Características

- Group-based advantage (sin critic network, como en DeepSeek-R1)
- Generación de G respuestas por prompt
- Recompensas basadas en verificación (exact_match, formato, código)
- KL penalty contra modelo de referencia
- Gradient accumulation (2 steps)
