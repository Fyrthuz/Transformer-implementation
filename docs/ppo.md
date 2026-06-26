# PPO Alignment

Reinforcement Learning from Human Feedback usando Proximal Policy Optimization.

## CLI

```bash
python run.py ppo -c configs/ppo_anthropic.yaml -m sft_model.pt
```

## Arquitectura

- **Policy model**: el transformer con `add_value_head()` para estimar value function
- **Reference model**: copia congelada del policy para KL penalty
- **Reward model**: usa el mismo transformer con cabeza de regresión (score)

## Loss

```python
def ppo_loss_with_value(log_probs, old_log_probs, advantages, values, returns,
                        epsilon=0.2, vf_coef=0.5):
    ratio = torch.exp(log_probs - old_log_probs)
    ratio = torch.clamp(ratio, max=100.0)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
    value_loss = F.mse_loss(values.squeeze(-1), returns.detach())
    return policy_loss + vf_coef * value_loss
```

## Config

```yaml
ppo:
  enabled: true
  kl_coef: 0.02
  clip_range: 0.2
  vf_coef: 0.5
  max_gen_length: 256
training:
  learning_rate: 0.000001
```

## Características

- Generación de respuestas on-policy
- Value head para advantage estimation
- KL penalty contra modelo de referencia
- PPO clipping para estabilidad
