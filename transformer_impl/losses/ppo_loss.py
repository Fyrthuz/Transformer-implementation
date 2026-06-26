import torch
import torch.nn.functional as F


def ppo_loss(log_probs, old_log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    ratio = torch.clamp(ratio, max=100.0)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


def ppo_loss_with_value(log_probs, old_log_probs, advantages, values, returns,
                        epsilon=0.2, vf_coef=0.5):
    policy_loss = ppo_loss(log_probs, old_log_probs, advantages, epsilon)
    value_loss = F.mse_loss(values.squeeze(-1), returns.detach())
    return policy_loss + vf_coef * value_loss
