import torch


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
    kl_diff = (ref_logps - policy_logps).clamp(min=-50, max=50)
    kl_loss = (torch.exp(kl_diff) - kl_diff - 1).mean()
    return pg_loss + kl_coef * kl_loss
