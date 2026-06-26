import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    pi_diff = policy_chosen_logps - policy_rejected_logps
    ref_diff = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_diff - ref_diff)
    return -F.logsigmoid(logits).mean()
