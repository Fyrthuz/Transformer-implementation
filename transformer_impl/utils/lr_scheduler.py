import math
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def _warmup_cosine_schedule(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def _warmup_constant_schedule(warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda


def _warmup_linear_schedule(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 1.0 - progress
    return lr_lambda


def get_scheduler(name, optimizer, warmup_steps=0, total_steps=None, num_epochs=None, steps_per_epoch=None):
    if total_steps is None and num_epochs is not None and steps_per_epoch is not None:
        total_steps = num_epochs * steps_per_epoch
    if name == "warmup_cosine":
        total = total_steps or 100000
        return LambdaLR(optimizer, _warmup_cosine_schedule(warmup_steps, total))
    elif name == "warmup_constant":
        return LambdaLR(optimizer, _warmup_constant_schedule(warmup_steps))
    elif name == "warmup_linear":
        total = total_steps or 100000
        return LambdaLR(optimizer, _warmup_linear_schedule(warmup_steps, total))
    elif name == "cosine":
        total = total_steps or 100000
        return CosineAnnealingLR(optimizer, T_max=total)
    elif name == "linear":
        total = total_steps or 100000
        return LambdaLR(optimizer, lambda s: 1.0 - s / total)
    else:
        return LambdaLR(optimizer, lambda s: 1.0)
