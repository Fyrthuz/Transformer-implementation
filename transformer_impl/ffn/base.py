import torch

class BaseFFN(torch.nn.Module):
    def forward(self, x):
        raise NotImplementedError

    def auxiliary_losses(self):
        return []
