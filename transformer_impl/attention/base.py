import torch

class BaseAttention(torch.nn.Module):
    def forward(self, x, mask=None, **kwargs):
        raise NotImplementedError
