import torch

class BasePosition(torch.nn.Module):
    def forward(self, x):
        return x
