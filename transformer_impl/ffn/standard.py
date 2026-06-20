import torch
from . import register_ffn

def _get_activation(name):
    if name == "gelu":
        return torch.nn.GELU()
    elif name == "relu":
        return torch.nn.ReLU()
    elif name == "silu":
        return torch.nn.SiLU()
    else:
        return torch.nn.GELU()

@register_ffn("standard")
class StandardFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff, activation="gelu", dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            _get_activation(activation),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
