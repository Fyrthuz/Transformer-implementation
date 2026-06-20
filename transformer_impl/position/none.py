import torch
from . import register_position

@register_position("none")
class NoPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x)
