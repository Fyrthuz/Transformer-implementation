import torch
import torch.nn.functional as F
from . import register_ffn

@register_ffn("swiglu")
class SwiGLUFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)
        self.w3 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
