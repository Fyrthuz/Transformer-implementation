import torch
import torch.nn.functional as F
from . import register_ffn

@register_ffn("gated")
class GatedFFN(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff, bias=False)
        self.w2 = torch.nn.Linear(d_ff, d_model, bias=False)
        self.w_gate = torch.nn.Linear(d_model, d_ff, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.sigmoid(self.w_gate(x))
        hidden = F.gelu(self.w1(x))
        return self.dropout(self.w2(hidden * gate))
