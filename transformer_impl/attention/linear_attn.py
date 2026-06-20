import torch
import math
from . import register_attention

def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1.0

@register_attention("linear")
class LinearAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        q = elu_feature_map(q)
        k = elu_feature_map(k)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # causal masking via diagonal masking in linear attention
            mask = mask.to(x.dtype)
            k = k * mask.unsqueeze(-1)
            q = q * mask.unsqueeze(-1)

        kv = torch.matmul(k.transpose(-2, -1), v)
        context = torch.matmul(q, kv)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(context)
