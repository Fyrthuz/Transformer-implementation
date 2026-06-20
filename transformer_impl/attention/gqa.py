import torch
import math
from . import register_attention

@register_attention("gqa")
class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.1, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else max(1, num_heads // 4)
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.w_q = torch.nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.w_k = torch.nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = torch.nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        n_repeat = self.num_heads // self.num_kv_heads
        if n_repeat > 1:
            k = k[:, :, None, :, :].expand(B, self.num_kv_heads, n_repeat, T, self.d_k).reshape(B, self.num_heads, T, self.d_k)
            v = v[:, :, None, :, :].expand(B, self.num_kv_heads, n_repeat, T, self.d_k).reshape(B, self.num_heads, T, self.d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.dropout(torch.softmax(scores, dim=-1), self.dropout.p, train=self.training)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(context)
