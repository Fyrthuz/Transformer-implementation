import torch
import math
from . import register_attention

@register_attention("dilated")
class DilatedAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, window_size=64, dilation=4, dropout=0.1, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.dilation = dilation
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        dilate_mask = torch.ones(T, T, device=x.device, dtype=torch.bool)
        half_w = self.window_size // 2
        for i in range(T):
            start = max(0, i - half_w * self.dilation)
            end = min(T, i + half_w * self.dilation + 1)
            for j in range(start, end):
                if (j - i) % self.dilation == 0 or abs(j - i) <= 2:
                    dilate_mask[i, j] = False
        scores = scores.masked_fill(dilate_mask[None, None, :, :], float('-inf'))

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.dropout(torch.softmax(scores, dim=-1), self.dropout.p, train=self.training)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(context)
