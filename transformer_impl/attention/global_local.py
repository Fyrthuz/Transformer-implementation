import torch
import math
from . import register_attention

@register_attention("global_local")
class GlobalLocalAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, window_size=64, dropout=0.1, scale_attention=True, num_global_tokens=8, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.scale_attention = scale_attention
        self.w_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.w_o = torch.nn.Linear(d_model, d_model, bias=False)
        self.global_tokens = torch.nn.Parameter(torch.randn(num_global_tokens, d_model) * 0.02)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        global_tok = self.global_tokens.unsqueeze(0).expand(B, -1, -1)

        x_cat = torch.cat([global_tok, x], dim=1)
        G = self.num_global_tokens
        total_T = G + T

        q_all = self.w_q(x_cat).view(B, total_T, self.num_heads, self.d_k).transpose(1, 2)
        k_all = self.w_k(x_cat).view(B, total_T, self.num_heads, self.d_k).transpose(1, 2)
        v_all = self.w_v(x_cat).view(B, total_T, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q_all, k_all.transpose(-2, -1))
        if self.scale_attention:
            scores = scores / math.sqrt(self.d_k)

        full_mask = torch.ones(total_T, total_T, device=x.device, dtype=torch.bool)
        half_w = self.window_size // 2

        for i in range(total_T):
            if i < G:
                for j in range(total_T):
                    full_mask[i, j] = False
            else:
                local_idx = i - G
                start = max(G, i - half_w)
                end = min(total_T, i + half_w + 1)
                for j in range(total_T):
                    if j < G or (start <= j < end):
                        full_mask[i, j] = False

        scores = scores.masked_fill(full_mask[None, None, :, :], float('-inf'))

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            extended_mask = torch.ones(B, 1, total_T, total_T, device=x.device, dtype=torch.bool)
            extended_mask[:, :, G:, G:] = mask.bool()
            scores = scores.masked_fill(extended_mask == 0, float('-inf'))

        attn = torch.dropout(torch.softmax(scores, dim=-1), self.dropout.p, train=self.training)
        context = torch.matmul(attn, v_all)
        context = context.transpose(1, 2).contiguous().view(B, total_T, D)
        context = context[:, G:, :]
        return self.w_o(context)
