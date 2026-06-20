import torch
import math
from . import register_position

@register_position("rope")
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, rope_theta=10000.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        cos = emb.cos()
        sin = emb.sin()
        return self.dropout(self._apply_rotary(x, cos, sin))

    def _apply_rotary(self, x, cos, sin):
        half = x.size(-1) // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos = cos[:, :x.size(1), :x.size(-1)]
        sin = sin[:, :x.size(1), :x.size(-1)]
        rotated = torch.cat([x1 * cos[..., :half] - x2 * sin[..., :half],
                             x1 * sin[..., :half] + x2 * cos[..., :half]], dim=-1)
        return rotated

@register_position("rope")
class RotaryEmbeddingSimple(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, rope_theta=10000.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
        self._cos = None
        self._sin = None

    def _build_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos()
        self._sin = emb.sin()

    def forward(self, x):
        seq_len = x.size(1)
        if self._cos is None or self._cos.size(0) < seq_len:
            self._build_cache(seq_len, x.device)
        cos = self._cos[:seq_len].unsqueeze(0)
        sin = self._sin[:seq_len].unsqueeze(0)
        return self.dropout(self._apply_rotary(x, cos, sin))

    def _apply_rotary(self, x, cos, sin):
        half = x.size(-1) // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        cos = cos[:, :x.size(1), :half]
        sin = sin[:, :x.size(1), :half]
        rotated = torch.cat([x1 * cos - x2 * sin,
                             x1 * sin + x2 * cos], dim=-1)
        return rotated
