import torch
from transformer_impl.attention import get_attention
from transformer_impl.ffn import get_ffn
from transformer_impl.config import ModelConfig

class TransformerBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        cfg = config.attention
        attn_kwargs = {
            "d_model": config.d_model,
            "num_heads": cfg.num_heads,
            "dropout": config.dropout,
        }
        if cfg.num_kv_heads is not None:
            attn_kwargs["num_kv_heads"] = cfg.num_kv_heads
        if cfg.window_size is not None:
            attn_kwargs["window_size"] = cfg.window_size
        if cfg.dilation is not None:
            attn_kwargs["dilation"] = cfg.dilation
        if cfg.d_state is not None:
            attn_kwargs["d_state"] = cfg.d_state
        if cfg.expand_factor is not None:
            attn_kwargs["expand_factor"] = cfg.expand_factor
        if cfg.d_conv is not None:
            attn_kwargs["d_conv"] = cfg.d_conv

        self.attention = get_attention(cfg.type, **attn_kwargs)
        ffn_kwargs = {
            'd_model': config.d_model,
            'd_ff': config.ffn.d_ff,
            'dropout': config.dropout,
        }
        if config.ffn.type == 'moe':
            ffn_kwargs['num_experts'] = config.ffn.num_experts
            ffn_kwargs['top_k'] = config.ffn.top_k
        elif config.ffn.type == 'standard':
            ffn_kwargs['activation'] = config.ffn.activation
        self.ffn = get_ffn(config.ffn.type, **ffn_kwargs)
        self.norm1 = torch.nn.LayerNorm(config.d_model)
        self.norm2 = torch.nn.LayerNorm(config.d_model)

    def forward(self, x, mask=None):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def auxiliary_losses(self):
        if hasattr(self.ffn, 'auxiliary_losses'):
            return self.ffn.auxiliary_losses()
        return []
