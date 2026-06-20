import torch
from transformer_impl.attention import get_attention
from transformer_impl.ffn import get_ffn
from transformer_impl.config import ModelConfig

class TransformerBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0, num_layers: int = 1):
        super().__init__()
        self.layer_idx = layer_idx

        attn_drop = config.attention_dropout if config.attention_dropout is not None else config.dropout
        ffn_drop = config.ffn_dropout if config.ffn_dropout is not None else config.dropout

        cfg = config.attention
        attn_kwargs = {
            "d_model": config.d_model,
            "num_heads": cfg.num_heads,
            "dropout": attn_drop,
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
            'dropout': ffn_drop,
        }
        if config.ffn.type == 'moe':
            ffn_kwargs['num_experts'] = config.ffn.num_experts
            ffn_kwargs['top_k'] = config.ffn.top_k
        elif config.ffn.type == 'standard':
            ffn_kwargs['activation'] = config.ffn.activation
        self.ffn = get_ffn(config.ffn.type, **ffn_kwargs)
        self.norm1 = torch.nn.LayerNorm(config.d_model)
        self.norm2 = torch.nn.LayerNorm(config.d_model)

        self.stochastic_depth = config.stochastic_depth > 0.0
        if self.stochastic_depth:
            end = config.stochastic_depth
            start = end / 2.0
            self.drop_prob = start + (end - start) * layer_idx / max(num_layers - 1, 1)
            self.keep_prob = 1.0 - self.drop_prob

    def forward(self, x, mask=None):
        if self.stochastic_depth and self.training:
            if torch.rand(1).item() < self.drop_prob:
                return x
            attn_out = self.attention(self.norm1(x), mask)
            x = x + attn_out / self.keep_prob
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out / self.keep_prob
        else:
            x = x + self.attention(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
        return x

    def auxiliary_losses(self):
        if hasattr(self.ffn, 'auxiliary_losses'):
            return self.ffn.auxiliary_losses()
        return []
