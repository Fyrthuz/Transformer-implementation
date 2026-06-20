ATTENTION_REGISTRY = {}

def register_attention(name):
    def decorator(cls):
        ATTENTION_REGISTRY[name] = cls
        return cls
    return decorator

def get_attention(name, **kwargs):
    if name not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention: {name}. Available: {list(ATTENTION_REGISTRY.keys())}")
    return ATTENTION_REGISTRY[name](**kwargs)

from .mha import MultiHeadAttention
from .mqa import MultiQueryAttention
from .gqa import GroupedQueryAttention
from .linear_attn import LinearAttention
from .window_attn import WindowAttention
from .dilated_attn import DilatedAttention
from .global_local import GlobalLocalAttention
from .mamba import MambaBlock
from .ssm import SSMBlock
