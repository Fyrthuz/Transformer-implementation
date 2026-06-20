POSITION_REGISTRY = {}

def register_position(name):
    def decorator(cls):
        POSITION_REGISTRY[name] = cls
        return cls
    return decorator

def get_position(name, d_model, dropout=0.1, max_len=5000):
    if name not in POSITION_REGISTRY:
        raise ValueError(f"Unknown position encoding: {name}. Available: {list(POSITION_REGISTRY.keys())}")
    return POSITION_REGISTRY[name](d_model, dropout=dropout, max_len=max_len)

from .sinusoidal import SinusoidalEncoding
from .rope import RotaryEmbedding
from .none import NoPositionalEncoding
