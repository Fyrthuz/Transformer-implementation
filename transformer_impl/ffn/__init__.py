FFN_REGISTRY = {}

def register_ffn(name):
    def decorator(cls):
        FFN_REGISTRY[name] = cls
        return cls
    return decorator

def get_ffn(name, **kwargs):
    if name not in FFN_REGISTRY:
        raise ValueError(f"Unknown FFN: {name}. Available: {list(FFN_REGISTRY.keys())}")
    return FFN_REGISTRY[name](**kwargs)

from .standard import StandardFFN
from .swiglu import SwiGLUFFN
from .gated import GatedFFN
from .moe import MixtureOfExperts
