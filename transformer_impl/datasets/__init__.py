DATASET_REGISTRY = {}

def register_dataset(name):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset_preparer(name):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]

from .tokenizer import train_bpe_tokenizer, CharTokenizer, BPETokenizer
from .tinyshakespeare import TinyShakespearePreparer
from .wikitext import WikiText2Preparer, WikiText103Preparer
