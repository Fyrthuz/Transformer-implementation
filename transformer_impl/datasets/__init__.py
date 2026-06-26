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

from .tinystories import TinyStoriesPreparer
from .fineweb import FineWebPreparer
from .alpaca_sft import AlpacaCleanedPreparer
from .oasst_sft import OASST1Preparer
from .dolly_sft import DollyPreparer
from .hh_rlhf import HHRlhfPreparer
from .ultrafeedback import UltraFeedbackPreparer
from .gsm8k import GSM8KPreparer
from .math_dataset import MATHPreparer
from .mbpp import MBPPPreparer
