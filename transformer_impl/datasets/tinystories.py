from .hf_dataset import HuggingFaceDatasetPreparer
from . import register_dataset


@register_dataset("tinystories")
class TinyStoriesPreparer(HuggingFaceDatasetPreparer):
    HF_PATH = "roneneldan/TinyStories"
    TEXT_FIELD = "text"
