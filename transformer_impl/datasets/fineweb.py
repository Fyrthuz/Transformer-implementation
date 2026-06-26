from .hf_dataset import HuggingFaceDatasetPreparer
from . import register_dataset


@register_dataset("fineweb")
class FineWebPreparer(HuggingFaceDatasetPreparer):
    HF_PATH = "HuggingFaceFW/fineweb"
    HF_SPLIT = "train"
    TEXT_FIELD = "text"
