import os
import json
import torch
from dataclasses import dataclass

@dataclass
class DatasetOutput:
    train_data: list
    test_data: list
    vocab_size: int
    pad_token_id: int
    tokenizer: object


def cache_chunks(path, train_chunks, test_chunks):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_t = torch.tensor(train_chunks, dtype=torch.long)
    test_t = torch.tensor(test_chunks, dtype=torch.long)
    torch.save({'train': train_t, 'test': test_t}, path)


def load_cached_chunks(path):
    data = torch.load(path, weights_only=True)
    return data['train'].tolist(), data['test'].tolist()


class BaseDatasetPreparer:
    def prepare(self, cfg) -> DatasetOutput:
        raise NotImplementedError
