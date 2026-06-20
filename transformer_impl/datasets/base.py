from dataclasses import dataclass

@dataclass
class DatasetOutput:
    train_data: list
    test_data: list
    vocab_size: int
    pad_token_id: int
    tokenizer: object

class BaseDatasetPreparer:
    def prepare(self, cfg) -> DatasetOutput:
        raise NotImplementedError
