import os
from . import register_dataset
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput


DOLLY_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"


@register_dataset("dolly")
class DollyPreparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use Dolly dataset")

        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        tokenizer = self._get_tokenizer(cfg, dataset)

        max_seq_len = cfg.get('max_seq_len', 512)
        max_train = cfg.get('max_train_chunks', 15000)

        examples = []
        for item in dataset:
            text = DOLLY_TEMPLATE.format(
                instruction=item['instruction'],
                context=item.get('context', '') or '',
                response=item['response'],
            )
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_seq_len:
                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                examples.append({'text': padded[:max_seq_len]})

        if max_train and len(examples) > max_train:
            examples = examples[:max_train]

        return DatasetOutput(
            train_data=examples,
            test_data=examples[:max(1, len(examples)//10)],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )

    def _get_tokenizer(self, cfg, dataset):
        from transformer_impl.datasets.tokenizer import train_bpe_tokenizer
        cache_dir = cfg.get('cache_dir', './data')
        texts = [f"{d['instruction']} {d.get('context','')} {d['response']}" for d in dataset]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'dolly_bpe.json'))
