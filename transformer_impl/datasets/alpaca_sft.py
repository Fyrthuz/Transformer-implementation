import os
import json
import torch
from . import register_dataset
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput


ALPACA_TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"


@register_dataset("alpaca_cleaned")
class AlpacaCleanedPreparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use Alpaca dataset")

        dataset = load_dataset("yahma/alpaca-cleaned", split="train")
        tokenizer = self._get_tokenizer(cfg, dataset)

        max_seq_len = cfg.get('max_seq_len', 512)
        max_train = cfg.get('max_train_chunks', 52000)

        examples = []
        for item in dataset:
            instruction = ALPACA_TEMPLATE.format(
                instruction=item['instruction'],
                input=item.get('input', '') or '',
                response='',
            )
            response_text = item['output']
            instr_tokens = tokenizer.encode(instruction)
            resp_tokens = tokenizer.encode(response_text)
            tokens = instr_tokens + resp_tokens
            if len(tokens) <= max_seq_len:
                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                labels = ([-100] * len(instr_tokens) + resp_tokens +
                          [-100] * (max_seq_len - len(tokens)))
                examples.append({
                    'input_ids': padded,
                    'labels': labels[:max_seq_len],
                    'attention_mask': [1] * len(tokens) + [0] * (max_seq_len - len(tokens)),
                })

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
        texts = [f"{d['instruction']} {d.get('input','')} {d['output']}" for d in dataset]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'alpaca_cleaned_bpe.json'))
