import os
from . import register_dataset
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput


@register_dataset("mbpp")
class MBPPPreparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use MBPP dataset")

        dataset = load_dataset("google-research-datasets/mbpp", split="train", trust_remote_code=True)
        tokenizer = self._get_tokenizer(cfg, dataset)

        max_seq_len = cfg.get('max_seq_len', 1024)
        max_train = cfg.get('max_train_chunks', 500)

        examples = []
        for item in dataset:
            prompt = item['prompt']
            code = item['code']
            tests = item.get('test_list', [])
            text = f"Task: {prompt}\nWrite Python code:\n"
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_seq_len:
                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                examples.append({
                    'text': padded[:max_seq_len],
                    'prompt': prompt,
                    'code': code,
                    'tests': tests,
                    'input_ids': padded[:max_seq_len],
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
        texts = [d['prompt'] + " " + d['code'] for d in dataset]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'mbpp_bpe.json'))
