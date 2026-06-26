import os
from . import register_dataset
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput


@register_dataset("hh_rlhf")
class HHRlhfPreparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use HH-RLHF dataset")

        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        tokenizer = self._get_tokenizer(cfg, dataset)

        max_seq_len = cfg.get('max_seq_len', 512)
        max_train = cfg.get('max_train_chunks', 35000)

        examples = []
        for item in dataset:
            chosen = item['chosen']
            rejected = item['rejected']
            chosen_tokens = tokenizer.encode(chosen)
            rejected_tokens = tokenizer.encode(rejected)
            if len(chosen_tokens) <= max_seq_len and len(rejected_tokens) <= max_seq_len:
                prompt = self._extract_prompt(chosen, rejected)
                prompt_tokens = tokenizer.encode(prompt)
                examples.append({
                    'prompt': prompt_tokens[:max_seq_len],
                    'chosen': chosen_tokens[:max_seq_len],
                    'rejected': rejected_tokens[:max_seq_len],
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

    def _extract_prompt(self, chosen, rejected):
        import re
        match = re.match(r'(.*?)(Assistant:|Human:)(.*)', chosen, re.DOTALL)
        if match:
            return match.group(1).strip()
        return chosen[:len(chosen)//3]

    def _get_tokenizer(self, cfg, dataset):
        from transformer_impl.datasets.tokenizer import train_bpe_tokenizer
        cache_dir = cfg.get('cache_dir', './data')
        texts = [d['chosen'] + " " + d['rejected'] for d in dataset]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'hh_rlhf_bpe.json'))
