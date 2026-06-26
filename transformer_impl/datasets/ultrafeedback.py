import os
from . import register_dataset
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput


@register_dataset("ultrafeedback")
class UltraFeedbackPreparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use UltraFeedback dataset")

        raw = list(load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train"))
        tokenizer = self._build_tokenizer(cfg, raw)

        max_seq_len = cfg.get('max_seq_len', 512)
        max_train = cfg.get('max_train_chunks', 35000)

        examples = []
        for item in raw:
            chosen = self._extract_text(item['chosen'])
            rejected = self._extract_text(item['rejected'])
            prompt = item.get('prompt', '')
            chosen_tokens = tokenizer.encode(chosen)
            rejected_tokens = tokenizer.encode(rejected)
            prompt_tokens = tokenizer.encode(prompt)
            if all(len(t) <= max_seq_len for t in [chosen_tokens, rejected_tokens, prompt_tokens]):
                examples.append({
                    'prompt': prompt_tokens,
                    'chosen': chosen_tokens,
                    'rejected': rejected_tokens,
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

    def _extract_text(self, messages):
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            return " ".join(m.get('content', '') for m in messages if isinstance(m, dict))
        return str(messages)

    def _build_tokenizer(self, cfg, raw):
        from transformer_impl.datasets.tokenizer import train_bpe_tokenizer
        cache_dir = cfg.get('cache_dir', './data')
        texts = [self._extract_text(d['chosen']) + " " + self._extract_text(d['rejected']) for d in raw[:5000]]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'ultrafeedback_bpe.json'))
