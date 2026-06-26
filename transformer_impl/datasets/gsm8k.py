import os
import re
from . import register_dataset
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput


@register_dataset("gsm8k")
class GSM8KPreparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use GSM8K dataset")

        dataset = load_dataset("openai/gsm8k", "main", split="train")
        tokenizer = self._get_tokenizer(cfg, dataset)

        max_seq_len = cfg.get('max_seq_len', 1024)
        max_train = cfg.get('max_train_chunks', 7500)

        examples = []
        for item in dataset:
            question = item['question']
            answer = item['answer']
            answer_num = self._extract_answer(answer)
            text = f"Question: {question}\nAnswer: Let me solve this step by step.\n"
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_seq_len:
                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                examples.append({
                    'text': padded[:max_seq_len],
                    'question': question,
                    'answer': answer,
                    'answer_num': answer_num,
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

    def _extract_answer(self, answer_text):
        match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
        if match:
            return match.group(1)
        return ""

    def _get_tokenizer(self, cfg, dataset):
        from transformer_impl.datasets.tokenizer import train_bpe_tokenizer
        cache_dir = cfg.get('cache_dir', './data')
        texts = [d['question'] + " " + d['answer'] for d in dataset]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'gsm8k_bpe.json'))
