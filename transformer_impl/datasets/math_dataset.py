import os
from . import register_dataset
from .gsm8k import GSM8KPreparer


@register_dataset("math")
class MATHPreparer(GSM8KPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use MATH dataset")

        dataset = load_dataset("hendrycks/math", split="train", trust_remote_code=True)
        tokenizer = self._get_tokenizer(cfg, dataset)

        max_seq_len = cfg.get('max_seq_len', 1024)
        max_train = cfg.get('max_train_chunks', 12000)

        examples = []
        for item in dataset:
            problem = item['problem']
            solution = item['solution']
            text = f"Problem: {problem}\nSolution: Let me solve this step by step.\n"
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_seq_len:
                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                examples.append({
                    'text': padded[:max_seq_len],
                    'question': problem,
                    'answer': solution,
                    'answer_num': "",
                })

        if max_train and len(examples) > max_train:
            examples = examples[:max_train]

        return self._make_output(examples, tokenizer, cfg)

    def _make_output(self, examples, tokenizer, cfg):
        from transformer_impl.datasets.base import DatasetOutput
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
        texts = [d['problem'] + " " + d['solution'] for d in dataset]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'math_bpe.json'))
