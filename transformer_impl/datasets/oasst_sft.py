import os
from . import register_dataset
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput


@register_dataset("oasst1")
class OASST1Preparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use OASST1 dataset")

        dataset = load_dataset("OpenAssistant/oasst1", split="train", trust_remote_code=True)
        tokenizer = self._get_tokenizer(cfg, dataset)

        max_seq_len = cfg.get('max_seq_len', 512)
        max_train = cfg.get('max_train_chunks', 35000)

        assistant_msgs = [d['text'] for d in dataset if d['role'] == 'assistant']
        prompt_msgs = [d['text'] for d in dataset if d['role'] == 'prompter']

        examples = self._build_conversations(dataset, max_seq_len, tokenizer)

        if max_train and len(examples) > max_train:
            examples = examples[:max_train]

        return DatasetOutput(
            train_data=examples,
            test_data=examples[:max(1, len(examples)//10)],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )

    def _build_conversations(self, dataset, max_seq_len, tokenizer):
        from collections import defaultdict
        threads = defaultdict(list)
        for msg in dataset:
            threads[msg['message_tree_id']].append(msg)

        examples = []
        for tree_id, messages in threads.items():
            messages.sort(key=lambda m: m['created_date'])
            convo_text = ""
            for msg in messages:
                prefix = "### Human:\n" if msg['role'] == 'prompter' else "### Assistant:\n"
                convo_text += prefix + msg['text'] + "\n"
            tokens = tokenizer.encode(convo_text)
            if len(tokens) <= max_seq_len:
                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                examples.append({
                    'text': padded,
                    'input_ids': padded,
                })
        return examples

    def _get_tokenizer(self, cfg, dataset):
        from transformer_impl.datasets.tokenizer import train_bpe_tokenizer
        cache_dir = cfg.get('cache_dir', './data')
        texts = [d['text'] for d in dataset]
        return train_bpe_tokenizer(texts, vocab_size=cfg.get('vocab_size', 4096),
                                   cache_path=os.path.join(cache_dir, 'oasst1_bpe.json'))
