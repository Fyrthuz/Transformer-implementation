import os
import torch
from transformer_impl.datasets.base import BaseDatasetPreparer, DatasetOutput, cache_chunks, load_cached_chunks


class HuggingFaceDatasetPreparer(BaseDatasetPreparer):
    HF_PATH = None
    HF_SPLIT = "train"
    TEXT_FIELD = "text"

    def prepare(self, cfg):
        cache_path = self._cache_path(cfg)
        if cache_path and os.path.exists(cache_path):
            train_data, test_data = load_cached_chunks(cache_path)
            return DatasetOutput(
                train_data=train_data,
                test_data=test_data,
                vocab_size=cfg.get('vocab_size', 4096),
                pad_token_id=0,
                tokenizer=None,
            )

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets to use HuggingFace datasets")

        dataset = load_dataset(self.HF_PATH, split=self.HF_SPLIT, streaming=cfg.get('streaming', False))
        texts = [example[self.TEXT_FIELD] for example in dataset]

        from transformer_impl.datasets.tokenizer import train_bpe_tokenizer
        tokenizer = train_bpe_tokenizer(
            texts,
            vocab_size=cfg.get('vocab_size', 4096),
            cache_path=os.path.join(cfg.get('cache_dir', './data'), f'{self.HF_PATH.replace("/", "_")}_bpe.json'),
        )

        max_seq_len = cfg.get('max_seq_len', 512)
        train_stride = cfg.get('train_stride', max_seq_len // 2)
        max_train = cfg.get('max_train_chunks', 35000)
        max_test = cfg.get('max_test_chunks', 2000)

        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text)
            all_ids.extend(ids)

        split_idx = int(len(all_ids) * 0.95)
        train_ids = all_ids[:split_idx]
        test_ids = all_ids[split_idx:]

        train_chunks = []
        for i in range(0, max(1, len(train_ids) - max_seq_len), train_stride):
            chunk = train_ids[i:i + max_seq_len]
            if len(chunk) == max_seq_len:
                train_chunks.append({'text': chunk})

        test_chunks = []
        for i in range(0, max(1, len(test_ids) - max_seq_len), max_seq_len):
            chunk = test_ids[i:i + max_seq_len]
            if len(chunk) == max_seq_len:
                test_chunks.append({'text': chunk})

        if max_train and len(train_chunks) > max_train:
            import random
            random.shuffle(train_chunks)
            train_chunks = train_chunks[:max_train]
        if max_test and len(test_chunks) > max_test:
            test_chunks = test_chunks[:max_test]

        if cache_path:
            cache_chunks(cache_path, [c['text'] for c in train_chunks], [c['text'] for c in test_chunks])

        return DatasetOutput(
            train_data=train_chunks,
            test_data=test_chunks,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )

    def _cache_path(self, cfg):
        cache_dir = cfg.get('cache_dir', './data')
        ml = cfg.get('max_seq_len', 512)
        st = cfg.get('train_stride', ml // 2)
        vs = cfg.get('vocab_size', 4096)
        name = self.HF_PATH.replace("/", "_") if self.HF_PATH else "hf_dataset"
        return os.path.join(cache_dir, f'{name}_sl{ml}_str{st}_voc{vs}.pt')


class StreamingTextDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids, max_seq_len, pad_token_id=0):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        total = len(token_ids)
        self.num_chunks = total // max_seq_len
        self.token_ids = token_ids[:self.num_chunks * max_seq_len]

    def __len__(self):
        return max(1, self.num_chunks)

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        chunk = self.token_ids[start:start + self.max_seq_len]
        return torch.tensor(chunk, dtype=torch.long)
