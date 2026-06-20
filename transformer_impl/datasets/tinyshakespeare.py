import os
import random
import urllib.request
from . import register_dataset
from .base import BaseDatasetPreparer, DatasetOutput
from .tokenizer import CharTokenizer, BPETokenizer, train_bpe_tokenizer

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "tinyshakespeare.txt"

@register_dataset("tinyshakespeare")
class TinyShakespearePreparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        tokenization = cfg.get('tokenization', 'bpe')
        max_seq_len = cfg.get('max_seq_len', 128)
        train_stride = cfg.get('train_stride', 16)
        vocab_size = cfg.get('vocab_size', 4096)
        cache_dir = cfg.get('cache_dir', './data')
        max_train = cfg.get('max_train_chunks', 35000)
        max_test = cfg.get('max_test_chunks', 2000)

        if not os.path.exists(DATA_PATH):
            print("Downloading TinyShakespeare...")
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)

        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        if tokenization == 'char':
            chars = sorted(list(set(raw_text)))
            vocab_list = ['<pad>'] + chars
            tokenizer = CharTokenizer(vocab_list)
        else:
            cache_path = os.path.join(cache_dir, 'tinyshakespeare_bpe.json')
            tokenizer = train_bpe_tokenizer(raw_text, vocab_size=vocab_size, cache_path=cache_path)

        all_tokens = tokenizer.encode(raw_text)
        split_idx = int(len(all_tokens) * 0.9)
        train_tokens = all_tokens[:split_idx]
        test_tokens = all_tokens[split_idx:]

        train_chunks = []
        for i in range(0, len(train_tokens) - max_seq_len, train_stride):
            train_chunks.append(train_tokens[i:i + max_seq_len])

        test_chunks = []
        for i in range(0, len(test_tokens) - max_seq_len, max_seq_len):
            test_chunks.append(test_tokens[i:i + max_seq_len])

        random.seed(42)
        random.shuffle(train_chunks)

        return DatasetOutput(
            train_data=[{'text': c} for c in train_chunks[:max_train]],
            test_data=[{'text': c} for c in test_chunks[:max_test]],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )
