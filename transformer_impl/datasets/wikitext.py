import os
import random
import urllib.request
from . import register_dataset
from .base import BaseDatasetPreparer, DatasetOutput
from .tokenizer import train_bpe_tokenizer

WIKITEXT_URLS = {
    "wikitext2": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "wikitext103": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
}

def download_wikitext(name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, f"{name}_train.txt")
    test_path = os.path.join(cache_dir, f"{name}_test.txt")

    if not os.path.exists(train_path):
        base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data"
        datasets_to_files = {
            "wikitext2": f"{base_url}/wikitext-2",
            "wikitext103": f"{base_url}/wikitext-2",
        }
        url_base = datasets_to_files.get(name, f"{base_url}/wikitext-2")
        for split, dest in [("train.txt", train_path), ("test.txt", test_path)]:
            url = f"{url_base}/{split}"
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, dest)

    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = f.read()
    return train_text, test_text


@register_dataset("wikitext2")
class WikiText2Preparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        return self._prepare("wikitext2", cfg)

    def _prepare(self, name, cfg):
        max_seq_len = cfg.get('max_seq_len', 128)
        train_stride = cfg.get('train_stride', 16)
        vocab_size = cfg.get('vocab_size', 4096)
        cache_dir = cfg.get('cache_dir', './data')
        max_train = cfg.get('max_train_chunks', 35000)
        max_test = cfg.get('max_test_chunks', 2000)

        train_text, test_text = download_wikitext(name, cache_dir)

        tokenizer_cache = os.path.join(cache_dir, f'{name}_bpe.json')
        tokenizer = train_bpe_tokenizer(train_text, vocab_size=vocab_size, cache_path=tokenizer_cache)

        train_tokens = tokenizer.encode(train_text)
        test_tokens = tokenizer.encode(test_text)

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


@register_dataset("wikitext103")
class WikiText103Preparer(WikiText2Preparer):
    def prepare(self, cfg):
        return self._prepare("wikitext103", cfg)
