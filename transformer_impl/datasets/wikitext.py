import os
import random
import urllib.request
from . import register_dataset
from .base import BaseDatasetPreparer, DatasetOutput, cache_chunks, load_cached_chunks
from .tokenizer import train_bpe_tokenizer


def download_wikitext(name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, f"{name}_train.txt")
    test_path = os.path.join(cache_dir, f"{name}_test.txt")

    if not os.path.exists(train_path):
        base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data"
        url_base = f"{base_url}/wikitext-2" if name == "wikitext2" else f"{base_url}/wikitext-103"
        for split, dest in [("train.txt", train_path), ("test.txt", test_path)]:
            url = f"{url_base}/{split}"
            print(f"Downloading {url}...")
            urllib.request.urlretrieve(url, dest)

    with open(train_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(test_path, 'r', encoding='utf-8') as f:
        test_text = f.read()
    return train_text, test_text


def _prepare_dataset(name, cfg):
    max_seq_len = cfg.get('max_seq_len', 128)
    train_stride = cfg.get('train_stride', 16)
    vocab_size = cfg.get('vocab_size', 4096)
    cache_dir = cfg.get('cache_dir', './data')
    max_train = cfg.get('max_train_chunks', 35000)
    max_test = cfg.get('max_test_chunks', 2000)

    chunk_cache = os.path.join(cache_dir, f"{name}_sl{max_seq_len}_str{train_stride}.pt")
    if os.path.exists(chunk_cache):
        print(f"[Dataset] Loading cached chunks from {chunk_cache}")
        train_chunks, test_chunks = load_cached_chunks(chunk_cache)
        tokenizer_cache = os.path.join(cache_dir, f'{name}_bpe.json')
        tokenizer = train_bpe_tokenizer("", vocab_size=vocab_size, cache_path=tokenizer_cache)
        return DatasetOutput(
            train_data=[{'text': c} for c in train_chunks[:max_train]],
            test_data=[{'text': c} for c in test_chunks[:max_test]],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
        )

    train_text, test_text = download_wikitext(name, cache_dir)

    print(f"[Dataset] Training BPE tokenizer (vocab={vocab_size})...")
    tokenizer_cache = os.path.join(cache_dir, f'{name}_bpe.json')
    tokenizer = train_bpe_tokenizer(train_text, vocab_size=vocab_size, cache_path=tokenizer_cache)

    print(f"[Dataset] Tokenizing text...")
    train_tokens = tokenizer.encode(train_text)
    test_tokens = tokenizer.encode(test_text)

    print(f"[Dataset] Generating chunks (stride={train_stride}, seq_len={max_seq_len})...")
    train_chunks = [train_tokens[i:i + max_seq_len] for i in range(0, len(train_tokens) - max_seq_len, train_stride)]
    test_chunks = [test_tokens[i:i + max_seq_len] for i in range(0, len(test_tokens) - max_seq_len, max_seq_len)]

    random.seed(42)
    random.shuffle(train_chunks)

    print(f"[Dataset] Caching chunks to {chunk_cache}...")
    cache_chunks(chunk_cache, train_chunks, test_chunks)

    return DatasetOutput(
        train_data=[{'text': c} for c in train_chunks[:max_train]],
        test_data=[{'text': c} for c in test_chunks[:max_test]],
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        tokenizer=tokenizer,
    )


@register_dataset("wikitext2")
class WikiText2Preparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        return _prepare_dataset("wikitext2", cfg)


@register_dataset("wikitext103")
class WikiText103Preparer(BaseDatasetPreparer):
    def prepare(self, cfg):
        return _prepare_dataset("wikitext103", cfg)
