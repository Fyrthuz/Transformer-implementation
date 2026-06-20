import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class CharTokenizer:
    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)
        self.id_to_token = {i: c for i, c in enumerate(vocab_list)}
        self.token_to_id = {c: i for i, c in enumerate(vocab_list)}
        self.pad_token_id = self.token_to_id.get('<pad>', 0)

    def encode(self, text):
        return [self.token_to_id.get(c, self.pad_token_id) for c in text]

    def decode(self, ids):
        return ''.join(self.id_to_token.get(i, '') for i in ids if i != self.pad_token_id)


class BPETokenizer:
    def __init__(self, path=None, vocab_size=4096):
        self._target_vocab_size = vocab_size
        if path and os.path.exists(path):
            self.tokenizer = Tokenizer.from_file(path)
        else:
            self.tokenizer = Tokenizer(models.BPE())
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.post_processor = None
        self.pad_token_id = 0

    def train(self, texts, cache_path=None):
        trainer = trainers.BpeTrainer(
            vocab_size=self._target_vocab_size,
            special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'],
            min_frequency=2,
        )
        if isinstance(texts, str):
            texts = [texts]
        self.tokenizer.train_from_iterator(texts, trainer)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.tokenizer.save(cache_path)
        self.pad_token_id = self.tokenizer.token_to_id('<pad>')

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self):
        if self.tokenizer:
            try:
                return self.tokenizer.get_vocab_size()
            except Exception:
                pass
        return self._target_vocab_size


def train_bpe_tokenizer(texts, vocab_size=4096, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        tok = BPETokenizer(path=cache_path)
        tok.pad_token_id = tok.tokenizer.token_to_id('<pad>') if tok.tokenizer.token_to_id('<pad>') is not None else 0
        return tok
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(texts, cache_path=cache_path)
    return tok
