# Datasets

Three datasets are available via the registry pattern (`@register_dataset("name")` / `get_dataset_preparer(name)`).

## Interface

```python
class BaseDatasetPreparer:
    def prepare(self, cfg: dict) -> DatasetOutput
```

`DatasetOutput` is a dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `train_data` | `list[dict]` | Each entry: `{'text': list[int]}` — token IDs |
| `test_data` | `list[dict]` | Same format |
| `vocab_size` | `int` | Number of unique tokens |
| `pad_token_id` | `int` | Padding token ID (always 0) |
| `tokenizer` | `object` | Tokenizer with `.encode()`, `.decode()` |

## Available Datasets

| Registry Name | Class | Source |
|---------------|-------|--------|
| `tinyshakespeare` | `TinyShakespearePreparer` | Karpathy's char-rnn repo |
| `wikitext2` | `WikiText2Preparer` | WikiText-2 (pytorch/examples) |
| `wikitext103` | `WikiText103Preparer` | WikiText-103 (pytorch/examples) |

## Data Pipeline

```
Raw text → Tokenizer → Token IDs → Chunking → Shuffle → Cache (tensor .pt) → DatasetOutput
```

### Chunking

- **Train chunks**: Sliding window with stride: `train_tokens[i : i + max_seq_len]` for `i = 0, stride, 2*stride, ...`
- **Test chunks**: Non-overlapping: `test_tokens[i : i + max_seq_len]` for `i = 0, max_seq_len, 2*max_seq_len, ...`
- Train chunks are shuffled with seed 42; test chunks are left in order.
- Limited by `max_train_chunks` / `max_test_chunks` (default: 35000 / 2000).

### Caching

Chunks are cached as a single `.pt` file containing `{'train': LongTensor, 'test': LongTensor}`:

```
cache_dir/tinyshakespeare_char_sl128_str16.pt
cache_dir/wikitext2_sl128_str16.pt
```

On subsequent runs, the cache is loaded directly (~40x speedup), skipping tokenization and chunking entirely.

## Tokenizers

### CharTokenizer

**File:** `transformer_impl/datasets/tokenizer.py:CharTokenizer`

- Vocabulary built from sorted unique characters in the text + `<pad>`.
- Encoding: character-level lookup.
- Decoding: joins characters back into string.

```python
tokenizer = CharTokenizer(['<pad>', 'a', 'b', ...])
tokenizer.encode("hello")  # → [id_h, id_e, id_l, id_l, id_o]
```

### BPETokenizer

**File:** `transformer_impl/datasets/tokenizer.py:BPETokenizer`

- Wraps the HuggingFace `tokenizers` library's BPE model.
- Byte-level pre-tokenization (handles all Unicode).
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`.
- Trained on the dataset text; cached as JSON for reuse.
- Falls back to `<pad>` ID=0 if `token_to_id` returns None.

```python
tokenizer = train_bpe_tokenizer(text, vocab_size=4096, cache_path="data/wikitext2_bpe.json")
tokenizer.encode("Hello world")  # → [ids...]
```

## Dataset-Specific Details

### TinyShakespeare

- Downloaded from Karpathy's char-rnn repo if not present locally.
- Supports both `tokenization: char` (character-level) and `tokenization: bpe` (subword).
- Train/test split: 90% / 10%.
- **Char tokenization bug fix**: When loading cached chunks with `tokenization=char`, the tokenizer is recreated from the raw file to ensure correct vocabulary, rather than loading a stale BPE tokenizer.

### WikiText-2 / WikiText-103

- Downloaded from pytorch/examples repository.
- Tokenization is always BPE (no char mode).
- Same chunking and caching as TinyShakespeare.
