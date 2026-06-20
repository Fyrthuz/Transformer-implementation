#!/usr/bin/env python3
"""Test all attention × FFN × position combinations for model creation and training."""
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformer_impl.config import config_from_cli
from transformer_impl.model import Transformer
from transformer_impl.train import train_model, TextDataset
from transformer_impl.datasets import get_dataset_preparer
from transformer_impl.attention import ATTENTION_REGISTRY
from transformer_impl.ffn import FFN_REGISTRY
from transformer_impl.position import POSITION_REGISTRY


PASS = "✓"
FAIL = "✗"
SKIP = "—"

results = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"{'='*80}")

def test_combinations():
    d_model = 64
    num_layers = 2
    d_ff = 128

    for attn_name in sorted(ATTENTION_REGISTRY.keys()):
        for ffn_name in sorted(FFN_REGISTRY.keys()):
            for pos_name in sorted(POSITION_REGISTRY.keys()):
                label = f"{attn_name:12s} + {ffn_name:8s} + {pos_name:12s}"
                start = time.time()

                try:
                    cfg_dict = {
                        'name': f'test_{attn_name}_{ffn_name}_{pos_name}',
                        'model': {
                            'd_model': d_model,
                            'num_layers': num_layers,
                            'dropout': 0.1,
                            'attention': {'type': attn_name, 'num_heads': 4,
                                          'num_kv_heads': 2 if attn_name == 'gqa' else None,
                                          'd_state': 8, 'expand_factor': 2, 'd_conv': 3},
                            'ffn': {'type': ffn_name, 'd_ff': d_ff,
                                    'num_experts': 4, 'top_k': 2},
                            'position': {'type': pos_name},
                        },
                        'dataset': {
                            'name': 'tinyshakespeare', 'tokenization': 'char',
                            'max_seq_len': 32, 'train_stride': 32,
                            'max_train_chunks': 5, 'max_test_chunks': 3,
                        },
                        'training': {
                            'batch_size': 4, 'num_epochs': 1,
                            'learning_rate': 0.001, 'weight_decay': 0.0,
                            'scheduler': 'none',
                        },
                    }

                    cfg = config_from_cli(None, cfg_dict)

                    dout = get_dataset_preparer('tinyshakespeare')().prepare({
                        'tokenization': 'char', 'max_seq_len': 32,
                        'max_train_chunks': 5, 'max_test_chunks': 3,
                    })

                    model = Transformer(cfg, dout.vocab_size).to(device)

                    def init_weights(m):
                        if isinstance(m, torch.nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                torch.nn.init.constant_(m.bias, 0)
                        elif isinstance(m, torch.nn.LayerNorm):
                            torch.nn.init.constant_(m.weight, 1.0)
                            torch.nn.init.constant_(m.bias, 0)
                        elif isinstance(m, torch.nn.Embedding):
                            torch.nn.init.normal_(m.weight, mean=0, std=d_model**-0.5)
                    model.apply(init_weights)

                    with torch.no_grad():
                        dummy = torch.randint(0, dout.vocab_size, (2, 16), device=device)
                        mask = model.generate_causal_mask(16, device)
                        logits = model(dummy, mask=mask)
                        assert logits.shape == (2, 16, dout.vocab_size), f"Shape mismatch: {logits.shape}"
                        params = sum(p.numel() for p in model.parameters())

                    loss, ppl = train_model(model, cfg, dout, device)

                    elapsed = time.time() - start
                    results.append((True, label, params, loss, ppl, elapsed))
                    print(f"  {PASS} {label:35s} | params={params/1e3:.1f}K | loss={loss:.4f} | ppl={ppl:.1f} | {elapsed:.1f}s")

                except Exception as e:
                    elapsed = time.time() - start
                    results.append((False, label, 0, 0, 0, elapsed))
                    print(f"  {FAIL} {label:35s} | {type(e).__name__}: {e}")
                    traceback.print_exc()
                    print()


if __name__ == "__main__":
    test_combinations()

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    total = len(results)
    passed = sum(1 for r in results if r[0])
    failed = total - passed
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print()

    if failed > 0:
        print("FAILED:")
        for ok, label, params, loss, ppl, elapsed in results:
            if not ok:
                print(f"  {FAIL} {label}")
    else:
        print("All models created and trained successfully!")

    sys.exit(1 if failed > 0 else 0)
