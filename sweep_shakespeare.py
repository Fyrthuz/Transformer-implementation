#!/usr/bin/env python3
"""Full sweep of all attention × FFN × position combinations on TinyShakespeare."""
import sys, os, time, json, math, itertools
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformer_impl.config import config_from_cli
from transformer_impl.model import Transformer
from transformer_impl.train import train_model
from transformer_impl.datasets import get_dataset_preparer
from transformer_impl.attention import ATTENTION_REGISTRY
from transformer_impl.ffn import FFN_REGISTRY
from transformer_impl.position import POSITION_REGISTRY

PASS = "✓"
FAIL = "✗"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model params
D_MODEL = 64
NUM_LAYERS = 2
D_FF = 128
NUM_HEADS = 4
D_STATE = 8
EXPAND_FACTOR = 2
D_CONV = 3
NUM_EXPERTS = 4
TOP_K = 2

# Dataset params
DATASET = 'tinyshakespeare'
TOKENIZATION = 'char'
MAX_SEQ_LEN = 64
TRAIN_STRIDE = 32
MAX_TRAIN_CHUNKS = 500
MAX_TEST_CHUNKS = 50

# Training params
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 5
LABEL_SMOOTHING = 0.05

def build_cfg(attn_name, ffn_name, pos_name, dataset_kwargs, vocab_size):
    cfg_dict = {
        'name': f'sweep_{attn_name}_{ffn_name}_{pos_name}',
        'seed': 42,
        'model': {
            'd_model': D_MODEL,
            'num_layers': NUM_LAYERS,
            'dropout': 0.1,
            'stochastic_depth': 0.0,
            'attention': {
                'type': attn_name,
                'num_heads': NUM_HEADS,
                'num_kv_heads': 2 if attn_name == 'gqa' else None,
                'd_state': D_STATE,
                'expand_factor': EXPAND_FACTOR,
                'd_conv': D_CONV,
            },
            'ffn': {
                'type': ffn_name,
                'd_ff': D_FF,
                'activation': 'gelu',
                'num_experts': NUM_EXPERTS,
                'top_k': TOP_K,
            },
            'position': {'type': pos_name},
        },
        'dataset': {
            'name': DATASET,
            'tokenization': TOKENIZATION,
            'max_seq_len': MAX_SEQ_LEN,
            'train_stride': TRAIN_STRIDE,
            'max_train_chunks': MAX_TRAIN_CHUNKS,
            'max_test_chunks': MAX_TEST_CHUNKS,
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'grad_clip': 1.0,
            'scheduler': 'cosine',
            'early_stop_patience': EARLY_STOP_PATIENCE,
            'loss': {'type': 'cross_entropy', 'label_smoothing': LABEL_SMOOTHING},
        },
    }
    return config_from_cli(None, cfg_dict)

def init_weights(m, d_model):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0, std=d_model**-0.5)

def run_sweep():
    print(f"{'='*90}")
    print(f"  Shakespeare Sweep — All Combinations")
    print(f"  Device: {device}  |  Model: d_model={D_MODEL}, layers={NUM_LAYERS}, d_ff={D_FF}")
    print(f"  Dataset: {DATASET}, {TOKENIZATION}, seq_len={MAX_SEQ_LEN}, stride={TRAIN_STRIDE}")
    print(f"  Chunks: train={MAX_TRAIN_CHUNKS}, test={MAX_TEST_CHUNKS}")
    print(f"  Training: epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"{'='*90}")
    print()

    combos = list(itertools.product(
        sorted(ATTENTION_REGISTRY.keys()),
        sorted(FFN_REGISTRY.keys()),
        sorted(POSITION_REGISTRY.keys()),
    ))
    print(f"Total combinations: {len(combos)}")
    print(f"{'='*90}\n")

    # Prepare dataset once (shared across all runs)
    dataset_kwargs = {
        'tokenization': TOKENIZATION,
        'max_seq_len': MAX_SEQ_LEN,
        'train_stride': TRAIN_STRIDE,
        'max_train_chunks': MAX_TRAIN_CHUNKS,
        'max_test_chunks': MAX_TEST_CHUNKS,
    }
    print("Preparing dataset...", end=" ", flush=True)
    dout = get_dataset_preparer(DATASET)().prepare(dataset_kwargs)
    print(f"done. Vocab size: {dout.vocab_size}")
    print()

    all_results = []
    passed = 0
    failed = 0
    total = len(combos)

    for idx, (attn_name, ffn_name, pos_name) in enumerate(combos, 1):
        label = f"{attn_name:12s} + {ffn_name:8s} + {pos_name:12s}"
        start = time.time()
        sys.stdout.write(f"[{idx:3d}/{total}] {label} ... ")
        sys.stdout.flush()

        try:
            cfg = build_cfg(attn_name, ffn_name, pos_name, dataset_kwargs, dout.vocab_size)

            model = Transformer(cfg, dout.vocab_size).to(device)
            model.apply(lambda m: init_weights(m, D_MODEL))

            params = sum(p.numel() for p in model.parameters())

            best_loss, best_ppl = train_model(model, cfg, dout, device)

            elapsed = time.time() - start
            passed += 1
            status = PASS
            all_results.append((attn_name, ffn_name, pos_name, best_loss, best_ppl, params, elapsed, True))
            print(f"{PASS} loss={best_loss:.4f}  ppl={best_ppl:.1f}  params={params/1e3:.1f}K  {elapsed:.1f}s")

        except Exception as e:
            import traceback
            elapsed = time.time() - start
            failed += 1
            status = FAIL
            all_results.append((attn_name, ffn_name, pos_name, 0, 0, 0, elapsed, False))
            tb = traceback.format_exc().split('\n')
            tb_short = '\n'.join(tb[-4:-1]) if len(tb) > 3 else traceback.format_exc()[:200]
            print(f"{FAIL} {type(e).__name__}: {e}")
            print(f"       {tb_short}")

    # ---- Ranking ----
    print(f"\n{'='*90}")
    print(f"  RANKING — Best configurations by Test Perplexity")
    print(f"{'='*90}")
    print(f"  Total: {total} | Passed: {passed} | Failed: {failed}")
    print()

    successful = [(a, f, p, l, ppl, pa, e) for (a, f, p, l, ppl, pa, e, ok) in all_results if ok]
    successful.sort(key=lambda x: x[4])  # sort by PPL ascending

    print(f"  {'Rank':>4s}  {'Attention':12s}  {'FFN':10s}  {'Position':12s}  {'Loss':>8s}  {'PPL':>8s}  {'Params':>8s}  {'Time':>6s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for rank, (a, f, p, loss, ppl, params, elapsed) in enumerate(successful[:30], 1):
        print(f"  {rank:4d}  {a:12s}  {f:10s}  {p:12s}  {loss:8.4f}  {ppl:8.2f}  {params/1e3:7.1f}K  {elapsed:5.1f}s")

    if len(successful) > 30:
        print(f"  {'...':>4s}  ({len(successful) - 30} more combinations not shown)")

    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        'config': {
            'd_model': D_MODEL, 'num_layers': NUM_LAYERS, 'd_ff': D_FF,
            'num_heads': NUM_HEADS, 'd_state': D_STATE, 'expand_factor': EXPAND_FACTOR,
            'dataset': DATASET, 'tokenization': TOKENIZATION,
            'max_seq_len': MAX_SEQ_LEN, 'train_stride': TRAIN_STRIDE,
            'max_train_chunks': MAX_TRAIN_CHUNKS, 'max_test_chunks': MAX_TEST_CHUNKS,
            'batch_size': BATCH_SIZE, 'num_epochs': NUM_EPOCHS, 'learning_rate': LEARNING_RATE,
        },
        'results': [
            {
                'attention': a, 'ffn': f, 'position': p,
                'test_loss': l, 'test_ppl': ppl,
                'params': pa, 'time_s': e, 'success': ok,
            }
            for (a, f, p, l, ppl, pa, e, ok) in all_results
        ],
        'ranking': [
            {
                'rank': r, 'attention': a, 'ffn': f, 'position': p,
                'test_loss': l, 'test_ppl': ppl, 'params': pa, 'time_s': e,
            }
            for r, (a, f, p, l, ppl, pa, e) in enumerate(successful, 1)
        ],
    }
    out_path = f"sweep_shakespeare_results_{timestamp}.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults exported to: {out_path}")

    sys.exit(1 if failed > 0 else 0)

if __name__ == "__main__":
    run_sweep()
