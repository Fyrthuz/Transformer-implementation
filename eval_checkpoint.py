#!/usr/bin/env python3
"""Evaluate a trained checkpoint and print metrics."""
import argparse
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformer_impl.config import config_from_yaml
from transformer_impl.model import Transformer
from transformer_impl.datasets import get_dataset_preparer
from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument('-m', '--model-path', required=True, help='Path to checkpoint .pt file')
    parser.add_argument('-c', '--config', default='configs/pretrain_tinystories.yaml',
                        help='YAML config (default: configs/pretrain_tinystories.yaml)')
    parser.add_argument('-d', '--dataset', default=None,
                        help='Dataset name (default: from config)')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of test samples to evaluate (default: 200)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = config_from_yaml(args.config)
    dataset_name = args.dataset or cfg.dataset.name

    preparer = get_dataset_preparer(dataset_name)()
    ds_cfg = {
        'tokenization': cfg.dataset.tokenization,
        'max_seq_len': cfg.dataset.max_seq_len,
        'vocab_size': cfg.dataset.vocab_size,
        'cache_dir': cfg.dataset.cache_dir,
        'max_train_chunks': 1,
        'max_test_chunks': args.samples,
    }
    if hasattr(cfg.dataset, 'train_stride'):
        ds_cfg['train_stride'] = cfg.dataset.train_stride
    ds = preparer.prepare(ds_cfg)

    model = Transformer(cfg, ds.vocab_size).to(device)
    load_checkpoint_with_adaptation(args.model_path, model, device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    total_loss = 0
    n = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for item in ds.test_data:
            if isinstance(item, dict):
                ids = item.get('input_ids') or item.get('text')
            else:
                ids = item
            if ids is None:
                continue
            ids = torch.tensor(ids, device=device).unsqueeze(0)
            if ids.size(1) < 2:
                continue
            mask = model.generate_causal_mask(ids.size(1), device)
            logits = model(ids, mask=mask)
            loss = loss_fn(logits[:, :-1].reshape(-1, ds.vocab_size), ids[:, 1:].reshape(-1))
            total_loss += loss.item()
            n += 1

    avg_loss = total_loss / max(n, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float('inf')

    arch = f"{cfg.model.attention.type}+{cfg.model.ffn.type}+{cfg.model.position.type}"
    dims = f"d={cfg.model.d_model} l={cfg.model.num_layers} h={cfg.model.attention.num_heads} ff={cfg.model.ffn.d_ff} v={ds.vocab_size}"

    print(f"\n{'='*50}")
    print(f"  Model:     {params/1e6:.2f}M params | {arch}")
    print(f"  Config:    {dims}")
    print(f"  Dataset:   {dataset_name} ({n} samples)")
    print(f"  Checkpoint: {args.model_path}")
    print(f"{'='*50}")
    print(f"  Test Loss: {avg_loss:.4f}")
    print(f"  Test PPL:  {ppl:.2f}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()
