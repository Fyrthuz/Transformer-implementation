#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_impl.config import config_from_cli
from transformer_impl.model import Transformer
from transformer_impl.generate import generate_text
from transformer_impl.datasets import get_dataset_preparer
import torch

if __name__ == "__main__":
    cfg = config_from_cli(None, {
        'name': 'generate',
        'model': {'d_model': 256, 'num_layers': 4, 'dropout': 0.0,
                  'attention': {'type': 'mha', 'num_heads': 8},
                  'ffn': {'type': 'standard', 'd_ff': 1024, 'activation': 'gelu'},
                  'position': {'type': 'sinusoidal'}},
        'dataset': {'name': 'tinyshakespeare', 'tokenization': 'char',
                    'max_seq_len': 128, 'train_stride': 16},
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_cfg = {
        'tokenization': cfg.dataset.tokenization, 'max_seq_len': cfg.dataset.max_seq_len,
        'train_stride': cfg.dataset.train_stride, 'vocab_size': cfg.dataset.vocab_size,
        'cache_dir': cfg.dataset.cache_dir,
    }
    dataset_output = get_dataset_preparer(cfg.dataset.name)().prepare(dataset_cfg)

    model = Transformer(cfg, dataset_output.vocab_size).to(device)
    checkpoint = torch.load('best_model.pt', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    generate_text(model, dataset_output.tokenizer,
                  prompt="BAPTISTA:\n", max_chars=600, temperature=0.65, device=device)
