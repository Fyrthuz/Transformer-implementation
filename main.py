#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_impl.config import config_from_cli
from transformer_impl.model import Transformer
from transformer_impl.train import train_model
from transformer_impl.datasets import get_dataset_preparer
import torch

def initialize_weights(m, d_model):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0, std=d_model**-0.5)

if __name__ == "__main__":
    cfg = config_from_cli(None, {
        'name': 'transformer_shakespeare',
        'model': {'d_model': 256, 'num_layers': 4, 'dropout': 0.1,
                  'attention': {'type': 'mha', 'num_heads': 8},
                  'ffn': {'type': 'standard', 'd_ff': 1024, 'activation': 'gelu'},
                  'position': {'type': 'sinusoidal'}},
        'dataset': {'name': 'tinyshakespeare', 'tokenization': 'char',
                    'max_seq_len': 128, 'train_stride': 16,
                    'max_train_chunks': 35000, 'max_test_chunks': 2000},
        'training': {'batch_size': 64, 'num_epochs': 30, 'learning_rate': 0.0005,
                     'weight_decay': 0.01, 'grad_clip': 1.0, 'scheduler': 'cosine',
                     'loss': {'type': 'cross_entropy', 'label_smoothing': 0.025}},
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)

    dataset_cfg = {
        'tokenization': cfg.dataset.tokenization, 'max_seq_len': cfg.dataset.max_seq_len,
        'train_stride': cfg.dataset.train_stride, 'vocab_size': cfg.dataset.vocab_size,
        'cache_dir': cfg.dataset.cache_dir, 'max_train_chunks': cfg.dataset.max_train_chunks,
        'max_test_chunks': cfg.dataset.max_test_chunks,
    }
    dataset_output = get_dataset_preparer(cfg.dataset.name)().prepare(dataset_cfg)

    model = Transformer(cfg, dataset_output.vocab_size).to(device)
    model.apply(lambda m: initialize_weights(m, cfg.model.d_model))

    train_model(model, cfg, dataset_output, device)
