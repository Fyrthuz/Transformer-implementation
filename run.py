#!/usr/bin/env python3
import argparse
import sys
import os
import math
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformer_impl.config import (
    config_from_cli, parse_cli_overrides, ExperimentConfig,
    AttentionConfig, FFNConfig, PositionConfig, ModelConfig, TrainConfig, DatasetConfig, LossConfig,
)
from transformer_impl.model import Transformer
from transformer_impl.train import train_model
from transformer_impl.generate import generate_text
from transformer_impl.datasets import get_dataset_preparer
from transformer_impl.attention import ATTENTION_REGISTRY
from transformer_impl.ffn import FFN_REGISTRY
from transformer_impl.position import POSITION_REGISTRY
from transformer_impl.datasets import DATASET_REGISTRY


def initialize_weights(m, d_model):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0, std=d_model ** -0.5)


def cmd_train(args):
    overrides = parse_cli_overrides(args.overrides)
    cfg = config_from_cli(args.config, overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)

    print(f"\n{'='*60}")
    print(f"Experiment: {cfg.name}")
    print(f"Attention: {cfg.model.attention.type} | FFN: {cfg.model.ffn.type} | Position: {cfg.model.position.type}")
    print(f"Dataset: {cfg.dataset.name} | Tokenization: {cfg.dataset.tokenization}")
    print(f"{'='*60}\n")

    dataset_preparer_cls = get_dataset_preparer(cfg.dataset.name)
    dataset_cfg = {
        'tokenization': cfg.dataset.tokenization,
        'max_seq_len': cfg.dataset.max_seq_len,
        'train_stride': cfg.dataset.train_stride,
        'vocab_size': cfg.dataset.vocab_size,
        'cache_dir': cfg.dataset.cache_dir,
        'max_train_chunks': cfg.dataset.max_train_chunks,
        'max_test_chunks': cfg.dataset.max_test_chunks,
    }
    dataset_output = dataset_preparer_cls().prepare(dataset_cfg)

    cfg_model = cfg.model
    cfg_model_dict = {
        'attention': {'type': cfg_model.attention.type, 'num_heads': cfg_model.attention.num_heads,
                       'num_kv_heads': cfg_model.attention.num_kv_heads,
                       'window_size': cfg_model.attention.window_size,
                       'dilation': cfg_model.attention.dilation,
                       'd_state': cfg_model.attention.d_state,
                       'expand_factor': cfg_model.attention.expand_factor,
                       'd_conv': cfg_model.attention.d_conv},
        'ffn': {'type': cfg_model.ffn.type, 'd_ff': cfg_model.ffn.d_ff,
                'num_experts': cfg_model.ffn.num_experts, 'top_k': cfg_model.ffn.top_k},
        'position': {'type': cfg_model.position.type, 'max_len': cfg_model.position.max_len,
                     'rope_theta': cfg_model.position.rope_theta},
        'd_model': cfg_model.d_model,
        'num_layers': cfg_model.num_layers,
        'dropout': cfg_model.dropout,
    }

    model = Transformer(cfg, dataset_output.vocab_size).to(device)
    model.apply(lambda m: initialize_weights(m, cfg.model.d_model))

    best_loss, best_ppl = train_model(model, cfg, dataset_output, device)

    print(f"\n{'='*60}")
    print(f"Experiment '{cfg.name}' completed!")
    print(f"Best Test Loss: {best_loss:.4f} | Best Perplexity: {best_ppl:.2f}")
    print(f"{'='*60}\n")


def cmd_generate(args):
    overrides = parse_cli_overrides(args.overrides)
    cfg = config_from_cli(args.config, overrides)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_preparer_cls = get_dataset_preparer(cfg.dataset.name)
    dataset_cfg = {
        'tokenization': cfg.dataset.tokenization,
        'max_seq_len': cfg.dataset.max_seq_len,
        'vocab_size': cfg.dataset.vocab_size,
        'cache_dir': cfg.dataset.cache_dir,
        'train_stride': cfg.dataset.train_stride,
    }
    dataset_output = dataset_preparer_cls().prepare(dataset_cfg)

    model = Transformer(cfg, dataset_output.vocab_size).to(device)
    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Warning: no model path provided or file not found. Using random weights.")

    model.eval()
    generate_text(
        model, dataset_output.tokenizer,
        prompt=args.prompt,
        max_chars=args.max_chars,
        temperature=args.temperature,
        device=device,
    )


def cmd_sweep(args):
    import yaml
    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    base_cfg = config_from_cli(args.config, {})
    param_grid = {}
    non_sweep_keys = {'name', 'strategy', 'max_combinations', 'repetitions'}
    sweep_meta = {k: sweep_cfg.get('sweep', {}).get(k) for k in non_sweep_keys}

    for key, values in sweep_cfg.get('sweep', {}).items():
        if key not in non_sweep_keys and isinstance(values, list):
            param_grid[key] = values

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    max_combos = sweep_meta.get('max_combinations', 50)
    if len(combos) > max_combos:
        print(f"Limiting from {len(combos)} to {max_combos} combinations")
        combos = combos[:max_combos]

    print(f"\nSweep: {sweep_meta.get('name', args.config)}")
    print(f"Total combinations: {len(combos)}")
    print(f"Parameters: {keys}")
    print()

    results = []
    for idx, combo in enumerate(combos):
        overrides = {}
        for k, v in zip(keys, combo):
            parts = k.split('.')
            d = overrides
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v

        cfg = config_from_cli(args.config, overrides)
        cfg.name = f"sweep_{idx:03d}"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(cfg.seed + idx)

        print(f"[{idx+1}/{len(combos)}] Running: {cfg.name}")
        print(f"  Config: {combo}")

        dataset_preparer_cls = get_dataset_preparer(cfg.dataset.name)
        dataset_cfg = {
            'tokenization': cfg.dataset.tokenization,
            'max_seq_len': cfg.dataset.max_seq_len,
            'train_stride': cfg.dataset.train_stride,
            'vocab_size': cfg.dataset.vocab_size,
            'cache_dir': cfg.dataset.cache_dir,
            'max_train_chunks': cfg.dataset.max_train_chunks,
            'max_test_chunks': cfg.dataset.max_test_chunks,
        }
        dataset_output = dataset_preparer_cls().prepare(dataset_cfg)

        model = Transformer(cfg, dataset_output.vocab_size).to(device)
        model.apply(lambda m: initialize_weights(m, cfg.model.d_model))

        writer = SummaryWriter(f'runs/{cfg.name}')

        best_loss, best_ppl = train_model(model, cfg, dataset_output, device, writer=writer)
        results.append((combo, best_loss, best_ppl))
        print(f"  Result: Loss={best_loss:.4f}, PPL={best_ppl:.2f}\n")

    print(f"\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<60} {'Test Loss':<12} {'PPL':<10}")
    print("-" * 82)
    for combo, loss, ppl in sorted(results, key=lambda x: x[1]):
        print(f"{str(combo):<60} {loss:<12.4f} {ppl:<10.2f}")
    print(f"{'='*60}\n")


def cmd_list(args):
    print("\nAvailable Attention Mechanisms:")
    for name in sorted(ATTENTION_REGISTRY.keys()):
        print(f"  - {name}")

    print("\nAvailable FFN Types:")
    for name in sorted(FFN_REGISTRY.keys()):
        print(f"  - {name}")

    print("\nAvailable Position Encodings:")
    for name in sorted(POSITION_REGISTRY.keys()):
        print(f"  - {name}")

    print("\nAvailable Datasets:")
    for name in sorted(DATASET_REGISTRY.keys()):
        print(f"  - {name}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Transformer Implementation - Experiment Runner")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('-c', '--config', type=str, default=None, help='Path to YAML config file')
    train_parser.add_argument('overrides', nargs='*', help='Config overrides: key=value')

    gen_parser = subparsers.add_parser('generate', help='Generate text with trained model')
    gen_parser.add_argument('-c', '--config', type=str, default=None, help='Path to YAML config file')
    gen_parser.add_argument('-m', '--model-path', type=str, default='best_model.pt', help='Path to model checkpoint')
    gen_parser.add_argument('-p', '--prompt', type=str, default='BAPTISTA:\n', help='Text prompt')
    gen_parser.add_argument('-t', '--temperature', type=float, default=0.7, help='Sampling temperature')
    gen_parser.add_argument('-n', '--max-chars', type=int, default=500, help='Max characters to generate')
    gen_parser.add_argument('overrides', nargs='*', help='Config overrides: key=value')

    sweep_parser = subparsers.add_parser('sweep', help='Run grid search sweep')
    sweep_parser.add_argument('-c', '--config', type=str, required=True, help='Path to sweep YAML config')

    list_parser = subparsers.add_parser('list', help='List available components')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'sweep':
        cmd_sweep(args)
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
