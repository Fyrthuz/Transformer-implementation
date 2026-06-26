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
    config_from_cli, parse_cli_overrides, make_run_name, ExperimentConfig,
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

from transformer_impl.trainers import PreTrainer, SFTTrainer, DPOTrainer, PPOTrainer, GRPOTrainer


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
    if 'name' not in overrides:
        cfg.name = make_run_name(cfg, prefix="train")

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
                       'd_conv': cfg_model.attention.d_conv,
                       'scale_attention': cfg_model.attention.scale_attention},
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
    print(f"Logs: runs/")
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
        from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
        load_checkpoint_with_adaptation(args.model_path, model, device)
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


def _setup_model_and_data(args, cfg, device):
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
    return model, dataset_output


def _load_model(path, model, device):
    if path and os.path.exists(path):
        from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
        load_checkpoint_with_adaptation(path, model, device)
    return model


def cmd_stage_train(stage_name, TrainerClass, args):
    overrides = parse_cli_overrides(args.overrides)
    cfg = config_from_cli(args.config, overrides)
    setattr(getattr(cfg, stage_name, cfg.training), 'enabled', True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)

    if 'name' not in overrides:
        cfg.name = make_run_name(cfg, prefix=stage_name)

    print(f"\n{'='*60}")
    print(f"Stage: {stage_name.upper()} — {cfg.name}")
    print(f"Attention: {cfg.model.attention.type} | FFN: {cfg.model.ffn.type} | Position: {cfg.model.position.type}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"{'='*60}\n")

    model, dataset_output = _setup_model_and_data(args, cfg, device)
    if args.model_path:
        model = _load_model(args.model_path, model, device)

    trainer = TrainerClass(cfg, model, dataset_output, device)
    trainer.train()
    print(f"\nVer resultados en TensorBoard:")
    print(f"  tensorboard --logdir runs/")


def cmd_pretrain(args):
    from transformer_impl.trainers import PreTrainer
    cmd_stage_train('pretrain', PreTrainer, args)


def cmd_sft(args):
    from transformer_impl.trainers import SFTTrainer
    cmd_stage_train('sft', SFTTrainer, args)


def cmd_dpo(args):
    from transformer_impl.trainers import DPOTrainer
    cmd_stage_train('dpo', DPOTrainer, args)


def cmd_ppo(args):
    from transformer_impl.trainers import PPOTrainer
    cmd_stage_train('ppo', PPOTrainer, args)


def cmd_grpo(args):
    from transformer_impl.trainers import GRPOTrainer
    cmd_stage_train('grpo', GRPOTrainer, args)


def cmd_pipeline(args):
    import yaml
    with open(args.config) as f:
        pipeline_cfg = yaml.safe_load(f)

    stages = pipeline_cfg.get('stages', [])
    if not stages:
        print("No stages defined in pipeline config")
        return

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline_name = pipeline_cfg.get('name', 'pipeline')
    print(f"\n{'='*60}")
    print(f"Pipeline: {pipeline_name} [{timestamp}]")
    print(f"Stages: {len(stages)}")
    print(f"{'='*60}\n")

    model = None
    checkpoint_path = None
    vocab_size = None

    for i, stage in enumerate(stages):
        stage_name = stage.get('name', f"Stage {i+1}")
        command = stage.get('command', '')
        config_path = stage.get('config', '')

        if not command or not config_path:
            print(f"  [SKIP] {stage_name}: missing command or config")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(stages)}] {stage_name} ({command.upper()})")
        print(f"{'='*60}")

        ckpt = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = checkpoint_path
        elif args.model_path and os.path.exists(args.model_path):
            ckpt = args.model_path

        stage_args = type('Args', (), {
            'config': config_path,
            'model_path': ckpt,
            'overrides': args.overrides,
        })()

        overrides = parse_cli_overrides(stage_args.overrides)
        cfg = config_from_cli(stage_args.config, overrides)
        setattr(getattr(cfg, command, cfg.training), 'enabled', True)
        if 'name' not in overrides:
            cfg.name = make_run_name(cfg, prefix=f"{pipeline_name}_{command}", timestamp=timestamp)
        else:
            cfg.name = overrides['name']

        torch.manual_seed(cfg.seed)

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

        if model is None:
            model = Transformer(cfg, dataset_output.vocab_size).to(device)
            model.apply(lambda m: initialize_weights(m, cfg.model.d_model))
        else:
            model = model.to(device)
        if vocab_size is not None and vocab_size != dataset_output.vocab_size:
            model.output_layer = torch.nn.Linear(model.cfg.model.d_model, dataset_output.vocab_size).to(device)
            model.vocab_size = dataset_output.vocab_size
            print(f"  Resized output layer to vocab_size={dataset_output.vocab_size}")

        if stage_args.model_path and model is not None:
            from transformer_impl.utils.checkpointing import load_checkpoint_with_adaptation
            load_checkpoint_with_adaptation(stage_args.model_path, model, device)

        vocab_size = dataset_output.vocab_size
        trainer_map = {
            'pretrain': PreTrainer,
            'sft': SFTTrainer,
            'dpo': DPOTrainer,
            'ppo': PPOTrainer,
            'grpo': GRPOTrainer,
        }

        TrainerClass = trainer_map.get(command)
        if TrainerClass is None:
            print(f"  [ERROR] Unknown command: {command}")
            continue

        try:
            trainer = TrainerClass(cfg, model, dataset_output, device)
            trainer.train()
        except Exception as e:
            print(f"\n  [ERROR] Stage '{stage_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            if 'trainer' in dir():
                trainer.cleanup()
                del trainer
            torch.cuda.empty_cache()
            continue

        trainer.cleanup()
        del trainer
        torch.cuda.empty_cache()

        checkpoint_path = f'runs/{cfg.name}/checkpoints/best_model.pt'

    print(f"\n{'='*60}")
    print(f"Pipeline '{pipeline_name}' completed! {len(stages)} stages executed.")
    print(f"{'='*60}")
    print(f"\nVer resultados en TensorBoard:")
    print(f"  tensorboard --logdir runs/")
    print()


def cmd_sweep(args):
    import yaml, json, time
    from datetime import datetime

    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)

    param_grid = {}
    non_sweep_keys = {'name', 'strategy', 'max_combinations', 'repetitions'}
    sweep_meta = {k: sweep_cfg.get('sweep', {}).get(k) for k in non_sweep_keys}

    for key, values in sweep_cfg.get('sweep', {}).items():
        if key not in non_sweep_keys and isinstance(values, list):
            param_grid[key] = values

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    max_combos = sweep_meta.get('max_combinations', 500)
    if len(combos) > max_combos:
        print(f"Limiting from {len(combos)} to {max_combos} combinations")
        combos = combos[:max_combos]

    print(f"\nSweep: {sweep_meta.get('name', args.config)}")
    print(f"Total combinations: {len(combos)}")
    print(f"Parameters: {keys}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []

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

        torch.manual_seed(cfg.seed + idx)
        start = time.time()

        label = f"{' × '.join(str(v) for v in combo)}"
        sys.stdout.write(f"[{idx+1}/{len(combos)}] {label} ... ")
        sys.stdout.flush()

        try:
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

            params = sum(p.numel() for p in model.parameters())
            writer = SummaryWriter(f'runs/{cfg.name}')
            best_loss, best_ppl = train_model(model, cfg, dataset_output, device, writer=writer)

            elapsed = time.time() - start
            all_results.append((combo, best_loss, best_ppl, params, elapsed, True))
            print(f"loss={best_loss:.4f}  ppl={best_ppl:.1f}  params={params/1e3:.1f}K  {elapsed:.1f}s")

        except Exception as e:
            import traceback
            elapsed = time.time() - start
            all_results.append((combo, 0, 0, 0, elapsed, False))
            print(f"ERROR: {e}")
            traceback.print_exc()
            print()

    # ---- Ranking ----
    passed = sum(1 for r in all_results if r[5])
    failed = len(all_results) - passed

    print(f"\n{'='*90}")
    print(f"  SWEEP RANKING — Best configurations by Test Perplexity")
    print(f"{'='*90}")
    print(f"  Total: {len(all_results)} | Passed: {passed} | Failed: {failed}")
    print()

    successful = [r for r in all_results if r[5]]
    successful.sort(key=lambda x: x[2])

    print(f"  {'Rank':>4s}  {'Config':<50s}  {'Loss':>8s}  {'PPL':>8s}  {'Params':>8s}  {'Time':>6s}")
    print(f"  {'-'*4}  {'-'*50}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    for rank, (combo, loss, ppl, params, elapsed, _) in enumerate(successful[:30], 1):
        label = ' × '.join(str(v) for v in combo)
        print(f"  {rank:4d}  {label:<50s}  {loss:8.4f}  {ppl:8.2f}  {params/1e3:7.1f}K  {elapsed:5.1f}s")
    if len(successful) > 30:
        print(f"  {'...':>4s}  ({len(successful) - 30} more not shown)")

    # ---- Export JSON ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = sweep_meta.get('name', os.path.splitext(os.path.basename(args.config))[0])
    out_path = f"sweep_{sweep_name}_{timestamp}.json"
    out = {
        'config': sweep_cfg,
        'results': [
            {
                'combo': list(combo),
                'test_loss': loss,
                'test_ppl': ppl,
                'params': params,
                'time_s': elapsed,
                'success': ok,
            }
            for combo, loss, ppl, params, elapsed, ok in all_results
        ],
        'ranking': [
            {
                'rank': r + 1,
                'combo': list(c), 'test_loss': l, 'test_ppl': p,
                'params': pa, 'time_s': e,
            }
            for r, (c, l, p, pa, e, _) in enumerate(successful)
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults exported to: {out_path}\n")


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

    def _add_stage_parser(name, help_text):
        p = subparsers.add_parser(name, help=help_text)
        p.add_argument('-c', '--config', type=str, default=None, help='Path to YAML config file')
        p.add_argument('-m', '--model-path', type=str, default=None, help='Path to pretrained model checkpoint')
        p.add_argument('overrides', nargs='*', help='Config overrides: key=value')
        return p

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

    _add_stage_parser('pretrain', 'Pre-training from scratch on large corpus')
    _add_stage_parser('sft', 'Supervised fine-tuning on instruction data')
    _add_stage_parser('dpo', 'Direct Preference Optimization alignment')
    _add_stage_parser('ppo', 'PPO alignment with reward model')
    _add_stage_parser('grpo', 'Group Relative Policy Optimization for reasoning')

    pipeline_parser = subparsers.add_parser('pipeline', help='Run multi-stage training pipeline')
    pipeline_parser.add_argument('-c', '--config', type=str, required=True, help='Path to pipeline YAML config')
    pipeline_parser.add_argument('-m', '--model-path', type=str, default=None, help='Path to initial model checkpoint')
    pipeline_parser.add_argument('overrides', nargs='*', help='Config overrides: key=value')

    sweep_parser = subparsers.add_parser('sweep', help='Run grid search sweep')
    sweep_parser.add_argument('-c', '--config', type=str, required=True, help='Path to sweep YAML config')

    list_parser = subparsers.add_parser('list', help='List available components')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'generate':
        cmd_generate(args)
    elif args.command == 'pretrain':
        cmd_pretrain(args)
    elif args.command == 'sft':
        cmd_sft(args)
    elif args.command == 'dpo':
        cmd_dpo(args)
    elif args.command == 'ppo':
        cmd_ppo(args)
    elif args.command == 'grpo':
        cmd_grpo(args)
    elif args.command == 'pipeline':
        cmd_pipeline(args)
    elif args.command == 'sweep':
        cmd_sweep(args)
    elif args.command == 'list':
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
