import torch
import math
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformer_impl.config import ExperimentConfig
from transformer_impl.utils.lr_scheduler import get_scheduler
from transformer_impl.utils.checkpointing import save_checkpoint, load_checkpoint, resume_training
from transformer_impl.utils.logging import Logger


class TextDataset(Dataset):
    def __init__(self, dataset, pad_token_id, max_len):
        self.data = []
        for example in dataset:
            tokens = example['text']
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            padded = tokens + [pad_token_id] * (max_len - len(tokens))
            self.data.append(torch.tensor(padded, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_loss_fn(cfg, pad_token_id):
    loss_cfg = cfg.training.loss
    ignore_idx = pad_token_id if loss_cfg.ignore_index == "auto" else int(loss_cfg.ignore_index)

    if loss_cfg.type == "cross_entropy":
        return nn.CrossEntropyLoss(
            ignore_index=ignore_idx,
            label_smoothing=loss_cfg.label_smoothing,
        )
    elif loss_cfg.type == "nll":
        return nn.NLLLoss(ignore_index=ignore_idx)
    elif loss_cfg.type == "mse":
        return nn.MSELoss()
    elif loss_cfg.type == "focal":
        gamma = loss_cfg.focal_gamma or 2.0
        class FocalLoss(nn.Module):
            def __init__(self, gamma, ignore_idx):
                super().__init__()
                self.gamma = gamma
                self.ignore_idx = ignore_idx

            def forward(self, logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, ignore_index=self.ignore_idx, reduction='none')
                pt = torch.exp(-ce)
                focal = ((1 - pt) ** self.gamma * ce).mean()
                return focal
        return FocalLoss(gamma, ignore_idx)
    else:
        return nn.CrossEntropyLoss(ignore_index=ignore_idx)


def _resolve_accum(cfg):
    val = getattr(cfg, 'gradient_accumulation_steps', None)
    return val if val is not None else 1


def train_model(model, model_cfg: ExperimentConfig, dataset_output, device, writer=None):
    cfg = model_cfg.training
    pad_id = dataset_output.pad_token_id
    vocab_size = dataset_output.vocab_size

    train_dataset = TextDataset(dataset_output.train_data, pad_id, model_cfg.dataset.max_seq_len)
    test_dataset = TextDataset(dataset_output.test_data, pad_id, model_cfg.dataset.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    loss_fn = build_loss_fn(model_cfg, pad_id)

    warmup_steps = cfg.warmup_steps if cfg.warmup_steps is not None else 0
    max_steps = cfg.max_steps or (cfg.num_epochs * len(train_loader))
    steps_per_epoch = len(train_loader)
    total_steps = max_steps
    accum_steps = _resolve_accum(cfg)
    mixed_precision = cfg.mixed_precision
    save_steps = cfg.save_steps if cfg.save_steps is not None else 0
    eval_steps = cfg.eval_steps if cfg.eval_steps is not None else 0
    log_type = cfg.logging if cfg.logging is not None else "tensorboard"
    resume_from = cfg.resume_from

    scheduler = get_scheduler(cfg.scheduler, optimizer, warmup_steps, total_steps, cfg.num_epochs, steps_per_epoch)

    own_writer = writer is None
    if own_writer:
        writer_obj = Logger(log_dir=f'runs/{model_cfg.name}', log_type=log_type)
    else:
        writer_obj = Logger(log_dir=None, log_type="tensorboard")
        writer_obj.writer = writer

    global_step = 0
    start_epoch = 0
    best_test_loss = float('inf')
    epochs_no_improve = 0
    patience = cfg.early_stop_patience

    scaler = torch.amp.GradScaler('cuda') if mixed_precision and device.type == 'cuda' else None

    if resume_from:
        start_epoch, global_step, _ = resume_training(resume_from, model, optimizer, scheduler, device)

    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Attention: {model_cfg.model.attention.type}")
    print(f"FFN: {model_cfg.model.ffn.type}")
    print(f"Position: {model_cfg.model.position.type}")
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps")
    if scaler:
        print(f"Mixed precision: {mixed_precision}")

    experiment_start_time = time.time()
    total_train_time = 0.0

    hparam_dict = {
        'attention': model_cfg.model.attention.type,
        'ffn': model_cfg.model.ffn.type,
        'position': model_cfg.model.position.type,
        'd_model': model_cfg.model.d_model,
        'num_layers': model_cfg.model.num_layers,
        'num_heads': model_cfg.model.attention.num_heads,
        'batch_size': cfg.batch_size,
        'learning_rate': cfg.learning_rate,
        'weight_decay': cfg.weight_decay,
        'num_epochs': cfg.num_epochs,
        'dataset': model_cfg.dataset.name,
        'tokenization': model_cfg.dataset.tokenization,
        'loss_type': cfg.loss.type,
    }

    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        train_loss_sum = 0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= total_steps:
                break

            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            mask = model.generate_causal_mask(inputs.size(1), device)

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                logits = model(inputs, mask=mask)
                main_loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
                aux_losses = model.auxiliary_losses()
                total_loss = main_loss + model_cfg.training.loss.moe_load_balance_coef * sum(aux_losses)
                total_loss = total_loss / accum_steps

            if scaler:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            train_loss_sum += main_loss.item()
            train_batches += 1

            if (batch_idx + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                writer_obj.log_scalar('Train/Loss_step', main_loss.item(), global_step)
                global_step += 1

                if save_steps and global_step % save_steps == 0:
                    ckpt_path = f'checkpoint_{model_cfg.name}_step{global_step}.pt'
                    save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, global_step,
                                    {'train_loss': train_loss_sum / max(train_batches, 1)},
                                    hparam_dict)
                    print(f"  --> Saved checkpoint: {ckpt_path}")

                if eval_steps and global_step % eval_steps == 0:
                    _run_eval(model, test_loader, loss_fn, vocab_size, device, model_cfg, writer_obj, epoch, dataset_output)

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_train_loss = train_loss_sum / max(train_batches, 1)
        train_ppl = math.exp(avg_train_loss) if avg_train_loss < 50 else float('inf')

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for t_batch in test_loader:
                t_batch = t_batch.to(device)
                t_inputs = t_batch[:, :-1]
                t_targets = t_batch[:, 1:]
                t_mask = model.generate_causal_mask(t_inputs.size(1), device)
                t_output = model(t_inputs, mask=t_mask)
                t_loss = loss_fn(t_output.reshape(-1, vocab_size), t_targets.reshape(-1))
                total_test_loss += t_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_ppl = math.exp(avg_test_loss) if avg_test_loss < 50 else float('inf')

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'attention': model_cfg.model.attention.type,
                    'ffn': model_cfg.model.ffn.type,
                    'position': model_cfg.model.position.type,
                    'd_model': model_cfg.model.d_model,
                    'num_layers': model_cfg.model.num_layers,
                    'num_heads': model_cfg.model.attention.num_heads,
                },
                'vocab_size': vocab_size,
                'test_loss': best_test_loss,
                'test_perplexity': test_ppl,
            }, 'best_model.pt')
            print(f"  --> New best model! Test Loss: {avg_test_loss:.4f}, PPL: {test_ppl:.2f}")
        elif patience > 0:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  --> Early stopping after {epoch+1} epochs")
                break

        writer_obj.log_scalar('Train/Loss', avg_train_loss, epoch)
        writer_obj.log_scalar('Test/Loss', avg_test_loss, epoch)
        writer_obj.log_scalar('Train/Perplexity', train_ppl, epoch)
        writer_obj.log_scalar('Test/Perplexity', test_ppl, epoch)
        writer_obj.log_scalar('Params/Learning_Rate', scheduler.get_last_lr()[0], epoch)
        writer_obj.log_scalar('Time/Epoch_seconds', epoch_time, epoch)

        if device.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            writer_obj.log_scalar('GPU/Memory_allocated_GB', mem_alloc, epoch)
            writer_obj.log_scalar('GPU/Memory_reserved_GB', mem_reserved, epoch)

        writer_obj.flush()

        timing_str = f" | Time: {epoch_time:.1f}s"
        if device.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
            timing_str += f" | GPU: {mem_alloc:.2f}GB"

        print(f"Epoch [{epoch+1:02d}/{cfg.num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test PPL: {test_ppl:.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}{timing_str}")

    total_exp_time = time.time() - experiment_start_time
    print(f"\n  --> Total time: {total_exp_time:.1f}s")

    writer_obj.log_scalar('Time/Total_experiment_seconds', total_exp_time, 0)
    writer_obj.log_hparams(hparam_dict, {'hparam/test_loss': best_test_loss, 'hparam/test_perplexity': math.exp(best_test_loss) if best_test_loss < 50 else 0})

    if own_writer:
        writer_obj.close()

    return best_test_loss, math.exp(best_test_loss) if best_test_loss < 50 else float('inf')


def _run_eval(model, test_loader, loss_fn, vocab_size, device, model_cfg, writer_obj, epoch, dataset_output):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for t_batch in test_loader:
            t_batch = t_batch.to(device)
            t_inputs = t_batch[:, :-1]
            t_targets = t_batch[:, 1:]
            t_mask = model.generate_causal_mask(t_inputs.size(1), device)
            t_output = model(t_inputs, mask=t_mask)
            t_loss = loss_fn(t_output.reshape(-1, vocab_size), t_targets.reshape(-1))
            total_loss += t_loss.item()
    avg = total_loss / len(test_loader)
    ppl = math.exp(avg) if avg < 50 else float('inf')
    writer_obj.log_scalar('Eval/Loss', avg, epoch)
    writer_obj.log_scalar('Eval/Perplexity', ppl, epoch)
    model.train()
