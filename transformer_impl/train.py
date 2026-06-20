import torch
import math
import time
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformer_impl.config import ExperimentConfig


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

    if cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    else:
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=lambda e: 1.0)

    own_writer = writer is None
    if own_writer:
        writer = SummaryWriter(f'runs/{model_cfg.name}')

    global_step = 0
    best_test_loss = float('inf')
    epochs_no_improve = 0
    patience = cfg.early_stop_patience

    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Attention: {model_cfg.model.attention.type}")
    print(f"FFN: {model_cfg.model.ffn.type}")
    print(f"Position: {model_cfg.model.position.type}")

    experiment_start_time = time.time()
    total_train_time = 0.0
    total_inference_time = 0.0
    total_inference_samples = 0

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

    for epoch in range(cfg.num_epochs):
        model.train()
        total_train_loss = 0
        epoch_start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            mask = model.generate_causal_mask(inputs.size(1), device)

            logits = model(inputs, mask=mask)
            main_loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
            aux_losses = model.auxiliary_losses()
            total_loss = main_loss + model_cfg.training.loss.moe_load_balance_coef * sum(aux_losses)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            total_train_loss += main_loss.item()
            writer.add_scalar('Train/Loss_step', main_loss.item(), global_step)
            global_step += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_train_loss = total_train_loss / len(train_loader)
        train_ppl = math.exp(avg_train_loss)

        model.eval()
        total_test_loss = 0
        infer_start = time.time()
        infer_batch_count = 0

        with torch.no_grad():
            for i, t_batch in enumerate(test_loader):
                t_batch = t_batch.to(device)
                t_inputs = t_batch[:, :-1]
                t_targets = t_batch[:, 1:]
                t_mask = model.generate_causal_mask(t_inputs.size(1), device)
                t_output = model(t_inputs, mask=t_mask)
                t_loss = loss_fn(t_output.reshape(-1, vocab_size), t_targets.reshape(-1))
                total_test_loss += t_loss.item()

                infer_batch_count += 1

                if i == 0 and epoch % 5 == 0:
                    preds = t_output.argmax(dim=-1)
                    clean_input = [tok for tok in t_inputs[0].tolist() if tok != pad_id]
                    clean_target = [tok for tok in t_targets[0].tolist() if tok != pad_id]
                    clean_preds = preds[0].tolist()[:len(clean_target)]
                    tokenizer = dataset_output.tokenizer
                    test_example = (
                        f"**Input**: {tokenizer.decode(clean_input)}  \n"
                        f"**Target**: {tokenizer.decode(clean_target)}  \n"
                        f"**Predicted**: {tokenizer.decode(clean_preds)}"
                    )
                    writer.add_text('Samples/Prediction_Test', test_example, epoch)

        infer_time = time.time() - infer_start
        total_inference_time += infer_time
        total_inference_samples += infer_batch_count * t_batch.size(0)
        infer_time_per_sample = infer_time / max(infer_batch_count * t_batch.size(0), 1) * 1000

        avg_test_loss = total_test_loss / len(test_loader)
        test_ppl = math.exp(avg_test_loss)

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
                print(f"  --> Early stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
                break

        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Test/Loss', avg_test_loss, epoch)
        writer.add_scalar('Train/Perplexity', train_ppl, epoch)
        writer.add_scalar('Test/Perplexity', test_ppl, epoch)
        writer.add_scalar('Params/Learning_Rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Time/Epoch_seconds', epoch_time, epoch)
        writer.add_scalar('Time/Inference_ms_per_sample', infer_time_per_sample, epoch)

        if device.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            writer.add_scalar('GPU/Memory_allocated_GB', mem_alloc, epoch)
            writer.add_scalar('GPU/Memory_reserved_GB', mem_reserved, epoch)

        writer.flush()

        timing_str = f" | Time: {epoch_time:.1f}s"
        if device.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
            timing_str += f" | GPU: {mem_alloc:.2f}GB"

        print(f"Epoch [{epoch+1:02d}/{cfg.num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.2f} | "
              f"Test Loss: {avg_test_loss:.4f} | Test PPL: {test_ppl:.2f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}{timing_str}")

    total_exp_time = time.time() - experiment_start_time
    avg_infer_per_sample = (total_inference_time / total_inference_samples * 1000) if total_inference_samples > 0 else 0

    print(f"\n  --> Total time: {total_exp_time:.1f}s | Train: {total_train_time:.1f}s | Inference: {total_inference_time:.1f}s")
    print(f"  --> Avg inference: {avg_infer_per_sample:.2f}ms/sample")

    writer.add_scalar('Time/Total_experiment_seconds', total_exp_time, 0)
    writer.add_scalar('Time/Avg_inference_ms_per_sample', avg_infer_per_sample, 0)

    writer.add_hparams(hparam_dict, {'hparam/test_loss': best_test_loss, 'hparam/test_perplexity': math.exp(best_test_loss)})

    if own_writer:
        writer.close()

    return best_test_loss, math.exp(best_test_loss)
