import os
import math
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer_impl.utils.lr_scheduler import get_scheduler
from transformer_impl.utils.checkpointing import save_checkpoint, resume_training
from transformer_impl.utils.logging import Logger


class BaseTrainer:
    def __init__(self, cfg, model, dataset_output, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.dataset_output = dataset_output
        self.pad_token_id = dataset_output.pad_token_id
        self.vocab_size = dataset_output.vocab_size
        self.tokenizer = dataset_output.tokenizer

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        self.stage_cfg = self._get_stage_cfg()
        self.accum_steps = self._resolve('gradient_accumulation_steps', 1)
        self.mixed_precision = self._resolve('mixed_precision', None)
        self.warmup_steps = self._resolve('warmup_steps', 0)
        self.max_steps = self._resolve('max_steps', 100000)
        self.save_steps = self._resolve('save_steps', 0)
        self.eval_steps = self._resolve('eval_steps', 0)
        self.log_type = self._resolve('logging', 'tensorboard')
        self.log_interval = self._resolve('log_interval', 10)
        self.save_total_limit = self._resolve('save_total_limit', 3)

        self.logger = Logger(log_dir=f'runs/{cfg.name}', name=cfg.name, log_type=self.log_type)
        self.scaler = torch.amp.GradScaler('cuda') if self.mixed_precision and device.type == 'cuda' else None

        self.ckpt_dir = f'runs/{cfg.name}/checkpoints'
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.scheduler = get_scheduler(
            cfg.training.scheduler, self.optimizer,
            self.warmup_steps, self.max_steps,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')

        self._print_config()

    def _get_stage_cfg(self):
        for key in ['pretrain', 'sft', 'dpo', 'ppo', 'grpo']:
            sc = getattr(self.cfg, key, None)
            if sc and getattr(sc, 'enabled', False):
                return sc
        return self.cfg.training

    def _resolve(self, name, default):
        train_val = getattr(self.cfg.training, name, None)
        stage_val = getattr(self.stage_cfg, name, None)
        if stage_val is not None and hasattr(self.stage_cfg, '__dataclass_fields__'):
            field = self.stage_cfg.__dataclass_fields__.get(name)
            if field is not None and stage_val == field.default:
                return train_val if train_val is not None else default
        return stage_val if stage_val is not None else (train_val if train_val is not None else default)

    def _print_config(self):
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Vocab size: {self.vocab_size}")
        if self.accum_steps > 1:
            print(f"Gradient accumulation: {self.accum_steps} steps")
        if self.scaler:
            print(f"Mixed precision: {self.mixed_precision}")

    def _make_loader(self, data, shuffle=True):
        return DataLoader(
            data, batch_size=self.cfg.training.batch_size, shuffle=shuffle,
            collate_fn=self._collate_batch,
        )

    def _collate_batch(self, batch):
        if not batch:
            return batch
        if isinstance(batch[0], dict):
            result = {}
            for key in batch[0]:
                values = [item[key] for item in batch]
                if isinstance(values[0], (list, tuple)):
                    max_len = max(len(v) for v in values)
                    pad_val = -100 if key == 'labels' else self.pad_token_id
                    padded = []
                    for v in values:
                        v_list = list(v[:max_len])
                        if len(v_list) < max_len:
                            v_list = v_list + [pad_val] * (max_len - len(v_list))
                        padded.append(v_list)
                    result[key] = torch.tensor(padded, dtype=torch.long)
                else:
                    result[key] = list(values)
            return result
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch)
        return torch.tensor(batch, dtype=torch.long)

    def train_step(self, batch):
        raise NotImplementedError

    def eval_step(self, batch):
        raise NotImplementedError

    def generate_sample(self):
        return None

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        train_data = self._get_train_data()
        test_data = self._get_test_data()
        train_loader = self._make_loader(train_data, shuffle=True)

        pbar = tqdm(total=self.max_steps, desc="Training", unit="step")
        train_loss_sum = 0
        train_batches = 0
        step_start = time.time()

        while self.global_step < self.max_steps:
            for batch in train_loader:
                if self.global_step >= self.max_steps:
                    break

                loss = self.train_step(batch)
                loss = loss / self.accum_steps

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss_sum += loss.item() * self.accum_steps
                train_batches += 1

                if (train_batches % self.accum_steps) == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)

                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    self.global_step += 1
                    pbar.update(1)

                    step_time = time.time() - step_start
                    avg_loss = train_loss_sum / max(train_batches, 1)
                    current_lr = self.scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                        grad=f"{grad_norm:.2f}",
                        step_time=f"{step_time:.2f}s"
                    )
                    step_start = time.time()

                    if self.global_step % self.log_interval == 0:
                        self.logger.log_scalar('Train/Loss', avg_loss, self.global_step)
                        self.logger.log_scalar('Train/LR', current_lr, self.global_step)
                        self.logger.log_scalar('Train/GradNorm', grad_norm, self.global_step)
                        self.logger.log_scalar('Train/StepTime_ms', step_time * 1000, self.global_step)
                        self.logger.log_scalar('Params/Total', sum(p.numel() for p in self.model.parameters()), self.global_step)
                        if self.device.type == 'cuda':
                            self.logger.log_scalar('GPU/Memory_allocated_GB',
                                torch.cuda.memory_allocated(self.device) / 1024**3, self.global_step)
                            self.logger.log_scalar('GPU/Memory_reserved_GB',
                                torch.cuda.memory_reserved(self.device) / 1024**3, self.global_step)
                        self.logger.flush()

                    if self.global_step % (self.log_interval * 10) == 0:
                        self.logger.log_gradient_histograms(self.model, self.global_step)
                        self.logger.flush()

                    if self.save_steps and self.global_step % self.save_steps == 0:
                        self._save_checkpoint()

                    if self.eval_steps and self.global_step % self.eval_steps == 0:
                        self._evaluate(test_data)

            self.epoch += 1

        pbar.close()
        self._evaluate(test_data)
        self._save_checkpoint(final=True)
        self.logger.close()

    def _evaluate(self, test_data):
        self.model.eval()
        loader = self._make_loader(test_data, shuffle=False)
        total_loss = 0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                loss = self.eval_step(batch)
                total_loss += loss.item()
                n_batches += 1
        avg = total_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 50)) if avg < 50 else float('inf')
        self.logger.log_scalar('Eval/Loss', avg, self.global_step)
        self.logger.log_scalar('Eval/Perplexity', ppl, self.global_step)
        print(f"\n  Eval [{self.global_step}] — Loss: {avg:.4f}, PPL: {ppl:.2f}")

        sample_text = self.generate_sample()
        if sample_text:
            self.logger.log_text('Samples/Generated', sample_text, self.global_step)

        if avg < self.best_metric:
            self.best_metric = avg
            best_path = os.path.join(self.ckpt_dir, 'best_model.pt')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'global_step': self.global_step,
                'eval_loss': avg,
                'eval_ppl': ppl,
            }, best_path)
            print(f"  --> New best model! Loss: {avg:.4f}, PPL: {ppl:.2f}")

        self.model.train()

    def _save_checkpoint(self, final=False):
        prefix = 'final' if final else f'step{self.global_step}'
        path = os.path.join(self.ckpt_dir, f'{prefix}.pt')
        save_checkpoint(
            path, self.model, self.optimizer, self.scheduler,
            self.epoch, self.global_step,
        )
        if not final:
            ckpts = sorted([f for f in os.listdir(self.ckpt_dir) if f.startswith('step') and f.endswith('.pt')])
            while len(ckpts) > self.save_total_limit:
                os.remove(os.path.join(self.ckpt_dir, ckpts.pop(0)))
            print(f"\n  --> Saved checkpoint at step {self.global_step}")

    def _get_train_data(self):
        return self.dataset_output.train_data

    def _get_test_data(self):
        return self.dataset_output.test_data

    def _tensor(self, data, dtype=torch.long):
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=self.device)
        return torch.tensor(data, dtype=dtype, device=self.device)

    def cleanup(self):
        if self.logger:
            self.logger.flush()
            self.logger.close()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.logger = None
        if hasattr(self, 'ref_model'):
            del self.ref_model
        self.model = self.model.cpu()
        torch.cuda.empty_cache()

    def _log_probs(self, model, input_ids, labels=None):
        was_training = model.training
        model.eval()
        with torch.no_grad():
            mask = model.generate_causal_mask(input_ids.size(1), self.device)
            logits = model(input_ids, mask=mask)
            log_probs = torch.log_softmax(logits, dim=-1)
        if was_training:
            model.train()
        if labels is not None:
            return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return log_probs
