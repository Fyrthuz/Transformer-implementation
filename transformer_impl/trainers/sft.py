import torch
import torch.nn as nn
from .base import BaseTrainer


class SFTTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_output, device):
        super().__init__(cfg, model, dataset_output, device)
        self.loss_on_response_only = getattr(self.stage_cfg, 'loss_on_response_only', True)
        smoothing = getattr(self.stage_cfg, 'label_smoothing', getattr(cfg.training.loss, 'label_smoothing', 0.0))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=smoothing)

    def _get_labels(self, batch):
        if isinstance(batch, dict) and batch.get('labels') is not None:
            return batch['labels'].to(self.device)
        return None

    def train_step(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(self.device)
        elif isinstance(batch, torch.Tensor):
            input_ids = batch.to(self.device)
        else:
            input_ids = torch.tensor(batch, device=self.device)

        labels = self._get_labels(batch)

        mask = self.model.generate_causal_mask(input_ids.size(1), self.device)

        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            logits = self.model(input_ids, mask=mask)
            if labels is not None:
                loss = self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
            else:
                targets = input_ids[:, 1:].contiguous()
                loss = self.loss_fn(logits[:, :-1].reshape(-1, self.vocab_size), targets.reshape(-1))
        return loss

    def eval_step(self, batch):
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(self.device)
        else:
            input_ids = batch.to(self.device)

        labels = self._get_labels(batch)
        mask = self.model.generate_causal_mask(input_ids.size(1), self.device)
        logits = self.model(input_ids, mask=mask)

        if labels is not None:
            return self.loss_fn(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
        targets = input_ids[:, 1:].contiguous()
        return self.loss_fn(logits[:, :-1].reshape(-1, self.vocab_size), targets.reshape(-1))

    def generate_sample(self):
        if self.tokenizer is None:
            return None
        prompt = "Instruction: Write a short sentence about AI.\nResponse:"
        prompt_ids = self._tensor(self.tokenizer.encode(prompt)[:64])
        prompt_ids = prompt_ids.unsqueeze(0)
        with torch.no_grad():
            output = self.model.generate(prompt_ids, max_new_tokens=50, temperature=0.7,
                                         eos_token_id=self.pad_token_id, device=self.device)
        text = self.tokenizer.decode(output[0].tolist())
        return f"**Prompt**: {prompt}\n\n**Generated**: {text}"

    def _build_response_mask(self, input_ids):
        labels = input_ids.clone()
        labels[:, :-1] = -100
        return labels
