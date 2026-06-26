import torch
import torch.nn as nn
from .base import BaseTrainer


class PreTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_output, device):
        super().__init__(cfg, model, dataset_output, device)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.pad_token_id,
            label_smoothing=cfg.training.loss.label_smoothing,
        )

    def _extract_input_ids(self, batch):
        if isinstance(batch, dict):
            return batch.get('input_ids', batch.get('text')).to(self.device)
        return batch.to(self.device)

    def train_step(self, batch):
        tokens = self._extract_input_ids(batch)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        mask = self.model.generate_causal_mask(inputs.size(1), self.device)

        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            logits = self.model(inputs, mask=mask)
            loss = self.loss_fn(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
            aux = self.model.auxiliary_losses()
            if aux:
                loss = loss + self.cfg.training.loss.moe_load_balance_coef * sum(aux)
        return loss

    def eval_step(self, batch):
        tokens = self._extract_input_ids(batch)
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        mask = self.model.generate_causal_mask(inputs.size(1), self.device)
        logits = self.model(inputs, mask=mask)
        return self.loss_fn(logits.reshape(-1, self.vocab_size), targets.reshape(-1))

    def generate_sample(self):
        if self.tokenizer is None:
            return None
        prompt_ids = self._tensor([[self.pad_token_id]])
        with torch.no_grad():
            output = self.model.generate(prompt_ids, max_new_tokens=50, temperature=0.7,
                                         eos_token_id=self.pad_token_id, device=self.device)
        text = self.tokenizer.decode(output[0].tolist())
        return f"**Generated sample**: {text}"
