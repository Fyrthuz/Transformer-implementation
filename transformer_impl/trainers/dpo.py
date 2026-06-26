import torch
import copy
from .base import BaseTrainer
from transformer_impl.losses import dpo_loss


class DPOTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_output, device):
        super().__init__(cfg, model, dataset_output, device)
        self.beta = self.stage_cfg.beta
        self.ref_model = self._build_reference_model()

    def _build_reference_model(self):
        ref_model = copy.deepcopy(self.model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        return ref_model

    def _get_train_data(self):
        return self.dataset_output.train_data

    def train_step(self, batch):
        if isinstance(batch, dict):
            prompt_ids = batch['prompt']
            chosen_ids = batch['chosen']
            rejected_ids = batch['rejected']
        elif isinstance(batch, (list, tuple)):
            prompt_ids, chosen_ids, rejected_ids = batch
        else:
            prompt_ids = chosen_ids = rejected_ids = batch
        prompt = self._tensor(prompt_ids)
        chosen = self._tensor(chosen_ids)
        rejected = self._tensor(rejected_ids)

        policy_chosen_logps = self._batch_log_probs(self.model, chosen)
        policy_rejected_logps = self._batch_log_probs(self.model, rejected)

        with torch.no_grad():
            ref_chosen_logps = self._batch_log_probs(self.ref_model, chosen)
            ref_rejected_logps = self._batch_log_probs(self.ref_model, rejected)

        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            loss = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                self.beta,
            )
        return loss

    def _batch_log_probs(self, model, input_ids):
        was_training = model.training
        model.eval()
        mask = model.generate_causal_mask(input_ids.size(1), self.device)
        logits = model(input_ids, mask=mask)
        if was_training:
            model.train()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = input_ids[:, 1:]
        return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(-1)

    def eval_step(self, batch):
        chosen = self._tensor(batch['chosen'])
        rejected = self._tensor(batch['rejected'])
        policy_chosen_logps = self._batch_log_probs(self.model, chosen)
        policy_rejected_logps = self._batch_log_probs(self.model, rejected)
        with torch.no_grad():
            ref_chosen_logps = self._batch_log_probs(self.ref_model, chosen)
            ref_rejected_logps = self._batch_log_probs(self.ref_model, rejected)
        return dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            self.beta,
        )
