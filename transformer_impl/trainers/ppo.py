import torch
import torch.nn.functional as F
import copy
from .base import BaseTrainer
from transformer_impl.losses import ppo_loss_with_value


class PPOTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_output, device):
        model.add_value_head()
        super().__init__(cfg, model, dataset_output, device)
        self.kl_coef = self.stage_cfg.kl_coef
        self.clip_range = self.stage_cfg.clip_range
        self.vf_coef = self.stage_cfg.vf_coef
        self.max_gen_length = self.stage_cfg.max_gen_length
        self.ref_model = self._build_reference_model()

    def _build_reference_model(self):
        ref_model = copy.deepcopy(self.model)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        return ref_model

    def _get_train_data(self):
        return self.dataset_output.train_data

    def _reward_model_score(self, responses):
        return torch.randn(responses.size(0), device=self.device)

    def train_step(self, batch):
        if isinstance(batch, dict) and 'prompt' in batch:
            prompts = self._tensor(batch['prompt'])
        elif isinstance(batch, torch.Tensor):
            prompts = batch.to(self.device)
        else:
            prompts = self._tensor(batch)

        with torch.no_grad():
            responses = self.model.generate(
                prompts,
                max_new_tokens=self.max_gen_length,
                temperature=0.7,
                eos_token_id=self.pad_token_id,
                device=self.device,
            )
            rewards = self._reward_model_score(responses)
            old_log_probs = self._batch_log_probs(self.model, responses)
            ref_log_probs = self._batch_log_probs(self.ref_model, responses)
            kl_penalty = self.kl_coef * (ref_log_probs - old_log_probs)
            adjusted_rewards = rewards + kl_penalty
            values = self.model.forward_value(responses)
            returns = adjusted_rewards
            advantages = adjusted_rewards - values.squeeze(-1)

        new_log_probs = self._batch_log_probs(self.model, responses)
        new_values = self.model.forward_value(responses)

        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            loss = ppo_loss_with_value(
                new_log_probs, old_log_probs, advantages,
                new_values.squeeze(-1), returns,
                epsilon=self.clip_range, vf_coef=self.vf_coef,
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
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return gathered.sum(-1)

    def eval_step(self, batch):
        if isinstance(batch, dict) and 'prompt' in batch:
            input_ids = self._tensor(batch['prompt'])
        elif isinstance(batch, torch.Tensor):
            input_ids = batch.to(self.device)
        else:
            input_ids = self._tensor(batch)
        with torch.no_grad():
            responses = self.model.generate(input_ids, max_new_tokens=50, temperature=0.7, device=self.device)
        return -torch.tensor(0.0, device=self.device)
