import torch
import re
import copy
from .base import BaseTrainer
from transformer_impl.losses import grpo_loss


class GRPOTrainer(BaseTrainer):
    def __init__(self, cfg, model, dataset_output, device):
        super().__init__(cfg, model, dataset_output, device)
        self.group_size = self.stage_cfg.group_size
        self.epsilon = self.stage_cfg.epsilon
        self.kl_coef = self.stage_cfg.kl_coef
        self.max_gen_length = self.stage_cfg.max_gen_length
        self.reward_fn = self.stage_cfg.reward_fn
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
            raw = batch.get('input_ids', batch.get('text'))
            if isinstance(raw, torch.Tensor):
                prompts = raw.to(self.device)
            else:
                prompts = self._tensor(raw)
            answers = batch.get('answer', [None] * prompts.size(0))
            answer_nums = batch.get('answer_num', [None] * prompts.size(0))
        elif isinstance(batch, torch.Tensor):
            prompts = batch.to(self.device)
            answers = [None] * prompts.size(0)
            answer_nums = [None] * prompts.size(0)
        else:
            prompts = self._tensor(batch)
            answers = [None] * prompts.size(0)
            answer_nums = [None] * prompts.size(0)

        B = prompts.size(0)
        all_responses = []
        all_rewards = []
        all_ref_logps = []

        with torch.no_grad():
            for i in range(B):
                responses = self._generate_group(prompts[i:i+1])
                reward = self._compute_rewards(responses, answers[i] if isinstance(answers, list) else None,
                                                answer_nums[i] if isinstance(answer_nums, list) else None)
                ref_logps = self._batch_log_probs(self.ref_model, responses)
                all_responses.append(responses)
                all_rewards.append(reward)
                all_ref_logps.append(ref_logps)

        all_policy_logps = []
        for i in range(B):
            all_policy_logps.append(self._batch_log_probs(self.model, all_responses[i]))

        rewards_t = torch.stack(all_rewards)
        policy_logps_t = torch.stack(all_policy_logps)
        ref_logps_t = torch.stack(all_ref_logps)

        with torch.amp.autocast('cuda', enabled=self.scaler is not None):
            loss = grpo_loss(
                policy_logps_t, ref_logps_t, rewards_t,
                epsilon=self.epsilon, kl_coef=self.kl_coef,
            )
        return loss

    def _generate_group(self, prompt):
        self.model.eval()
        with torch.no_grad():
            responses = self.model.generate(
                prompt,
                max_new_tokens=self.max_gen_length,
                temperature=0.8,
                eos_token_id=self.pad_token_id,
                device=self.device,
            )
        self.model.train()
        return responses

    def _compute_rewards(self, responses, answer=None, answer_num=None):
        if self.reward_fn == "exact_match":
            return self._exact_match_reward(responses, answer_num)
        elif self.reward_fn == "format":
            return self._format_reward(responses)
        return torch.zeros(self.group_size, device=self.device)

    def _exact_match_reward(self, responses, answer_num):
        B = responses.size(0)
        rewards = torch.zeros(B, device=self.device)
        if answer_num is None:
            return rewards
        for i in range(B):
            text = self._decode(responses[i])
            predicted = self._extract_last_number(text)
            if predicted is not None and str(predicted) == str(answer_num):
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0
        return rewards

    def _format_reward(self, responses):
        B = responses.size(0)
        rewards = torch.zeros(B, device=self.device)
        for i in range(B):
            text = self._decode(responses[i])
            if "<thinking>" in text and "</thinking>" in text:
                rewards[i] = 0.5
            if self._extract_last_number(text) is not None:
                rewards[i] += 0.5
        return rewards

    def _extract_last_number(self, text):
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else None

    def _decode(self, token_ids):
        return self.dataset_output.tokenizer.decode(token_ids.tolist()) if self.dataset_output.tokenizer else ""

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
        return torch.tensor(0.0, device=self.device)
