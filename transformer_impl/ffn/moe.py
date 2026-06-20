import torch
import torch.nn.functional as F
from . import register_ffn

@register_ffn("moe")
class MixtureOfExperts(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = torch.nn.Linear(d_model, num_experts, bias=False)

        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model, d_ff),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(d_ff, d_model),
                torch.nn.Dropout(dropout),
            ) for _ in range(num_experts)
        ])

        self.aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.router(x_flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            indices_i = indices[:, i]
            weights_i = weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (indices_i == e_idx)
                if mask.any():
                    out[mask] += weights_i[mask] * self.experts[e_idx](x_flat[mask])

        self.aux_loss = self._load_balancing_loss(logits, indices)
        return out.view(B, T, D)

    def _load_balancing_loss(self, logits, indices):
        counts = torch.zeros(self.num_experts, device=logits.device)
        for e in range(self.num_experts):
            counts[e] = (indices == e).sum()
        frac = counts / counts.sum()
        router_probs = F.softmax(logits, dim=-1).mean(dim=0)
        return (frac * router_probs).sum() * self.num_experts

    def auxiliary_losses(self):
        return [self.aux_loss]
