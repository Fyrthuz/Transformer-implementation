import torch
import torch.nn.functional as F
import math
from . import register_attention

@register_attention("ssm")
class SSMBlock(torch.nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dropout = torch.nn.Dropout(dropout)

        self.in_proj = torch.nn.Linear(d_model, d_model * 2)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).repeat(d_model, 1)
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.B = torch.nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = torch.nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = torch.nn.Parameter(torch.ones(d_model))
        self.dt = torch.nn.Parameter(torch.log(torch.ones(d_model) * 0.001))

    def forward(self, x, mask=None):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask,
                use_reentrant=False, preserve_rng_state=False,
            )
        return self._forward_impl(x, mask)

    def _forward_impl(self, x, mask=None):
        B, T, D = x.shape
        inp = self.in_proj(x)
        x_proj, z = inp.chunk(2, dim=-1)
        x_proj = F.silu(x_proj)

        dt = F.softplus(self.dt).view(1, 1, D)
        A = -torch.exp(self.A_log)

        deltaA = torch.exp(dt.unsqueeze(-1) * A)
        deltaB_u = dt.unsqueeze(-1) * self.B * x_proj.unsqueeze(-1)

        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            h = deltaA[:, :, t] * h + deltaB_u[:, t]
            y = (h * self.C).sum(-1) + self.D * x_proj[:, t]
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)
