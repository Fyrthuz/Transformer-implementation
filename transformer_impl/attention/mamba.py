import torch
import torch.nn.functional as F
import math
from . import register_attention

@register_attention("mamba")
class MambaBlock(torch.nn.Module):
    def __init__(self, d_model, d_state=16, expand_factor=2, d_conv=4, dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand_factor
        self.d_inner = d_model * expand_factor
        self.d_conv = d_conv
        self.dropout = torch.nn.Dropout(dropout)

        self.in_proj = torch.nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = torch.nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=False,
        )
        self.x_proj = torch.nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = torch.nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = torch.nn.Linear(self.d_inner, d_model, bias=False)

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

        x_proj = x_proj.transpose(-1, -2)
        x_proj = self.conv1d(x_proj)[:, :, :T]
        x_proj = x_proj.transpose(-1, -2)
        x_proj = F.silu(x_proj)

        dt_x, B_proj, C_proj = self.x_proj(x_proj).split([self.d_inner, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_x))
        A = -torch.exp(self.A_log)
        deltaA = torch.exp(dt.unsqueeze(-1) * A)
        deltaB_u = dt.unsqueeze(-1) * B_proj.unsqueeze(2) * x_proj.unsqueeze(-1)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y = (h * C_proj[:, t].unsqueeze(1)).sum(-1) + self.D * x_proj[:, t]
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)
