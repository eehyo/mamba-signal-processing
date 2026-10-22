"""Inspectable Mamba-1 mixer, adapted from John Ma's mamba-minimal (Apache-2.0).

MODIFIED for this course: classification-facing API, stable dt initialization,
constant-parameter SSM, elapsed-time factors, right-padding masks, recordings,
float64 support and convolution utilities. See NOTICE and third_party/.

Shapes: b=batch, l=time, d=d_model, h=expand*d, n=d_state, r=dt_rank.
This is an educational sequential scan, not the fused mamba_ssm implementation.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def valid_mask(x: torch.Tensor, mask=None) -> torch.Tensor:
    """Require nonempty sequences with valid tokens followed by right padding.

    Missing observations must be compressed into an irregular sequence first;
    holes in this mask do NOT represent elapsed physical time.
    """
    if x.ndim != 3 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError("x must be a nonempty floating tensor [batch, time, channels]")
    if mask is None:
        return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    mask = torch.as_tensor(mask, device=x.device)
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(x.shape[:2]):
        raise ValueError("mask must be bool [batch, time]")
    if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("mask must contain a nonempty prefix followed by right padding")
    return mask


def zero_pad(x, mask):
    # where, not multiplication: NaN * 0 is still NaN.
    return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))


def time_factors(dt_mult, x, mask):
    """Positive factors on observed tokens; padding is assigned a factor of zero."""
    if dt_mult is None:
        return mask.to(x.dtype).unsqueeze(-1)
    dt = torch.as_tensor(dt_mult, dtype=x.dtype, device=x.device)
    if dt.ndim == 0:
        dt = dt.expand(x.shape[:2])
    elif dt.ndim == 1 and dt.shape[0] == x.shape[1]:
        dt = dt[None, :].expand(x.shape[:2])
    elif dt.ndim == 3 and dt.shape[-1] == 1:
        dt = dt.squeeze(-1)
    if tuple(dt.shape) != tuple(x.shape[:2]):
        raise ValueError("dt_mult must be scalar, [time], [batch,time], or [batch,time,1]")
    if not torch.isfinite(dt[mask]).all() or (dt[mask] <= 0).any():
        raise ValueError("dt_mult must be finite and positive on valid tokens")
    return torch.where(mask, dt, torch.zeros_like(dt)).unsqueeze(-1)


def causal_conv(u: torch.Tensor, kernel: torch.Tensor, method="direct"):
    """Per-channel linear causal convolution; u [b,l,h], kernel [k,h].

    The returned first l samples correspond to a zero initial state. The kernel
    may include D at lag zero. No activation, gate, projection, or residual here.
    """
    if u.ndim != 3 or kernel.ndim != 2 or kernel.shape[1] != u.shape[2]:
        raise ValueError("expected u [b,l,h] and kernel [k,h]")
    if u.shape[1] < 1 or kernel.shape[0] < 1:
        raise ValueError("sequence and kernel must be nonempty")
    if method == "direct":
        # conv1d is cross-correlation: reverse the lag-indexed kernel.
        v = F.conv1d(F.pad(u.transpose(1, 2), (kernel.shape[0] - 1, 0)),
                     kernel.flip(0).T.unsqueeze(1), groups=u.shape[2])
        return v.transpose(1, 2)
    if method == "fft":
        full_len = u.shape[1] + kernel.shape[0] - 1
        n_fft = 1 << (full_len - 1).bit_length()
        v = torch.fft.rfft(u, n=n_fft, dim=1)
        k = torch.fft.rfft(kernel, n=n_fft, dim=0)
        return torch.fft.irfft(v * k[None], n=n_fft, dim=1)[:, :u.shape[1]]
    raise ValueError("method must be direct or fft")


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, d_model=32, d_state=16, expand=2, dt_rank="auto",
                 d_conv=4, selective=True, dt_min=0.001, dt_max=0.1):
        super().__init__()
        if min(d_model, d_state, expand, d_conv) < 1:
            raise ValueError("all model dimensions must be positive")
        if not (0 < dt_min <= dt_max < float("inf")):
            raise ValueError("require 0 < dt_min <= dt_max < infinity")
        self.d_model, self.d_state = d_model, d_state
        self.d_inner, self.d_conv = expand * d_model, d_conv
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        if self.dt_rank < 1:
            raise ValueError("dt_rank must be positive")
        self.selective = bool(selective)
        h, n = self.d_inner, d_state
        self.in_proj = nn.Linear(d_model, 2 * h, bias=False)
        self.conv1d = nn.Conv1d(h, h, d_conv, groups=h, padding=d_conv - 1, bias=True)
        self.A_log = nn.Parameter(torch.arange(1, n + 1).float().log().repeat(h, 1))
        self.D = nn.Parameter(torch.ones(h))
        dt = torch.exp(torch.empty(h).uniform_(math.log(dt_min), math.log(dt_max)))
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        if self.selective:
            self.x_proj = nn.Linear(h, self.dt_rank + 2 * n, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, h, bias=True)
            nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_softplus)
        else:
            # B,C retain the same across-channel sharing as selective B_t,C_t.
            # These learned constants are independent of both content and time.
            self.dt_bias = nn.Parameter(inv_softplus)
            self.B_const = nn.Parameter(torch.randn(n) / math.sqrt(n))
            self.C_const = nn.Parameter(torch.randn(n) / math.sqrt(n))
        self.out_proj = nn.Linear(h, d_model, bias=False)
        self.register_buffer("delta_scale", torch.tensor(1.0))
        self.record = {}
        self._record_enabled = self._record_states = False

    @property
    def A(self):
        return -self.A_log.exp()

    def set_delta_scale(self, s):
        s = float(s)
        if not math.isfinite(s) or s <= 0:
            raise ValueError("delta scale must be positive and finite")
        self.delta_scale.fill_(s)
        return self

    def set_record(self, enabled=True, states=False):
        """Overwrite a detached CPU record on each forward; do not accumulate batches."""
        self._record_enabled, self._record_states = bool(enabled), bool(states)
        self.record = {}
        return self

    def ssm(self, u, dt_mult=None, mask=None):
        """Only the SSM sublayer. Input [b,l,h], output [b,l,h].

        h_t = exp(delta_t*A)*h_(t-1) + delta_t*B_t*u_t
        y_t = sum_n(C_t*h_t) + D*u_t
        This uses mamba-minimal's first-order input term, NOT exact ZOH for B.
        """
        mask = valid_mask(u, mask)
        if u.shape[-1] != self.d_inner:
            raise ValueError("ssm expects d_inner channels, after convolution/SiLU")
        u = zero_pad(u, mask)
        b, l, h = u.shape
        if self.selective:
            raw_dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], -1)
            delta = F.softplus(self.dt_proj(raw_dt))
        else:
            delta = F.softplus(self.dt_bias)[None, None, :].expand(b, l, -1)
            B = self.B_const[None, None, :].expand(b, l, -1)
            C = self.C_const[None, None, :].expand(b, l, -1)
        delta = delta * self.delta_scale * time_factors(dt_mult, u, mask)
        A = self.A
        state = u.new_zeros(b, h, self.d_state)
        outputs, states = [], []
        # A sequential scan is intentional: every state update is inspectable.
        for t in range(l):
            dt = delta[:, t, :, None]                  # [b,h,1]
            candidate = (dt * A).exp() * state + dt * B[:, t, None, :] * u[:, t, :, None]
            state = torch.where(mask[:, t, None, None], candidate, state)
            y = (state * C[:, t, None, :]).sum(-1) + self.D * u[:, t]
            outputs.append(torch.where(mask[:, t, None], y, torch.zeros_like(y)))
            if self._record_enabled and self._record_states:
                states.append(state.detach().cpu())
        if self._record_enabled:
            self.record = {key: val.detach().cpu() for key, val in
                           {"delta": delta, "A": A, "B": B, "C": C, "mask": mask}.items()}
            if states:
                self.record["states"] = torch.stack(states, dim=1)  # [b,l,h,n]
        return torch.stack(outputs, dim=1)

    def forward(self, x, dt_mult=None, mask=None):
        mask = valid_mask(x, mask)
        x = zero_pad(x, mask)                          # [b,l,d]
        u, z = self.in_proj(x).chunk(2, dim=-1)        # each [b,l,h]
        u = self.conv1d(u.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        u = zero_pad(F.silu(u), mask)                  # short causal convolution
        y = self.ssm(u, dt_mult=dt_mult, mask=mask)    # selective recurrence
        return zero_pad(self.out_proj(y * F.silu(z)), mask)

    def lti_kernel(self, L, include_skip=True):
        """Return [L,h] SSM impulse response, h_-1=0, constant delta_scale.

        K[k,h] = sum_n C[n] exp(k*delta[h]*A[h,n]) delta[h] B[n].
        D[h] is added only at k=0 by default. This is NOT a whole-block kernel.
        Per-time-step dt_mult breaks time invariance and is not represented here.
        """
        if self.selective:
            raise ValueError("lti_kernel requires selective=False")
        if not isinstance(L, int) or L < 1:
            raise ValueError("L must be a positive integer")
        dt = F.softplus(self.dt_bias) * self.delta_scale
        lag = torch.arange(L, dtype=dt.dtype, device=dt.device)[:, None, None]
        kernel = ((lag * dt[None, :, None] * self.A).exp()
                  * dt[None, :, None] * self.B_const * self.C_const).sum(-1)
        if include_skip:
            impulse = torch.zeros_like(kernel)
            impulse[0] = self.D
            kernel = kernel + impulse
        return kernel
