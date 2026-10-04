"""Building blocks: selective state-space (Mamba) block, S4D layer, hurdle head and loss.

The selective scan is written in plain PyTorch (no custom CUDA kernel), so the same code runs on
Windows/CPU and on the AI-Lab GPUs. Token sequences here are short (<= 120 minutes, or ~19
patches), so the sequential scan is cheap.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_VECTOR_LIMIT = 3e8   # elements; above this, discretise step by step to save memory


_TINY = 1e-30          # state values below this are set to 0 (never subnormal -> no CPU slowdown)


def _flush(h):
    return torch.where(h.abs() < _TINY, torch.zeros_like(h), h)


def selective_scan(u, delta, A, B, C, D):
    """u, delta: (b, l, d); A: (d, n); B, C: (b, l, n); D: (d,). Returns (b, l, d).

    The state decays through exp(delta*A) and would otherwise underflow into subnormal floats,
    which makes CPU arithmetic up to ~100x slower (seen as random 'hangs' on laptops).
    The decay exponent is clamped and tiny state values are flushed to zero.
    """
    b, l, d = u.shape
    n = A.shape[1]
    h = u.new_zeros(b, d, n)
    ys = []
    if b * l * d * n <= _VECTOR_LIMIT:
        dA = torch.exp((delta.unsqueeze(-1) * A).clamp_min(-60.0))           # (b, l, d, n)
        dBu = _flush((delta * u).unsqueeze(-1) * B.unsqueeze(2))             # (b, l, d, n)
        for t in range(l):
            h = _flush(dA[:, t] * h + dBu[:, t])
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    else:
        for t in range(l):
            dA = torch.exp((delta[:, t, :, None] * A).clamp_min(-60.0))      # (b, d, n)
            dBu = delta[:, t, :, None] * B[:, t, None, :] * u[:, t, :, None]
            h = _flush(dA * h + dBu)
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=1) + u * D


class MambaBlock(nn.Module):
    """Mamba (Gu & Dao, 2023) selective SSM block.

    nonlinear=False gives the SAMBA simplification (Weng et al.): the SiLU activation between
    the causal convolution and the selective SSM is removed, which the SAMBA paper found to
    reduce overfitting on time series. The multiplicative gate is kept.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, nonlinear: bool = True):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.nonlinear = nonlinear
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.d_state = d_state

    def forward(self, x):                                     # (b, l, d_model)
        l = x.shape[1]
        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :l].transpose(1, 2)
        if self.nonlinear:
            u = F.silu(u)
        dbc = self.x_proj(u)
        dt, B, C = torch.split(dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = selective_scan(u.float(), delta.float(), A, B.float(), C.float(), self.D.float())
        y = y.to(x.dtype) * F.silu(z)
        return self.out_proj(y)


class ResidualMamba(nn.Module):
    def __init__(self, d_model, d_state, nonlinear, dropout=0.0, bidirectional=False, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = MambaBlock(d_model, d_state, expand=expand, nonlinear=nonlinear)
        self.bwd = MambaBlock(d_model, d_state, expand=expand, nonlinear=nonlinear) if bidirectional else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:                              # order-agnostic (variate) mixing
            y = 0.5 * (y + self.bwd(h.flip(1)).flip(1))
        return x + self.drop(y)


class S4DLayer(nn.Module):
    """Diagonal linear time-invariant SSM (S4D-Lin, Gu et al. 2022) computed as an FFT convolution."""

    def __init__(self, d_model: int, d_state: int = 32, dropout: float = 0.0):
        super().__init__()
        h, n = d_model, d_state // 2
        self.log_dt = nn.Parameter(torch.rand(h) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.C = nn.Parameter(torch.randn(h, n, 2) * 0.5 ** 0.5)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(h, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n).float().repeat(h, 1))
        self.D = nn.Parameter(torch.randn(h))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model))

    def kernel(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt[:, None]
        K = dtA[:, :, None] * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.0) / A
        return 2 * torch.einsum("hn,hnl->hl", C, torch.exp(K)).real

    def forward(self, x):                                     # (b, l, h)
        res = x
        x = self.norm(x)
        L = x.shape[1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            k = self.kernel(L)
            u = x.float().transpose(1, 2)
            y = torch.fft.irfft(torch.fft.rfft(u, n=2 * L) * torch.fft.rfft(k, n=2 * L), n=2 * L)[..., :L]
            y = y + u * self.D[:, None]
        return res + self.out(y.transpose(1, 2).to(res.dtype))


class AttnPool(nn.Module):
    """Learned-query attention pooling over a set of tokens (b, n, d) -> (b, d)."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.k = nn.Linear(d, d)

    def forward(self, x):
        a = torch.softmax(self.k(x) @ self.q / math.sqrt(x.shape[-1]), dim=1)
        return (a.unsqueeze(-1) * x).sum(1)


class HurdleHead(nn.Module):
    """Outputs logit P(rain), conditional wet amount and q90 (both in 0.1 mm units)."""

    def __init__(self, d, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 3))
        with torch.no_grad():   # start from a calibrated prior: ~2 % wet minutes, ~1 tip (0.1 mm) when wet
            self.net[-1].bias.copy_(torch.tensor([-4.0, 0.5, 0.0]))

    def forward(self, h):
        o = self.net(h)
        return o[:, 0], F.softplus(o[:, 1]), F.softplus(o[:, 2])


Y_SCALE = 10.0  # targets in 0.1 mm


def hurdle_loss(logit, amount, q90, y, w, weights: dict):
    ys = y * Y_SCALE
    wet = (y > 0).float()
    wsum = w.sum().clamp_min(1e-6)
    bce = (F.binary_cross_entropy_with_logits(logit, wet, reduction="none") * w).sum() / wsum
    wet_w = w * wet
    # Huber on the wet amount (0.1 mm units): robust to heavy showers, and on the same scale as the BCE
    amt = (F.huber_loss(amount, ys, reduction="none", delta=1.0) * wet_w).sum() / wet_w.sum().clamp_min(1e-6)
    d = ys - q90
    pin = (torch.maximum(0.9 * d, -0.1 * d) * w).sum() / wsum
    return weights["bce"] * bce + weights["amount"] * amt + weights["quantile"] * pin, {
        "bce": bce.item(), "amt": amt.item(), "pin": pin.item()}


def head_outputs(logit, amount, q90):
    p = torch.sigmoid(logit)
    return p, p * amount / Y_SCALE, q90 / Y_SCALE
