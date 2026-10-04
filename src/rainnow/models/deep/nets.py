"""Sequence networks. Input x: (batch, window, channels). Output: hurdle head (logit, amount, q90).

* samba    SAMBA (Weng et al., "Simplified Mamba with disentangled dependency encoding for long-term
           time series forecasting"): patch tokens, Mamba blocks *without* the nonlinear activation,
           and disentangled encoding: a temporal Mamba per variate (channel-independent), followed by a
           bidirectional Mamba across variates (cross-variate dependencies, order-agnostic).
* mamba    plain Mamba over per-minute tokens (all channels embedded per minute)
* s4d      linear time-invariant diagonal SSM (S4D) over per-minute tokens
* gru      2-layer GRU
* tcn      dilated causal convolutions
* patchtst PatchTST-style Transformer with the same patching/pooling scaffold as SAMBA
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AttnPool, HurdleHead, ResidualMamba, S4DLayer


class _PatchScaffold(nn.Module):
    """Channel-independent patching shared by SAMBA and PatchTST."""

    def __init__(self, n_ch, window, d, patch, stride):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.n_patch = (window - patch) // stride + 1
        self.embed = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d))
        self.ch_emb = nn.Parameter(torch.zeros(1, n_ch, d))

    def tokens(self, x):                                      # (b, L, C) -> (b*C, P, d)
        b, L, C = x.shape
        start = L - (self.n_patch - 1) * self.stride - self.patch   # align the last patch to "now"
        x = x[:, start:, :].transpose(1, 2)                    # (b, C, L')
        p = x.unfold(-1, self.patch, self.stride)              # (b, C, P, patch)
        return (self.embed(p) + self.pos.unsqueeze(1)).reshape(b * C, self.n_patch, -1)


class SAMBA(nn.Module):
    def __init__(self, n_ch, window, d_model=64, n_layers=2, d_state=16, patch=12, stride=6, dropout=0.1,
                 nonlinear=False, disentangled=True, expand=2):
        super().__init__()
        self.scaf = _PatchScaffold(n_ch, window, d_model, patch, stride)
        self.temporal = nn.ModuleList([ResidualMamba(d_model, d_state, nonlinear, dropout, expand=expand)
                                       for _ in range(n_layers)])
        self.t_norm = nn.LayerNorm(d_model)
        self.t_out = nn.Linear(self.scaf.n_patch * d_model, d_model)
        self.disentangled = disentangled
        if disentangled:
            self.variate = nn.ModuleList([ResidualMamba(d_model, d_state, nonlinear, dropout, bidirectional=True, expand=expand)
                                          for _ in range(n_layers)])
        self.pool = AttnPool(d_model)
        self.last = nn.Linear(n_ch, d_model)      # last-value anchor: the current minute, un-patched
        self.head = HurdleHead(d_model, dropout)

    def forward(self, x):
        b, L, C = x.shape
        h = self.scaf.tokens(x)
        for blk in self.temporal:
            h = blk(h)
        h = self.t_out(self.t_norm(h).reshape(b, C, -1)) + self.scaf.ch_emb   # (b, C, d) variate tokens
        if self.disentangled:
            for blk in self.variate:
                h = blk(h)
        return self.head(self.pool(h) + self.last(x[:, -1]))


class PatchTST(nn.Module):
    def __init__(self, n_ch, window, d_model=64, n_layers=2, patch=12, stride=6, dropout=0.1, n_heads=4, **_):
        super().__init__()
        self.scaf = _PatchScaffold(n_ch, window, d_model, patch, stride)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.t_out = nn.Linear(self.scaf.n_patch * d_model, d_model)
        self.pool = AttnPool(d_model)
        self.last = nn.Linear(n_ch, d_model)      # same last-value anchor as SAMBA (fair comparison)
        self.head = HurdleHead(d_model, dropout)

    def forward(self, x):
        b, L, C = x.shape
        h = self.enc(self.scaf.tokens(x))
        h = self.t_out(h.reshape(b, C, -1)) + self.scaf.ch_emb
        return self.head(self.pool(h) + self.last(x[:, -1]))


class MambaTS(nn.Module):
    def __init__(self, n_ch, window, d_model=64, n_layers=2, d_state=16, dropout=0.1, expand=2, **_):
        super().__init__()
        self.embed = nn.Linear(n_ch, d_model)
        self.blocks = nn.ModuleList([ResidualMamba(d_model, d_state, True, dropout, expand=expand) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = HurdleHead(d_model, dropout)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h[:, -1]))


class S4DNet(nn.Module):
    def __init__(self, n_ch, window, d_model=64, n_layers=2, d_state=16, dropout=0.1, **_):
        super().__init__()
        self.embed = nn.Linear(n_ch, d_model)
        self.layers = nn.ModuleList([S4DLayer(d_model, 2 * d_state, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = HurdleHead(d_model, dropout)

    def forward(self, x):
        h = self.embed(x)
        for lyr in self.layers:
            h = lyr(h)
        return self.head(self.norm(h[:, -1]))


class GRUNet(nn.Module):
    def __init__(self, n_ch, window, d_model=64, n_layers=2, dropout=0.1, **_):
        super().__init__()
        self.gru = nn.GRU(n_ch, d_model, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.head = HurdleHead(d_model, dropout)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class TCN(nn.Module):
    def __init__(self, n_ch, window, d_model=64, dropout=0.1, kernel=3, **_):
        super().__init__()
        self.inp = nn.Conv1d(n_ch, d_model, 1)
        self.blocks = nn.ModuleList()
        self.pads = []
        for dil in (1, 2, 4, 8, 16, 32):
            self.pads.append((kernel - 1) * dil)
            self.blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel, dilation=dil), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(d_model, d_model, 1)))
        self.head = HurdleHead(d_model, dropout)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for pad, blk in zip(self.pads, self.blocks):
            h = h + blk(nn.functional.pad(h, (pad, 0)))
        return self.head(h[:, :, -1])


NETS = {"samba": SAMBA, "patchtst": PatchTST, "mamba": MambaTS, "s4d": S4DNet, "gru": GRUNet, "tcn": TCN}


def build_net(name: str, n_ch: int, cfg) -> nn.Module:
    d = cfg.deep
    kw = dict(d_model=d.d_model, n_layers=d.n_layers, dropout=d.dropout)
    if name in ("samba", "samba_nonlinear", "samba_temporal_only"):
        return SAMBA(n_ch, d.window, d_state=d.d_state, patch=d.patch_len, stride=d.patch_stride,
                     nonlinear=(name == "samba_nonlinear"), disentangled=(name != "samba_temporal_only"),
                     expand=int(d.get("expand", 2)), **kw)
    if name == "patchtst":
        return PatchTST(n_ch, d.window, patch=d.patch_len, stride=d.patch_stride, **kw)
    if name in ("mamba", "s4d"):
        return NETS[name](n_ch, d.window, d_state=d.d_state, expand=int(d.get("expand", 2)), **kw)
    return NETS[name](n_ch, d.window, **kw)
