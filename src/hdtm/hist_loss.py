#!/usr/bin/env python3
"""Histogram (distributional) regression utilities with construction-derived decoding.

Target: K=20 bins over [0,1]; masses from a Gaussian of width sigma = 1/K,
with boundary reflection and renormalisation.

Decoding (pre-registered): expectation over bin centres, followed by a
deterministic inverse-LUT that undoes the *construction's own* truncation bias
(computed from the target construction only, never from labels or OOF).
Median decoding is kept as a diagnostic and is not used for reported results.
"""
import torch
import torch.nn.functional as F
from torch import nn

K = 20
SIGMA_BINS = 1.0
_LUT = {}


def bin_edges(k=K, device=None, dtype=torch.float32):
    return torch.linspace(0, 1, k + 1, device=device, dtype=dtype)


def bin_centers(k=K, device=None, dtype=torch.float32):
    return torch.linspace(0, 1, k + 1, device=device, dtype=dtype)[:-1] + 0.5 / k


def gaussian_target(y, k=K, sigma_bins=SIGMA_BINS):
    dev, dt = y.device, y.dtype
    edges = bin_edges(k, dev, dt)
    sigma = sigma_bins / k
    yb = y.unsqueeze(-1)
    lo = edges[:-1].view(1, 1, k)
    hi = edges[1:].view(1, 1, k)

    def mass(center):
        return torch.special.ndtr((hi - center) / sigma) - torch.special.ndtr((lo - center) / sigma)

    p = mass(yb) + mass(-yb) + mass(2 - yb)
    p = p.clamp_min(1e-12)
    return p / p.sum(-1, keepdim=True)


def _lut(k, device, dtype, n=401):
    key = (k, n)
    if key in _LUT:
        return _LUT[key]
    g = torch.linspace(0, 1, n, device=device, dtype=dtype)
    t = gaussian_target(g.view(-1, 1).repeat(1, 3), k)
    c = bin_centers(k, device, dtype)
    d = (t * c).sum(-1)[:, 0]
    _LUT[key] = (d, g)
    return _LUT[key]


def decode_expectation(logits, k=K):
    p = F.softmax(logits, dim=-1)
    c = bin_centers(k, logits.device, logits.dtype).expand_as(p)
    return (p * c).sum(-1)


def decode_corrected(logits, k=K):
    """Expectation + inverse LUT of the construction bias (pre-registered)."""
    d = decode_expectation(logits, k)
    dg, yg = _lut(k, d.device, d.dtype)
    d = d.clamp(dg.min(), dg.max())
    idx = torch.searchsorted(dg, d.contiguous())
    idx = idx.clamp(1, len(dg) - 1)
    d0, d1 = dg[idx - 1], dg[idx]
    y0, y1 = yg[idx - 1], yg[idx]
    w = ((d - d0) / (d1 - d0).clamp_min(1e-12)).clamp(0, 1)
    return (y0 + w * (y1 - y0)).clamp(0, 1)


def decode_median(logits, k=K):
    p = F.softmax(logits, dim=-1)
    c = bin_centers(k, logits.device, logits.dtype).expand_as(p)
    cdf = p.cumsum(-1)
    idx = (cdf < 0.5).sum(-1).clamp_max(k - 1)
    return c.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def hist_loss(logits, y, k=K, sigma_bins=SIGMA_BINS):
    t = gaussian_target(y, k, sigma_bins)
    return -(t * F.log_softmax(logits, dim=-1)).sum(-1).mean()


class HistHead(nn.Module):
    def __init__(self, hidden, k=K):
        super().__init__()
        self.k = k
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 3 * k)

    def forward(self, x):
        z = self.fc2(self.act(self.fc1(x)))
        return z.view(*z.shape[:-1], 3, self.k)
