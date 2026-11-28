#!/usr/bin/env python3
"""Pre-training numerical validation of the histogram target/decoding."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src/hdtm'))
from hist_loss import (K, bin_centers, decode_corrected, decode_expectation, decode_median,
                       gaussian_target, hist_loss)

fail = []


def check(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, detail, flush=True)
    if not ok:
        fail.append(name)


def main():
    torch.manual_seed(0)
    ys = torch.linspace(0, 1, 41).view(-1, 1).repeat(1, 3)
    t = gaussian_target(ys)
    check('T1_simplex', bool(torch.allclose(t.sum(-1), torch.ones_like(t.sum(-1)), atol=1e-6))
          and float(t.min()) >= 0)

    scale = 4.0
    c = bin_centers()
    d_raw = (t * c).sum(-1)
    raw_err = (d_raw - ys).abs() * scale
    logits = torch.log(t + 1e-12)
    d_cor = decode_corrected(logits)
    cor_err = (d_cor - ys).abs() * scale
    check('T2_raw_expectation_bias_known', float(raw_err.max()) < 0.2,
          f'raw_max={float(raw_err.max()):.4f}')
    check('T2_corrected_bias', float(cor_err.max()) < 0.01,
          f'corr_max={float(cor_err.max()):.5f} mean={float(cor_err.mean()):.5f}')
    idx = {round(float(x), 2): float(e) for x, e in zip(ys[:, 0], cor_err[:, 0])}
    print('corrected bias@0=%.5f @0.5=%.5f @1=%.5f (1..5 scale)'
          % (idx[0.0], idx[0.5], idx[1.0]), flush=True)

    med = torch.stack([decode_median(torch.log(t[:, dd] + 1e-12)) for dd in range(3)], dim=1)
    merr = (med - ys).abs() * scale
    print('median_bias_max=%.5f mean=%.5f' % (float(merr.max()), float(merr.mean())), flush=True)

    ce_t = float(hist_loss(logits, ys))
    uni = torch.zeros_like(logits)
    ce_u = float(hist_loss(uni, ys))
    shift = torch.full_like(ys, 0.25)
    ce_s = float(hist_loss(logits, shift))
    check('T4_ce_minimised_at_target', ce_t < ce_u and ce_t < ce_s,
          f'ce(t)={ce_t:.4f} ce(uniform)={ce_u:.4f} ce(shifted target)={ce_s:.4f}')

    t0 = gaussian_target(torch.full((1, 3), 0.5))[0, 0]
    check('T5_sigma_concentration', float(t0[K // 2]) > 0.3, f'center_mass={float(t0[K // 2]):.3f}')

    import torch.nn as nn
    head = nn.Linear(8, 3 * K)
    x = torch.randn(5, 8, requires_grad=True)
    out = head(x).view(5, 3, K)
    loss = hist_loss(out, torch.rand(5, 3))
    loss.backward()
    check('T6_grad_flows', x.grad is not None and float(x.grad.abs().sum()) > 0)
    check('T7_decode_shape', tuple(decode_corrected(out).shape) == (5, 3))

    print('RESULT', 'ALL_PASS' if not fail else 'FAILED:' + ','.join(fail), flush=True)
    return 0 if not fail else 1


if __name__ == '__main__':
    raise SystemExit(main())
