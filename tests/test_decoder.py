#!/usr/bin/env python3
"""Phase 2: decoder/boundary verification (bias, monotonicity, range, gradient, robustness)."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src/hdtm'))
from hist_loss import (K, bin_centers, decode_corrected, decode_expectation, gaussian_target,
                       hist_loss)

fail = []


def check(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, detail, flush=True)
    if not ok:
        fail.append(name)


def main():
    torch.manual_seed(0)
    ys = torch.linspace(0, 1, 201).view(-1, 1).repeat(1, 3)
    t = gaussian_target(ys)
    logits = torch.log(t + 1e-12)

    # 1) bias across the whole label range (construction level)
    dec = decode_corrected(logits)
    err = (dec - ys).abs() * 4
    check('P1_bias_all_range', float(err.max()) < 0.005, f'max={float(err.max()):.6f} (1..5)')

    # 2) monotonicity of the corrected decoder along y
    d = dec[:, 0]
    mono = bool((d[1:] >= d[:-1] - 1e-6).all())
    check('P2_monotone', mono, f'min_step={float((d[1:] - d[:-1]).min()):.3e}')

    # 3) output range
    check('P3_range', float(dec.min()) >= 0 and float(dec.max()) <= 1,
          f'[{float(dec.min()):.4f},{float(dec.max()):.4f}]')

    # 4) differentiability w.r.t. logits
    lg = torch.log(t[:20] + 1e-12).clone().requires_grad_(True)
    out = decode_corrected(lg)
    out.sum().backward()
    g = lg.grad
    check('P4_grad_flows', g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0,
          f'|g|1={None if g is None else float(g.abs().sum()):.3e}')
    check('P5_grad_nonzero_everywhere', g is not None and float((g.abs().sum(-1) > 0).float().mean()) > 0.99)

    # 6) behaviour on non-target distributions
    uni = decode_corrected(torch.zeros(1, 3, K))[0]
    check('P6_uniform_mid', bool(torch.allclose(uni, torch.full_like(uni, 0.5), atol=0.02)),
          f'{uni.tolist()}')
    spike = torch.full((1, 3, K), -20.0)
    spike[:, :, 5] = 20.0
    sv = decode_corrected(spike)[0]
    c = bin_centers()
    check('P7_spike_follows_bin', bool(torch.allclose(sv, c[5].repeat(3), atol=1e-3)),
          f'{sv.tolist()} vs {float(c[5]):.4f}')
    two = torch.full((1, 3, K), -20.0)
    two[:, :, 4] = 20.0
    two[:, :, 14] = 20.0
    tv = decode_corrected(two)[0]
    check('P8_two_spikes_between', bool(((tv > c[4]) & (tv < c[14])).all()), f'{tv.tolist()}')

    # 9) extreme logits stability
    ext = torch.full((2, 3, K), -2000.0)
    ext[:, :, 0] = 2000.0
    ev = decode_corrected(ext)
    check('P9_extreme_logits_finite', bool(torch.isfinite(ev).all()) and float(ev.min()) >= 0
          and float(ev.max()) <= 1, f'{ev.flatten().tolist()}')

    # 10) CE vs L1 gradient scale on the same head input (recorded, not tuned)
    head = torch.nn.Linear(8, 3 * K)
    x = torch.randn(64, 8)
    y = torch.rand(64, 3)
    lg2 = head(x).view(64, 3, K)
    g_ce = torch.autograd.grad(hist_loss(lg2, y), head.weight)[0].norm()
    lg3 = head(x).view(64, 3, K)
    g_l1 = torch.autograd.grad(F.l1_loss(decode_corrected(lg3), y), head.weight)[0].norm()
    print(f'grad_scale head: CE={float(g_ce):.4e} L1={float(g_l1):.4e} ratio={float(g_ce / g_l1):.2f}',
          flush=True)

    print('RESULT', 'ALL_PASS' if not fail else 'FAILED:' + ','.join(fail), flush=True)
    return 0 if not fail else 1


if __name__ == '__main__':
    raise SystemExit(main())


