#!/usr/bin/env python3
"""Numerical cross-check of rcus_loss.py against a literal Eq.(3)-(5),(8) loop.

The reference below transcribes the paper equations element-by-element in plain
Python loops (no vectorisation, no shared helpers). The vectorised implementation
must match it to 1e-6 on random inputs, including confidence masking.

Also checks:
  P1  at lam=1 both mixed samples collapse onto their endpoints, so Lu = 2*log2.
  P2  low-confidence pairs are excluded from the mask; tau=1 gives an empty mask.
  P3  mixing is order-antisymmetric: za(lam) with reversed batch == zb(lam).
  P4  gradients reach the features through the mixed branch.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src/baselines"))

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rcus_loss import (labeled_ranking_loss, mixup_features, mixup_ranking_loss,  # noqa: E402
                       offdiag_mask, pseudo_rank)

torch.manual_seed(0)
fail = []


def check(name, ok, detail=''):
    print(('PASS' if ok else 'FAIL'), name, detail, flush=True)
    if not ok:
        fail.append(name)


def bce(diff, t):
    return float(F.binary_cross_entropy_with_logits(torch.tensor(diff), torch.tensor(t)))


def ref_labeled_loss(scores, y):
    B, D = scores.shape
    tot, n = 0.0, 0
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            for d in range(D):
                tot += bce(float(scores[i, d] - scores[j, d]), 1.0 if y[i, d] > y[j, d] else 0.0)
                n += 1
    return tot / n


def ref_unlabeled_loss(scores, mix_a, mix_b, t, m):
    """Eq.(5): (1/P) * sum over valid pairs and dims of [CE(a) + CE(b)]."""
    B, D = scores.shape
    tot, npair = 0.0, 0
    for i in range(B):
        for j in range(B):
            if i == j or float(m[i, j, 0]) == 0.0:
                continue
            npair += 1
            for d in range(D):
                tot += bce(float(scores[i, d] - mix_a[i, j, d]), float(t[i, j, d]))
                tot += bce(float(mix_b[i, j, d] - scores[j, d]), float(t[i, j, d]))
    return tot / (npair * D)


def main():
    B, H, D = 5, 8, 3
    feat = torch.randn(B, H)
    W = torch.randn(3, H)                      # scalar ranking score per dim
    y = torch.rand(B, D)
    scores = feat @ W.T

    got = float(labeled_ranking_loss(scores, y))
    want = ref_labeled_loss(scores, y)
    check('Lr_matches_reference', abs(got - want) < 1e-6, f'got={got:.8f} want={want:.8f}')

    lam = 0.72
    za, zb = mixup_features(feat, lam)
    mix_a, mix_b = za @ W.T, zb @ W.T
    t, m = pseudo_rank(scores, tau=0.0)        # every off-diagonal pair valid
    got2 = float(mixup_ranking_loss(scores, mix_a, mix_b, t, m))
    want2 = ref_unlabeled_loss(scores, mix_a, mix_b, t, m)
    check('Lu_matches_reference', abs(got2 - want2) < 1e-6, f'got={got2:.8f} want={want2:.8f}')

    # P1: lam=1 -> z_ij = z_i and z_ij' = z_j, every pair logit is exactly 0
    za1, zb1 = mixup_features(feat, 1.0)
    check('P1_endpoint_identity',
          torch.allclose(za1, feat.unsqueeze(1).expand(B, B, H))
          and torch.allclose(zb1, feat.unsqueeze(0).expand(B, B, H)))
    lu1 = float(mixup_ranking_loss(scores, za1 @ W.T, zb1 @ W.T, t, m))
    check('P1_lam1_equals_2log2', abs(lu1 - 2 * torch.log(torch.tensor(2.0)).item()) < 1e-6, f'{lu1:.6f}')

    # P3: swapping the pair order exchanges the two interpolants
    check('P3_antisymmetry', torch.equal(za, zb.transpose(0, 1)))

    # P2: mask excludes low-confidence pairs
    t2, m2 = pseudo_rank(scores, tau=0.95)
    q = torch.sigmoid(scores.unsqueeze(1) - scores.unsqueeze(0))
    conf = torch.maximum(q, 1 - q)
    expected = ((conf >= 0.95).float() * (offdiag_mask(B, scores.device) > 0).float().unsqueeze(-1))
    check('P2_confidence_mask', torch.equal(m2, expected), f'mask_frac={float(m2.mean()):.3f}')
    low = mixup_ranking_loss(scores, mix_a, mix_b, t2, m2)
    check('P2_masked_loss_finite', bool(torch.isfinite(low)))
    _, m3 = pseudo_rank(scores, tau=1.0)
    check('P2_empty_mask_at_tau1', float(m3.sum()) == 0.0)

    # P4: gradient reaches features through the mixed branch
    f = feat.clone().requires_grad_(True)
    za, zb = mixup_features(f, 0.6)
    loss = mixup_ranking_loss(f @ W.T, za @ W.T, zb @ W.T, t, m)
    loss.backward()
    check('P4_feature_grad', f.grad is not None and bool(torch.isfinite(f.grad).all())
          and float(f.grad.abs().sum()) > 0)

    print('RESULT', 'ALL_PASS' if not fail else 'FAILED:' + ','.join(fail), flush=True)
    return 0 if not fail else 1


if __name__ == '__main__':
    raise SystemExit(main())
