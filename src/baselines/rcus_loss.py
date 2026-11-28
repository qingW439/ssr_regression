#!/usr/bin/env python3
"""Official SSR-RCUS ranking losses (Eq. 3-5, 8) adapted to 3D VAD.

Ranking predictor: q_ij = sigmoid(H'z_i - H'z_j), implemented as one scalar
score per dimension (H_d) so that the pair logit is s_i - s_j.

  Lr  (Eq. 8): CE/BCE over labeled pairs, target t_ij = I(y_i > y_j)
  Lu  (Eq. 5): manifold-mixup ranking consistency over unlabeled pairs
               z_ij  = lam*z_i + (1-lam)*z_j        (closer to i)
               z_ij' = (1-lam)*z_i + lam*z_j        (closer to j)
               q_ij  = h(z_i, z_j) from clean features (pseudo rank)
               tau_ij= max(q_ij, 1-q_ij) >= 0.95
               t     = I(q_ij > 0.5)
               loss  = CE(h(z_i,z_ij), t) + CE(h(z_ij',z_j), t)

Diagonal pairs (i == j) are excluded: their pair logit is identically zero and
their target is undefined, so including them only injects a constant log(2)
with a spurious gradient. Everything else follows the official equations.
"""
import torch
import torch.nn.functional as F

DIMS = 3


def pair_diff(scores):
    """scores: (..., B, D) -> (B, B, D) with out[i,j] = scores[i] - scores[j]."""
    return scores.unsqueeze(1) - scores.unsqueeze(0)


def offdiag_mask(n, device, dtype=torch.float32):
    m = torch.ones(n, n, device=device, dtype=dtype)
    m.fill_diagonal_(0.0)
    return m


def labeled_ranking_loss(scores, y):
    """Eq. (8). scores: (B,D) ranking scores; y: (B,D) normalized targets."""
    diff = pair_diff(scores)                     # (B,B,D) = s_i - s_j
    t = (y.unsqueeze(1) > y.unsqueeze(0)).to(diff.dtype)   # t_ij = I(y_i > y_j)
    m = offdiag_mask(y.shape[0], y.device, diff.dtype)
    loss = F.binary_cross_entropy_with_logits(diff, t, reduction='none')
    return (loss * m.unsqueeze(-1)).sum() / (m.sum() * y.shape[1])


def pseudo_rank(scores, tau=0.95):
    """Confident pseudo ranks from clean features (no grad)."""
    with torch.no_grad():
        q = torch.sigmoid(pair_diff(scores.detach()))        # (B,B,D)
        conf = torch.maximum(q, 1.0 - q)
        m = offdiag_mask(scores.shape[0], scores.device, q.dtype).unsqueeze(-1) \
            * (conf >= tau).to(q.dtype)                      # (B,B,D)
        t = (q > 0.5).to(q.dtype)
    return t, m


def mixup_ranking_loss(scores, scores_mix_a, scores_mix_b, t, m):
    """Eq. (5) given precomputed mixed-feature scores.

    scores_mix_a[i,j] = h(z_ij)  (z_ij close to z_i)
    scores_mix_b[i,j] = h(z_ij') (z_ij' close to z_j)
    t, m: (B,B,D) targets / mask from pseudo_rank() or labeled targets.
    """
    logit_a = scores.unsqueeze(1) - scores_mix_a      # h(z_i, z_ij)
    logit_b = scores_mix_b - scores.unsqueeze(0)      # h(z_ij', z_j)
    wa = F.binary_cross_entropy_with_logits(logit_a, t, reduction='none')
    wb = F.binary_cross_entropy_with_logits(logit_b, t, reduction='none')
    return ((wa + wb) * m).sum() / m.sum().clamp_min(1.0)


def mixup_features(feat, lam):
    """Manifold mixup of latent features, Eq. (6).

    feat: (B,H) -> za[i,j] = lam*f_i + (1-lam)*f_j ; zb[i,j] = (1-lam)*f_i + lam*f_j
    """
    za = lam * feat.unsqueeze(1) + (1.0 - lam) * feat.unsqueeze(0)
    zb = (1.0 - lam) * feat.unsqueeze(1) + lam * feat.unsqueeze(0)
    return za, zb
