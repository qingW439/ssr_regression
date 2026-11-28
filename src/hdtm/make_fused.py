#!/usr/bin/env python3
"""Fuse the four distribution teachers' unlabeled probabilities for one fold.

Outputs (per fold): fused probability Q = mean_j p_j (renormalised), the decoded
value from Q (used by HDT-M), the mean of per-teacher decoded values (reference),
and the order-difference statistics (decode-then-average vs average-then-decode).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/autodl-tmp/rankup_project')
sys.path.insert(0, str(ROOT / 'm1_hist_gate'))
from hist_loss import decode_corrected  # noqa: E402

P = ROOT
OUT = P / 'm1_hdt_gate/fused'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    for s, f in [(0, 1), (1, 2), (2, 3)]:
        probs, vals = [], []
        for j in range(4):
            z = np.load(P / f'm1_hdt_gate/teachers/B_s{s}_f{f}_j{j}.npz', allow_pickle=True)
            probs.append(torch.tensor(z['u_probs'], dtype=torch.float32))
            vals.append(torch.tensor(z['u_pred'], dtype=torch.float32))
        Q = torch.stack(probs).mean(0)
        Q = Q / Q.sum(-1, keepdim=True).clamp_min(1e-12)
        q_value = decode_corrected(torch.log(Q.clamp_min(1e-12)))     # average prob -> decode
        mean_value = torch.stack(vals).mean(0)                        # decode each -> average
        diff = (q_value - mean_value).abs() * 4
        rep[f's{s}_f{f}'] = {'order_gap_mean': float(diff.mean()), 'order_gap_max': float(diff.max()),
                             'q_value_std': q_value.std(0).tolist(),
                             'mean_value_std': mean_value.std(0).tolist()}
        np.savez(OUT / f'B_s{s}_f{f}.npz', u_ids=np.load(P / 'm1_v2_assets/u_ids.npy', allow_pickle=True),
                 u_probs=Q.numpy().astype(np.float32), u_value=q_value.numpy().astype(np.float32),
                 u_value_mean=mean_value.numpy().astype(np.float32))
        print(f's{s}_f{f} order_gap_mean={rep[f"s{s}_f{f}"]["order_gap_mean"]:.5f} '
              f'std(Q)={[round(x, 4) for x in rep[f"s{s}_f{f}"]["q_value_std"]]}', flush=True)
    (P / 'm1_report/FUSED_TARGETS.json').write_text(json.dumps(rep, indent=2) + '\n')


if __name__ == '__main__':
    main()
