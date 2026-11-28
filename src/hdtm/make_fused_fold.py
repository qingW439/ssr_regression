"""Fuse the four distribution teachers for one fold (formal pipeline)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
ROOT = ASSETS
sys.path.insert(0, str(REPO / 'src/hdtm'))
from hist_loss import decode_corrected  # noqa: E402

P = ROOT
OUT = P / 'm1_hdt_gate/fused'


def main():
    seed, fold = int(sys.argv[1]), int(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    probs, vals = [], []
    for j in range(4):
        z = np.load(P / f'm1_hdt_gate/teachers/B_s{seed}_f{fold}_j{j}.npz', allow_pickle=True)
        probs.append(torch.tensor(z['u_probs'], dtype=torch.float32))
        vals.append(torch.tensor(z['u_pred'], dtype=torch.float32))
    Q = torch.stack(probs).mean(0)
    Q = Q / Q.sum(-1, keepdim=True).clamp_min(1e-12)
    q_value = decode_corrected(torch.log(Q.clamp_min(1e-12)))
    mean_value = torch.stack(vals).mean(0)
    gap = float(((q_value - mean_value).abs() * 4).mean())
    np.savez(OUT / f'B_s{seed}_f{fold}.npz',
             u_ids=np.load(P / 'm1_v2_assets/u_ids.npy', allow_pickle=True),
             u_probs=Q.numpy().astype(np.float32), u_value=q_value.numpy().astype(np.float32),
             u_value_mean=mean_value.numpy().astype(np.float32))
    print(json.dumps({'seed': seed, 'fold': fold, 'order_gap': gap,
                      'std': [round(float(x), 4) for x in q_value.std(0)]}), flush=True)


if __name__ == '__main__':
    main()
