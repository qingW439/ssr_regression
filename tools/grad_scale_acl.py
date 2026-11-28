#!/usr/bin/env python3
"""Gradient-scale calibration for A-CL (labelled CE vs unlabeled CE vs unlabeled L1).

Measured on legal training batches for the three development units, at the HDT-M trained
state (and at init as a sanity check).  No outer-holdout data is used.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

A = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src/hdtm'))
sys.path.insert(0, str(REPO))
from hist_loss import decode_corrected, gaussian_target, hist_loss  # noqa: E402
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from train_hdt import HistVAD  # noqa: E402

DEV = (0, 1), (1, 2), (2, 3)
DIMS = ('V', 'A', 'D')


def norms(model):
    enc = [p.grad.detach().flatten() for p in model.encoder.parameters() if p.grad is not None]
    head = [p.grad.detach().flatten() for p in model.head.parameters() if p.grad is not None]
    return (float(torch.cat(enc).norm()) if enc else 0.0, float(torch.cat(head).norm()) if head else 0.0)


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(str(A / 'pretrained/roberta-base-simcse'), local_files_only=True, use_fast=True)
    out = {'device': str(dev), 'units': {}, 'batches_per_unit': 4}
    ratios = []
    for s, f in DEV:
        z = np.load(A / f'm1_hdt_gate/fused/B_s{s}_f{f}.npz', allow_pickle=True)
        q_val = torch.tensor(z['u_value'], device=dev)
        u_ids = np.load(A / 'm1_v2_assets/u_ids.npy', allow_pickle=True)
        u_all = __import__('pandas').read_parquet(A / 'data/emobank_processed/lb500_s0/train_ulb.parquet').set_index('id')
        utext = [u_all.at[i, 'text'] for i in u_ids]
        lb = read_gold(A / f'data/emobank_processed/lb500_s{s}/train_lb.parquet', expected_n=500)
        man = json.loads((A / ('Semi-Supervised-Regression-avrrc/outputs/p0' if s == 0
                               else f'Semi-Supervised-Regression-avrrc/outputs/p0_s{s}') / 'fold_manifest.json').read_text())
        tr = lb[~lb['id'].isin(set(man['fold_ids'][str(f)]))].reset_index(drop=True)
        lb_dl = DataLoader(GoldData(tr), batch_size=16, shuffle=True, collate_fn=TextCollator(tok, 128),
                           generator=torch.Generator().manual_seed(31000 + 100 * s + f))
        u_dl = DataLoader(list(range(len(utext))), batch_size=16, shuffle=True,
                          generator=torch.Generator().manual_seed(19000 + 100 * s + f))

        def measure(model, n=4):
            lb_it, u_it = iter(lb_dl), iter(u_dl)
            g = {'sup': [], 'ce_u': [], 'l1_u': []}
            for _ in range(n):
                b, y, _ = next(lb_it)
                idx = next(u_it)
                b = {k: v.to(dev) for k, v in b.items()}
                y = y.to(dev)
                ub = tok([utext[i] for i in idx], padding=True, truncation=True, max_length=128, return_tensors='pt')
                ub = {k: v.to(dev) for k, v in ub.items()}
                model.zero_grad(set_to_none=True)
                hist_loss(model(**b), y).backward()
                g['sup'].append(norms(model))
                model.zero_grad(set_to_none=True)
                t_u = gaussian_target(q_val[idx.to(dev)])
                (-(t_u * F.log_softmax(model(**ub), dim=-1)).sum(-1)).mean().backward()
                g['ce_u'].append(norms(model))
                model.zero_grad(set_to_none=True)
                F.l1_loss(decode_corrected(model(**ub)), q_val[idx.to(dev)]).backward()
                g['l1_u'].append(norms(model))
            return {k: {'enc': float(np.median([x[0] for x in v])), 'head': float(np.median([x[1] for x in v]))}
                    for k, v in g.items()}

        # trained state of the matching HDT-M unit
        ck = torch.load(A / f'm1_hdt_runs/hdtm_s{s}_f{f}/readout.pth', map_location='cpu', weights_only=False)
        m = HistVAD().to(dev)
        m.load_state_dict(ck['model'])
        m.train()
        trained = measure(m)
        torch.manual_seed(1000 * s + f)
        m0 = HistVAD().to(dev).train()
        init = measure(m0)
        r = trained['ce_u']['enc'] / max(trained['l1_u']['enc'], 1e-12)
        ratios.append(r)
        out['units'][f's{s}_f{f}'] = {'trained': trained, 'init': init, 'ratio_ce_over_l1_enc': r}
        print(f's{s}f{f} trained enc sup={trained["sup"]["enc"]:.2f} ce_u={trained["ce_u"]["enc"]:.2f} '
              f'l1_u={trained["l1_u"]["enc"]:.2f} ratio={r:.2f} | head sup={trained["sup"]["head"]:.2f} '
              f'ce_u={trained["ce_u"]["head"]:.2f} l1_u={trained["l1_u"]["head"]:.2f}', flush=True)

    med = float(np.median(ratios))
    out['median_ratio_ce_over_l1_enc'] = med
    out['lambda_acl'] = 0.25 * med
    out['original_ce_contribution'] = {'note': 'HDT-M uses 0.25 * unlabeled CE; the matched A-CL weight keeps the '
                                              'same unlabeled encoder-gradient contribution at the measured state',
                                       'lambda': 0.25}
    (A / 'm1_report/ACL_GRAD.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'median_ratio': med, 'lambda_acl': out['lambda_acl']}, indent=2))


if __name__ == '__main__':
    main()
