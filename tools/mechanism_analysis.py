#!/usr/bin/env python3
"""Phase 1 mechanism analysis: fusion order, decoder comparison, residual error profile.

Reads only existing weights/caches; trains nothing.  Scores are reported on the original
1..5 scale (normalised values * 4) and kept alongside the normalised statistics.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

A = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src/hdtm'))
sys.path.insert(0, str(REPO))
from hist_loss import decode_corrected, decode_expectation  # noqa: E402

DIMS = ('V', 'A', 'D')
OUT = A / 'm1_report'
OUT.mkdir(exist_ok=True)


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def macro(y, p):
    per = {}
    for i, d in enumerate(DIMS):
        yi, pi = y[:, i], p[:, i]
        per[d] = {'mae': float(np.abs(pi - yi).mean() * 4), 'ccc': ccc(yi, pi),
                  'bias': float((pi - yi).mean() * 4), 'sd_ratio': float(pi.std() / yi.std())}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])), 'per_dim': per}


# ---------------------------------------------------------------- A. fusion order
def fusion_order():
    rows = []
    for p in sorted((A / 'm1_hdt_gate/fused').glob('B_s*_f*.npz')):
        z = np.load(p, allow_pickle=True)
        v, vm = z['u_value'], z['u_value_mean']
        diff = (v - vm) * 4.0
        absd = np.abs(diff)
        rec = {'file': p.name, 'n': int(v.shape[0]), 'finite': bool(np.isfinite(v).all() and np.isfinite(vm).all()),
               'prob_sums_ok': bool(np.allclose(z['u_probs'].sum(-1), 1.0, atol=1e-4))}
        for i, d in enumerate(DIMS):
            a = absd[:, i]
            rec[f'{d}_mean_signed'] = float(diff[:, i].mean())
            rec[f'{d}_mean_abs'] = float(a.mean())
            rec[f'{d}_p95'] = float(np.quantile(a, 0.95))
            rec[f'{d}_p99'] = float(np.quantile(a, 0.99))
            rec[f'{d}_max'] = float(a.max())
            rec[f'{d}_frac_gt1e-4'] = float((a > 1e-4).mean())
            rec[f'{d}_frac_gt1e-3'] = float((a > 1e-3).mean())
            rec[f'{d}_frac_gt1e-2'] = float((a > 1e-2).mean())
            near = (v[:, i] < 0.05) | (v[:, i] > 0.95)
            rec[f'{d}_near_edge_frac'] = float(near.mean())
            rec[f'{d}_near_edge_absdiff'] = float(a[near].mean()) if near.any() else 0.0
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'FUSION_ORDER.csv', index=False)
    cols = [c for c in df.columns if c.endswith(('mean_abs', 'p99', 'max', 'frac_gt1e-3', 'frac_gt1e-2'))]
    print('== fusion order (1..5 scale), 15 folds')
    print(df[cols].mean().round(6).to_string())
    print('all finite:', bool(df['finite'].all()), 'prob normalised:', bool(df['prob_sums_ok'].all()))
    return df


# ---------------------------------------------------------------- B. decoder comparison
def decoder_comparison():
    import torch
    from semilearn.datasets.nlp_datasets.emobank import read_gold
    from transformers import AutoTokenizer
    from train_hdt import HistVAD

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(str(A / 'pretrained/roberta-base-simcse'), local_files_only=True, use_fast=True)
    rows = []
    for s in range(3):
        for f in range(5):
            run = A / f'm1_hdt_runs/hdtm_s{s}_f{f}'
            if not (run / 'readout.pth').is_file():
                print('MISSING', run)
                continue
            man = json.loads((A / ('Semi-Supervised-Regression-avrrc/outputs/p0' if s == 0
                                   else f'Semi-Supervised-Regression-avrrc/outputs/p0_s{s}') / 'fold_manifest.json').read_text())
            lb = read_gold(A / f'data/emobank_processed/lb500_s{s}/train_lb.parquet', expected_n=500)
            ho = lb[lb['id'].isin(set(man['fold_ids'][str(f)]))].reset_index(drop=True)
            m = HistVAD().to(dev)
            m.load_state_dict(torch.load(run / 'readout.pth', map_location='cpu', weights_only=False)['model'])
            m.eval()
            outs = []
            with torch.inference_mode():
                for i in range(0, len(ho), 64):
                    b = tok(ho['text'].tolist()[i:i + 64], padding=True, truncation=True, max_length=128,
                            return_tensors='pt')
                    b = {k: v.to(dev) for k, v in b.items()}
                    outs.append(m(**b).float().cpu())
            logits = torch.cat(outs)
            y = ho[['V_norm', 'A_norm', 'D_norm']].to_numpy(np.float32)
            for name, fn in (('expectation', decode_expectation), ('corrected', decode_corrected)):
                p = fn(logits).numpy()
                met = macro(y, p)
                rows.append({'seed': s, 'fold': f, 'decoder': name, 'mae': met['mae'], 'ccc': met['ccc'],
                             'v_mae': met['per_dim']['V']['mae'], 'a_mae': met['per_dim']['A']['mae'],
                             'd_mae': met['per_dim']['D']['mae'],
                             'v_ccc': met['per_dim']['V']['ccc'], 'a_ccc': met['per_dim']['A']['ccc'],
                             'd_ccc': met['per_dim']['D']['ccc']})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'DECODER_COMPARISON.csv', index=False)
    g = df.groupby('decoder')[['mae', 'ccc', 'v_mae', 'a_mae', 'd_mae', 'v_ccc', 'a_ccc', 'd_ccc']].mean().round(6)
    print('\n== decoder comparison (final weights, 15 folds)')
    print(g.to_string())
    piv = df.pivot(index=['seed', 'fold'], columns='decoder', values='mae')
    print('corrected - expectation: dMAE =', round(float((piv['corrected'] - piv['expectation']).mean()), 6),
          'wins =', int((piv['corrected'] < piv['expectation']).sum()), '/', len(piv))
    return df


# ---------------------------------------------------------------- C. residual profile
def residual_profile():
    rows = []
    for arm, pat in (('HDTM', 'm1_hdt_runs/hdtm_s{s}_f{f}'), ('HS', 'm1_hist_runs/hs_s{s}_f{f}'),
                     ('HU', 'm1_hist_runs/hu_s{s}_f{f}')):
        for s in range(3):
            for f in range(5):
                d = A / pat.format(s=s, f=f)
                if not (d / 'oof_predictions.parquet').is_file():
                    continue
                df = pd.read_parquet(d / 'oof_predictions.parquet')
                for i, dim in enumerate(DIMS):
                    y, p = df[f'y_{dim}'].to_numpy(), df[f'raw_{dim}'].to_numpy()
                    lo = y < 0.2
                    hi = y > 0.8
                    mid = ~(lo | hi)
                    rows.append({'arm': arm, 'seed': s, 'fold': f, 'dim': dim,
                                 'mae': float(np.abs(p - y).mean() * 4), 'bias': float((p - y).mean() * 4),
                                 'ccc': ccc(y, p), 'sd_ratio': float(p.std() / y.std()),
                                 'low_mae': float(np.abs(p[lo] - y[lo]).mean() * 4) if lo.any() else np.nan,
                                 'high_mae': float(np.abs(p[hi] - y[hi]).mean() * 4) if hi.any() else np.nan,
                                 'mid_mae': float(np.abs(p[mid] - y[mid]).mean() * 4),
                                 'low_n': int(lo.sum()), 'high_n': int(hi.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'RESIDUAL_PROFILE.csv', index=False)
    print('\n== residual profile by arm/dim')
    print(df.groupby(['arm', 'dim'])[['mae', 'bias', 'ccc', 'sd_ratio', 'low_mae', 'high_mae', 'mid_mae']]
          .mean().round(6).to_string())
    print('\nper seed macro MAE')
    print(df.groupby(['arm', 'seed'])['mae'].mean().unstack(0).round(6).to_string())
    return df


if __name__ == '__main__':
    fusion_order()
    decoder_comparison()
    residual_profile()
