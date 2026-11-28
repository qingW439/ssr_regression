#!/usr/bin/env python3
"""Phase 2/4 aggregation: formal 15-fold table incl. T-HS and A-CL, plus the 2x2 teacher x student cross."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
DIMS = ('V', 'A', 'D')
PATHS = {'U0': 'm1_uul_runs/U0_s{s}_f{f}', 'U1': 'm1_uul_runs/U1_s{s}_f{f}',
         'HPL': 'm1_hpl_runs/HPL_s{s}_f{f}', 'RCUS': 'm1_rcus_runs/RCUS2_s{s}_f{f}',
         'HS': 'm1_hist_runs/hs_s{s}_f{f}', 'HU': 'm1_hist_runs/hu_s{s}_f{f}',
         'THS': 'm1_hdt_runs/ths_s{s}_f{f}', 'HDTM': 'm1_hdt_runs/hdtm_s{s}_f{f}',
         'ACL': 'm1_hdt_runs/acl_s{s}_f{f}'}


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def met(d):
    per = {}
    for x in DIMS:
        y, p = d[f'y_{x}'].to_numpy(), d[f'raw_{x}'].to_numpy()
        q = np.quantile(y, [0.2, 0.8])
        lo, hi = y < q[0], y > q[1]
        per[x] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p),
                  'pearson': float(np.corrcoef(y, p)[0, 1]), 'srcc': float(spearmanr(y, p).statistic),
                  'r2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                  'tail_mae': float(np.abs(y[lo | hi] - p[lo | hi]).mean() * 4),
                  'low_mae': float(np.abs(y[lo] - p[lo]).mean() * 4),
                  'high_mae': float(np.abs(y[hi] - p[hi]).mean() * 4),
                  'mid_mae': float(np.abs(y[~lo & ~hi] - p[~lo & ~hi]).mean() * 4),
                  'sd_ratio': float(p.std() / y.std()), 'bias': float((p - y).mean() * 4)}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'pearson': float(np.mean([v['pearson'] for v in per.values()])),
            'srcc': float(np.mean([v['srcc'] for v in per.values()])),
            'r2': float(np.mean([v['r2'] for v in per.values()])),
            'tail_mae': float(np.mean([v['tail_mae'] for v in per.values()])),
            'low_mae': float(np.mean([v['low_mae'] for v in per.values()])),
            'high_mae': float(np.mean([v['high_mae'] for v in per.values()])),
            'mid_mae': float(np.mean([v['mid_mae'] for v in per.values()])),
            'sd_ratio': float(np.mean([v['sd_ratio'] for v in per.values()])),
            'bias': float(np.mean([v['bias'] for v in per.values()])),
            'per_dim': per}


rows = []
for s in range(3):
    for f in range(5):
        for arm, pat in PATHS.items():
            oof = A / pat.format(s=s, f=f) / 'oof_predictions.parquet'
            if not oof.is_file():
                continue
            m = met(pd.read_parquet(oof))
            rows.append({'arm': arm, 'seed': s, 'fold': f,
                         **{k: m[k] for k in ('mae', 'ccc', 'pearson', 'srcc', 'r2', 'tail_mae', 'low_mae',
                                              'high_mae', 'mid_mae', 'sd_ratio', 'bias')},
                         **{f'{x}_mae': m['per_dim'][x]['mae'] for x in DIMS},
                         **{f'{x}_ccc': m['per_dim'][x]['ccc'] for x in DIMS}})
df = pd.DataFrame(rows)
df.to_csv(A / 'm1_report/PHASE24_TABLE.csv', index=False)
cols = ['mae', 'ccc', 'pearson', 'srcc', 'r2', 'tail_mae', 'low_mae', 'high_mae', 'mid_mae', 'sd_ratio', 'bias']
print(df.groupby('arm')[cols].mean().round(6).to_string())
print('\nn:', df.groupby('arm').size().to_dict())

summ = {'means': df.groupby('arm')[cols].mean().round(8).to_dict(), 'n': df.groupby('arm').size().to_dict()}


def paired(a, b):
    if a not in set(df.arm) or b not in set(df.arm):
        return None
    piv = df.pivot(index=['seed', 'fold'], columns='arm', values='mae')
    pc = df.pivot(index=['seed', 'fold'], columns='arm', values='ccc')
    keep = piv[[a, b]].dropna().index
    dm = piv.loc[keep, a] - piv.loc[keep, b]
    dc = pc.loc[keep, a] - pc.loc[keep, b]
    return {'n': int(len(dm)), 'dmae': float(dm.mean()),
            'dmae_se': float(dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 1 else None,
            'dccc': float(dc.mean()), 'wins': int((dm < 0).sum()), 'losses': int((dm > 0).sum())}


for a, b in (('THS', 'U1'), ('HDTM', 'HU'), ('HDTM', 'THS'), ('THS', 'HU'), ('ACL', 'HDTM'), ('ACL', 'HS')):
    r = paired(a, b)
    if r:
        summ[f'{a}_minus_{b}'] = r

# 2x2 on the 3 development units (the only units all four share)
dev = [(0, 1), (1, 2), (2, 3)]
piv = df.set_index(['arm', 'seed', 'fold']).mae
try:
    g = {}
    for arm in ('U1', 'THS', 'HU', 'HDTM'):
        g[arm] = float(np.mean([piv.loc[(arm, s, f)] for s, f in dev]))
    summ['dev3_2x2'] = {'U1': g['U1'], 'THS': g['THS'], 'HU': g['HU'], 'HDTM': g['HDTM'],
                        'teacher_gain_scalar': g['U1'] - g['THS'],
                        'teacher_gain_dist': g['HU'] - g['HDTM'],
                        'interaction': (g['HU'] - g['HDTM']) - (g['U1'] - g['THS'])}
except KeyError as e:
    summ['dev3_2x2'] = f'missing {e}'

(A / 'm1_report/PHASE24_SUMMARY.json').write_text(json.dumps(summ, indent=2) + '\n')
print(json.dumps({k: v for k, v in summ.items() if k != 'means'}, indent=2)[:2600])
