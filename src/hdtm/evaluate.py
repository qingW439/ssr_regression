"""Formal 15-fold table for HDT-M and all compared models (unified metric code)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path('/root/autodl-tmp/rankup_project')
DIMS = ('V', 'A', 'D')
PATHS = {'U0': 'm1_uul_runs/U0_s{s}_f{f}', 'U1': 'm1_uul_runs/U1_s{s}_f{f}',
         'HPL': 'm1_hpl_runs/HPL_s{s}_f{f}', 'RCUS': 'm1_rcus_runs/RCUS2_s{s}_f{f}',
         'HS': 'm1_hist_runs/hs_s{s}_f{f}', 'HU': 'm1_hist_runs/hu_s{s}_f{f}',
         'HDTM': 'm1_hdt_runs/hdtm_s{s}_f{f}'}


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def met(d):
    per = {}
    for x in DIMS:
        y, p = d[f'y_{x}'].to_numpy(), d[f'raw_{x}'].to_numpy()
        q = np.quantile(y, [0.2, 0.8])
        tail = (y < q[0]) | (y > q[1])
        per[x] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p),
                  'pearson': float(np.corrcoef(y, p)[0, 1]),
                  'srcc': float(spearmanr(y, p).statistic),
                  'r2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                  'tail_mae': float(np.abs(y[tail] - p[tail]).mean() * 4),
                  'sd_ratio': float(p.std() / y.std()),
                  'mid_mae': float(np.abs(y[~tail] - p[~tail]).mean() * 4)}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'pearson': float(np.mean([v['pearson'] for v in per.values()])),
            'srcc': float(np.mean([v['srcc'] for v in per.values()])),
            'r2': float(np.mean([v['r2'] for v in per.values()])),
            'tail_mae': float(np.mean([v['tail_mae'] for v in per.values()])),
            'mid_mae': float(np.mean([v['mid_mae'] for v in per.values()])),
            'sd_ratio': float(np.mean([v['sd_ratio'] for v in per.values()])),
            'per_dim': per}


rows = []
for s in range(3):
    for f in range(5):
        for arm, pat in PATHS.items():
            oof = P / pat.format(s=s, f=f) / 'oof_predictions.parquet'
            if not oof.is_file():
                print('MISSING', arm, s, f)
                continue
            m = met(pd.read_parquet(oof))
            rows.append({'arm': arm, 'seed': s, 'fold': f,
                         **{k: m[k] for k in ('mae', 'ccc', 'pearson', 'srcc', 'r2', 'tail_mae', 'mid_mae', 'sd_ratio')},
                         **{f'{x}_mae': m['per_dim'][x]['mae'] for x in DIMS},
                         **{f'{x}_ccc': m['per_dim'][x]['ccc'] for x in DIMS}})
df = pd.DataFrame(rows)
df.to_csv(P / 'm1_hdt_gate/formal15.csv', index=False)
cols = ['mae', 'ccc', 'pearson', 'srcc', 'r2', 'tail_mae', 'mid_mae', 'sd_ratio']
g = df.groupby('arm')[cols].mean().round(6)
print(g.to_string())
print('\nn per arm:', df.groupby('arm').size().to_dict())
print('\nper seed MAE\n', df.pivot_table(index='arm', columns='seed', values='mae').round(6).to_string())

piv = {k: df.pivot(index=['seed', 'fold'], columns='arm', values=k) for k in ('mae', 'ccc', 'tail_mae')}
summary = {'means': g.to_dict(), 'n': df.groupby('arm').size().to_dict()}
for b in ('U0', 'U1', 'HPL', 'RCUS', 'HS', 'HU'):
    if 'HDTM' in piv['mae'].columns and b in piv['mae'].columns:
        dm = (piv['mae']['HDTM'] - piv['mae'][b]).dropna()
        dc = (piv['ccc']['HDTM'] - piv['ccc'][b]).dropna()
        dt = (piv['tail_mae']['HDTM'] - piv['tail_mae'][b]).dropna()
        summary[f'HDTM_minus_{b}'] = {'n': int(len(dm)), 'dmae': float(dm.mean()),
                                       'dmae_se': float(dm.std(ddof=1) / np.sqrt(len(dm))),
                                       'dccc': float(dc.mean()), 'dtail': float(dt.mean()),
                                       'wins': int((dm < 0).sum()), 'losses': int((dm > 0).sum())}
(P / 'm1_hdt_gate/formal15_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k not in ('means',)}, indent=2)[:3500])
