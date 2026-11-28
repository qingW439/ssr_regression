"""HDT-M / HDT-D vs H-S / H-U / U1 / HPL on the three development units."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

P = Path('/root/autodl-tmp/rankup_project')
DIMS = ('V', 'A', 'D')
UNITS = [(0, 1), (1, 2), (2, 3)]
PATHS = {'U1': 'm1_uul_runs/U1_s{s}_f{f}', 'HS': 'm1_hist_runs/hs_s{s}_f{f}',
         'HU': 'm1_hist_runs/hu_s{s}_f{f}', 'HPL': 'm1_hpl_runs/HPL_s{s}_f{f}',
         'HDTM': 'm1_hdt_runs/hdtm_s{s}_f{f}', 'HDTD': 'm1_hdt_runs/hdtd_s{s}_f{f}'}


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
                  'tail_mae': float(np.abs(y[tail] - p[tail]).mean() * 4),
                  'sd_ratio': float(p.std() / (y.std() + 1e-12)),
                  'srcc': float(spearmanr(y, p).statistic)}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'tail_mae': float(np.mean([v['tail_mae'] for v in per.values()])),
            'sd_ratio': float(np.mean([v['sd_ratio'] for v in per.values()])),
            'per_dim': per}


rows = []
for s, f in UNITS:
    for arm, pat in PATHS.items():
        oof = P / pat.format(s=s, f=f) / 'oof_predictions.parquet'
        if not oof.is_file():
            print('MISSING', arm, s, f)
            continue
        m = met(pd.read_parquet(oof))
        rows.append({'arm': arm, 'seed': s, 'fold': f, 'mae': m['mae'], 'ccc': m['ccc'],
                     'tail_mae': m['tail_mae'], 'sd_ratio': m['sd_ratio'],
                     **{f'{x}_mae': m['per_dim'][x]['mae'] for x in DIMS}})
df = pd.DataFrame(rows)
df.to_csv(P / 'm1_hdt_gate/hdt_dev3.csv', index=False)
print(df.round(6).to_string(index=False))
print('\nMEANS\n', df.groupby('arm')[['mae', 'ccc', 'tail_mae', 'sd_ratio']].mean().round(6).to_string())

piv = {k: df.pivot(index=['seed', 'fold'], columns='arm', values=k) for k in ('mae', 'ccc')}
summary = {'means': df.groupby('arm')[['mae', 'ccc', 'tail_mae', 'sd_ratio']].mean().round(8).to_dict()}
for a, b in (('HDTM', 'HU'), ('HDTD', 'HDTM'), ('HDTD', 'HS'), ('HDTD', 'HPL'), ('HDTM', 'HS'), ('HDTD', 'HU')):
    if a in piv['mae'].columns and b in piv['mae'].columns:
        dm = (piv['mae'][a] - piv['mae'][b]).dropna()
        dc = (piv['ccc'][a] - piv['ccc'][b]).dropna()
        summary[f'{a}_minus_{b}'] = {'n': int(len(dm)), 'dmae': float(dm.mean()),
                                      'per_unit': [float(x) for x in dm], 'dccc': float(dc.mean())}
(P / 'm1_hdt_gate/hdt_dev3_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2)[:3000])
