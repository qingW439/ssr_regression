"""RCUS-3D 15-fold aggregate vs U1 (same folds/seeds)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/rankup_project')
DIMS = ('V', 'A', 'D')


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def met(d):
    per = {}
    for x in DIMS:
        y, p = d[f'y_{x}'].to_numpy(), d[f'raw_{x}'].to_numpy()
        per[x] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p)}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'per_dim': per}


rows = []
for s in range(3):
    for f in range(5):
        for arm, d in (('RCUS', P / f'm1_rcus_runs/RCUS2_s{s}_f{f}'),
                       ('U1', P / f'm1_uul_runs/U1_s{s}_f{f}')):
            oof = d / 'oof_predictions.parquet'
            if not oof.is_file():
                rows.append({'arm': arm, 'seed': s, 'fold': f, 'missing': True})
                continue
            m = met(pd.read_parquet(oof))
            rows.append({'arm': arm, 'seed': s, 'fold': f, 'mae': m['mae'], 'ccc': m['ccc'],
                         **{f'{x}_mae': m['per_dim'][x]['mae'] for x in DIMS}})

df = pd.DataFrame(rows)
have = df[~df['missing'].fillna(False).astype(bool)] if 'missing' in df.columns else df
df.to_csv(P / 'm1_rcus_gate/rcus15.csv', index=False)
print(have.groupby('arm')[['mae', 'ccc']].agg(['mean', 'std']).round(6).to_string())
a = have.pivot(index=['seed', 'fold'], columns='arm', values='mae')
c = have.pivot(index=['seed', 'fold'], columns='arm', values='ccc')
summary = {'n_rcus': int((have['arm'] == 'RCUS').sum())}
if {'RCUS', 'U1'} <= set(a.columns):
    d = (a['RCUS'] - a['U1']).dropna()
    dc = (c['RCUS'] - c['U1']).dropna()
    summary['RCUS_minus_U1'] = {'n': int(len(d)), 'dmae': float(d.mean()),
                                'se': float(d.std(ddof=1) / np.sqrt(len(d))),
                                'dccc': float(dc.mean()),
                                'wins': int((d < 0).sum()), 'losses': int((d > 0).sum())}
(P / 'm1_rcus_gate/rcus15_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
