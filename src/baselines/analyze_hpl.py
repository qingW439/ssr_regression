"""HPL-3D formal 15-fold aggregate vs U1 (same folds/seeds)."""
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
        q = np.quantile(y, [0.2, 0.8])
        tail = (y < q[0]) | (y > q[1])
        per[x] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p),
                  'tail_mae': float(np.abs(y[tail] - p[tail]).mean() * 4)}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'tail_mae': float(np.mean([v['tail_mae'] for v in per.values()])),
            'per_dim': per}


rows = []
for s in range(3):
    for f in range(5):
        for arm, base in (('HPL', P / f'm1_hpl_runs/HPL_s{s}_f{f}'),
                          ('U1', P / f'm1_uul_runs/U1_s{s}_f{f}')):
            oof = base / 'oof_predictions.parquet'
            if not oof.is_file():
                rows.append({'arm': arm, 'seed': s, 'fold': f, 'missing': True})
                continue
            m = met(pd.read_parquet(oof))
            rows.append({'arm': arm, 'seed': s, 'fold': f, 'mae': m['mae'], 'ccc': m['ccc'],
                         'tail_mae': m['tail_mae'],
                         **{f'{x}_mae': m['per_dim'][x]['mae'] for x in DIMS},
                         **{f'{x}_ccc': m['per_dim'][x]['ccc'] for x in DIMS}})

df = pd.DataFrame(rows)
have = df[~df['missing'].fillna(False).astype(bool)] if 'missing' in df.columns else df
df.to_csv(P / 'm1_hpl_gate/hpl_full.csv', index=False)
print(df.round(6).to_string(index=False))
print('\nMEANS\n', have.groupby('arm')[['mae', 'ccc', 'tail_mae']].mean().round(6).to_string())
a = have.pivot(index=['seed', 'fold'], columns='arm', values='mae')
c = have.pivot(index=['seed', 'fold'], columns='arm', values='ccc')
summary = {'n_units': int((have['arm'] == 'HPL').sum()),
           'means': have.groupby('arm')[['mae', 'ccc', 'tail_mae']].mean().round(8).to_dict()}
if {'HPL', 'U1'} <= set(a.columns):
    d = (a['HPL'] - a['U1']).dropna()
    dc = (c['HPL'] - c['U1']).dropna()
    summary['HPL_minus_U1'] = {'units': int(len(d)), 'dmae_mean': float(d.mean()),
                               'dmae_sd': float(d.std(ddof=1)), 'dmae_se': float(d.std(ddof=1) / np.sqrt(len(d))),
                               'wins': int((d < 0).sum()), 'losses': int((d > 0).sum()),
                               'dccc_mean': float(dc.mean()),
                               'dccc_se': float(dc.std(ddof=1) / np.sqrt(len(dc)))}
(P / 'm1_hpl_gate/hpl_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('\nSUMMARY', json.dumps(summary, indent=2))
