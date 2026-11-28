"""Aggregate SUP-6 / UUL-6 vs U1, with inference cost."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/rankup_project')
DIMS = ('V', 'A', 'D')
UNITS = [(0, 1), (1, 2), (2, 3)]
ARMS = [('C0', 'm1_uul_runs/U1_s{s}_f{f}'), ('SUP6', 'm1_shallow_runs/sup6_s{s}_f{f}'),
        ('UUL6', 'm1_shallow_runs/uul6_s{s}_f{f}')]


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def met(d):
    per = {}
    for x in DIMS:
        y, p = d[f'y_{x}'].to_numpy(), d[f'raw_{x}'].to_numpy()
        per[x] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p),
                  'pearson': float(np.corrcoef(y, p)[0, 1]),
                  'r2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                  'tail_mae': float(np.abs(y[y < np.quantile(y, 0.2)] - p[y < np.quantile(y, 0.2)]).mean() * 4)}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'pearson': float(np.mean([v['pearson'] for v in per.values()])),
            'r2': float(np.mean([v['r2'] for v in per.values()])), 'per_dim': per}


rows, infos = [], {}
for s, f in UNITS:
    for arm, pat in ARMS:
        d = P / pat.format(s=s, f=f)
        if not (d / 'oof_predictions.parquet').is_file():
            print('MISSING', arm, s, f)
            continue
        m = met(pd.read_parquet(d / 'oof_predictions.parquet'))
        r = {'arm': arm, 'seed': s, 'fold': f, **{k: m[k] for k in ('mae', 'ccc', 'pearson', 'r2')},
             **{f'{x}_mae': m['per_dim'][x]['mae'] for x in DIMS},
             **{f'{x}_ccc': m['per_dim'][x]['ccc'] for x in DIMS}}
        if arm != 'C0':
            rep = json.loads((d / 'result.json').read_text())
            r['n_params'] = rep.get('n_params')
            r['ms_per_batch64'] = rep['inference']['ms_per_batch']
            r['peak_mem_mb'] = rep['inference']['peak_mem_mb']
            infos[f'{arm}_s{s}_f{f}'] = rep['inference']
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(P / 'm1_shallow_gate/shallow_full.csv', index=False)
print(df.round(6).to_string(index=False))
g = df.groupby('arm')[['mae', 'ccc', 'pearson', 'r2']].mean()
print('\nMEANS\n', g.round(6).to_string())
a = df.pivot(index=['seed', 'fold'], columns='arm', values='mae')
c = df.pivot(index=['seed', 'fold'], columns='arm', values='ccc')
summary = {'means': g.round(8).to_dict(), 'inference': infos}
for arm in ('SUP6', 'UUL6'):
    if {arm, 'C0'} <= set(a.columns):
        summary[f'{arm}_minus_C0'] = {'dmae_mean': float((a[arm] - a['C0']).mean()),
                                      'dmae_per_unit': [float(v) for v in (a[arm] - a['C0'])],
                                      'dccc_mean': float((c[arm] - c['C0']).mean())}
if {'UUL6', 'SUP6'} <= set(a.columns):
    summary['UUL6_minus_SUP6'] = {'dmae_mean': float((a['UUL6'] - a['SUP6']).mean()),
                                  'dccc_mean': float((c['UUL6'] - c['SUP6']).mean()),
                                  'dmae_per_unit': [float(v) for v in (a['UUL6'] - a['SUP6'])]}
(P / 'm1_shallow_gate/shallow_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('\nSUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'inference'}, indent=2))
