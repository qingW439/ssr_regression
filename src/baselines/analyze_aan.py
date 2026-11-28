import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/rankup_project')
OUT = P / 'm1_aan_gate'
DIMS = ['V', 'A', 'D']


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def met(df, prefix):
    per = {}
    for d in DIMS:
        y = df[f'y_{d}'].to_numpy(); p = df[f'{prefix}_{d}'].to_numpy()
        per[d] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p),
                  'pearson': float(np.corrcoef(y, p)[0, 1]),
                  'r2': float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    return {'macro_mae': float(np.mean([per[d]['mae'] for d in DIMS])),
            'macro_ccc': float(np.mean([per[d]['ccc'] for d in DIMS])),
            'macro_pearson': float(np.mean([per[d]['pearson'] for d in DIMS])),
            'macro_r2': float(np.mean([per[d]['r2'] for d in DIMS])),
            'per_dim': per}


def collect(pattern):
    seeds = []
    for s in range(3):
        dfs = []
        for f in range(5):
            p = Path(pattern.format(s=s, f=f))
            if p.exists():
                dfs.append(pd.read_parquet(p))
        if len(dfs) == 5:
            seeds.append(pd.concat(dfs, ignore_index=True))
    return seeds


rows, oof = [], []
for method, pat in [('AAN', str(P / 'm1_aan_runs/aan_s{s}_f{f}/oof_predictions.parquet')),
                    ('U0', str(P / 'm1_uul_runs/U0_s{s}_f{f}/oof_predictions.parquet')),
                    ('U1', str(P / 'm1_uul_runs/U1_s{s}_f{f}/oof_predictions.parquet'))]:
    seeds = collect(pat)
    if len(seeds) < 3:
        print(method, 'incomplete', len(seeds)); continue
    for s, d in enumerate(seeds):
        d = d.copy(); d['arm'] = method; d['seed'] = s
        oof.append(d)
        for level, prefix in [('base', 'base'), ('readout', 'raw')]:
            m = met(d, prefix)
            rows.append({'arm': method, 'seed': s, 'level': level,
                         'macro_mae': m['macro_mae'], 'macro_ccc': m['macro_ccc'],
                         'macro_pearson': m['macro_pearson'], 'macro_r2': m['macro_r2'],
                         **{f'{x}_mae': m['per_dim'][x]['mae'] for x in DIMS},
                         **{f'{x}_ccc': m['per_dim'][x]['ccc'] for x in DIMS}})

per = pd.DataFrame(rows)
per.to_csv(OUT / 'aan_results.csv', index=False)
if oof:
    pd.concat(oof, ignore_index=True).to_parquet(OUT / 'aan_oof.parquet', index=False)

print('=== per-seed pooled, mean of 3 seeds ===')
print(per[per.level == 'readout'].groupby('arm')[['macro_mae', 'macro_ccc', 'macro_pearson', 'macro_r2']].mean().round(6).to_string())
print('\n=== per seed MAE ===')
print(per[per.level == 'readout'].pivot_table(index='seed', columns='arm', values='macro_mae').round(6).to_string())
print('\n=== base ===')
print(per[per.level == 'base'].groupby('arm')[['macro_mae', 'macro_ccc']].mean().round(6).to_string())

for a, b in [('AAN', 'U0'), ('AAN', 'U1')]:
    aa = per[(per.arm == a) & (per.level == 'readout')].set_index('seed')
    bb = per[(per.arm == b) & (per.level == 'readout')].set_index('seed')
    if len(aa) and len(bb):
        d = (aa.macro_mae - bb.macro_mae)
        print(f'{a}-{b}: mean {d.mean():+.6f} per_seed {[round(x,6) for x in d]}')

(OUT / 'aan_summary.json').write_text(json.dumps({'rows': rows}, indent=1))
print('wrote aan_results.csv / aan_oof.parquet / aan_summary.json')
