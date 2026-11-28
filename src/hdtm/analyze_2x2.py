"""2x2 teacher-structure x student-structure interaction on the 3 development units."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

P = Path('/root/autodl-tmp/rankup_project')
DIMS = ('V', 'A', 'D')
UNITS = [(0, 1), (1, 2), (2, 3)]
PATHS = {'U1': 'm1_uul_runs/U1_s{s}_f{f}', 'HU': 'm1_hist_runs/hu_s{s}_f{f}',
         'THS': 'm1_hdt_runs/ths_s{s}_f{f}', 'HDTM': 'm1_hdt_runs/hdtm_s{s}_f{f}',
         'HS': 'm1_hist_runs/hs_s{s}_f{f}', 'HPL': 'm1_hpl_runs/HPL_s{s}_f{f}'}


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
                  'sd_ratio': float(p.std() / y.std())}
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
df.to_csv(P / 'm1_hdt_gate/ths_2x2.csv', index=False)
print(df.round(6).to_string(index=False))
means = df.groupby('arm')[['mae', 'ccc', 'tail_mae', 'sd_ratio']].mean().round(6)
print('\nMEANS\n', means.to_string())

res = {'means': means.to_dict()}
if {'U1', 'THS', 'HU', 'HDTM'} <= set(df.arm.unique()):
    g_scalar = float(means.loc['U1', 'mae'] - means.loc['THS', 'mae'])
    g_dist = float(means.loc['HU', 'mae'] - means.loc['HDTM', 'mae'])
    res['teacher_gain_scalar_student'] = g_scalar
    res['teacher_gain_dist_student'] = g_dist
    res['interaction_mae'] = g_dist - g_scalar
    for k in ('ccc', 'tail_mae', 'sd_ratio'):
        gs = float(means.loc['THS', k] - means.loc['U1', k])
        gd = float(means.loc['HDTM', k] - means.loc['HU', k])
        res[f'gain_scalar_{k}'] = gs
        res[f'gain_dist_{k}'] = gd
        res[f'interaction_{k}'] = gd - gs
    # per-unit interaction
    pu = []
    for s, f in UNITS:
        a = float(df[(df.arm == 'U1') & (df.seed == s) & (df.fold == f)].mae.iloc[0])
        b = float(df[(df.arm == 'THS') & (df.seed == s) & (df.fold == f)].mae.iloc[0])
        c = float(df[(df.arm == 'HU') & (df.seed == s) & (df.fold == f)].mae.iloc[0])
        d = float(df[(df.arm == 'HDTM') & (df.seed == s) & (df.fold == f)].mae.iloc[0])
        pu.append({'unit': f's{s}f{f}', 'gain_scalar': a - b, 'gain_dist': c - d,
                   'interaction': (c - d) - (a - b)})
    res['per_unit'] = pu
print(json.dumps(res, indent=2))
(P / 'm1_hdt_gate/ths_2x2_summary.json').write_text(json.dumps(res, indent=2) + '\n')
