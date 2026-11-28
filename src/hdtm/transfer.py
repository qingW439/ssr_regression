"""Teacher -> student transfer: per-fold teacher quality vs HDT-M student quality."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
ROOT = ASSETS
sys.path.insert(0, str(REPO / 'src/hdtm'))
DIMS = ('V', 'A', 'D')


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


rows = []
for s in range(3):
    for f in range(5):
        old_mae, new_mae, new_ccc, old_ccc = [], [], [], []
        for j in range(4):
            o = np.load(ROOT / f'm1_v2_assets/xfit/B_s{s}_f{f}_j{j}.npz', allow_pickle=True)
            n = np.load(ROOT / f'm1_hdt_gate/teachers/B_s{s}_f{f}_j{j}.npz', allow_pickle=True)
            y = o['hold_y']
            for pred, acc, acc_c in ((o['hold_pred'], old_mae, old_ccc), (n['hold_pred'], new_mae, new_ccc)):
                acc.append(float(np.abs(pred - y).mean() * 4))
                acc_c.append(float(np.mean([ccc(y[:, i], pred[:, i]) for i in range(3)])))
        st = json.loads((ROOT / f'm1_hdt_runs/hdtm_s{s}_f{f}/result.json').read_text())['metrics']
        rows.append({'seed': s, 'fold': f, 'teacher_old_mae': float(np.mean(old_mae)),
                     'teacher_new_mae': float(np.mean(new_mae)),
                     'teacher_new_ccc': float(np.mean(new_ccc)),
                     'teacher_old_ccc': float(np.mean(old_ccc)),
                     'student_mae': st['mae'], 'student_ccc': st['ccc'],
                     'student_tail': st['tail_mae'], 'student_sd': st['sd_ratio']})
df = pd.DataFrame(rows)
df.to_csv(ROOT / 'm1_hdt_gate/transfer.csv', index=False)
print(df.round(5).to_string(index=False))
out = {}
for a, b in (('teacher_new_mae', 'student_mae'), ('teacher_new_ccc', 'student_ccc'),
             ('teacher_old_mae', 'student_mae'), ('teacher_new_mae', 'student_tail')):
    r, p = pearsonr(df[a], df[b])
    out[f'{a}_vs_{b}'] = {'pearson': float(r), 'p': float(p), 'spearman': float(spearmanr(df[a], df[b]).statistic)}
print(json.dumps(out, indent=2))
out['corner'] = {'teacher_new_mae_range': [float(df.teacher_new_mae.min()), float(df.teacher_new_mae.max())],
                 'student_mae_range': [float(df.student_mae.min()), float(df.student_mae.max())]}
(ROOT / 'm1_report/TRANSFER.json').write_text(json.dumps(out, indent=2) + '\n')
