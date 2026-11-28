#!/usr/bin/env python3
"""Completion checker: a run is complete only if weights + result + OOF + ids + config agree.

Usage:
  python tools/check_status.py --runs m1_hdt_runs/hdtm_s*_f* [--recompute]
  python tools/check_status.py --runs m1_aas --teacher-guard
Exit code 0 only if every checked run is complete.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('/root/autodl-tmp/rankup_project')
DIMS = ('V', 'A', 'D')


def manifest_holdout(seed, fold):
    p = R / ('Semi-Supervised-Regression-avrrc/outputs/p0' if seed == 0
             else f'Semi-Supervised-Regression-avrrc/outputs/p0_s{seed}') / 'fold_manifest.json'
    return set(json.loads(p.read_text())['fold_ids'][str(fold)])


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def check(run: Path, recompute: bool):
    res, notes = {}, []
    ok = True
    rj, oof, w = run / 'result.json', run / 'oof_predictions.parquet', run / 'readout.pth'
    res['weights'] = w.is_file()
    res['result'] = rj.is_file()
    res['oof'] = oof.is_file()
    if not res['result']:
        return {'run': str(run), 'ok': False, 'notes': ['missing result.json']}
    rep = json.loads(rj.read_text())
    seed, fold = rep.get('seed'), rep.get('fold')
    res['config'] = all(k in rep for k in ('seed', 'fold', 'arm')) or 'metrics' in rep
    if not res['oof']:
        return {'run': str(run), 'ok': False, 'notes': ['missing oof_predictions.parquet']}
    df = pd.read_parquet(oof)
    need = [f'y_{d}' for d in DIMS] + [f'raw_{d}' for d in DIMS]
    res['oof_columns'] = all(c in df.columns for c in need)
    if seed is not None and fold is not None and res['oof_columns']:
        hold = manifest_holdout(int(seed), int(fold))
        got = set(df['id'].astype(str))
        res['ids_match'] = got == hold
        if got != hold:
            notes.append(f'id mismatch: {len(got ^ hold)} symmetric difference')
    res['finite'] = bool(np.isfinite(df[[f'raw_{d}' for d in DIMS]].to_numpy()).all())
    out = {'run': str(run), 'ok': all(v for k, v in res.items() if k != 'oof_columns') and res.get('ids_match', True),
           'checks': res, 'notes': notes}
    if recompute:
        maes, cccs = [], []
        for d in DIMS:
            y, p = df[f'y_{d}'].to_numpy(), df[f'raw_{d}'].to_numpy()
            maes.append(float(np.abs(y - p).mean() * 4))
            cccs.append(ccc(y, p))
        out['mae'] = round(float(np.mean(maes)), 6)
        out['ccc'] = round(float(np.mean(cccs)), 6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='globs, e.g. m1_hdt_runs/hdtm_s*_f*')
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--teacher-guard', action='store_true',
                    help='also require all 4 teacher caches and the fused target for a fold')
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()
    runs = []
    for g in a.runs:
        runs += [Path(p) for p in sorted(glob.glob(str(R / g)))] if '*' in g else [R / g]
    runs = [r for r in runs if r.is_dir()]
    reports = []
    for r in runs:
        rep = check(r, a.recompute)
        if a.teacher_guard and rep['ok']:
            m = pd.read_parquet(r / 'oof_predictions.parquet')['id']
            fold = int(r.name.rsplit('_f', 1)[-1])
            seed = int(r.name.split('_s')[-1].split('_f')[0])
            teachers = [R / f'm1_hdt_gate/teachers/B_s{seed}_f{fold}_j{j}.npz' for j in range(4)]
            fused = R / f'm1_hdt_gate/fused/B_s{seed}_f{fold}.npz'
            miss = [t.name for t in teachers if not t.is_file()]
            rep['teachers'] = {'n': sum(t.is_file() for t in teachers), 'missing': miss,
                               'fused': fused.is_file()}
            rep['ok'] = rep['ok'] and not miss and fused.is_file()
            if miss:
                rep['notes'].append(f'missing teacher caches: {miss}')
        reports.append(rep)
        print(json.dumps(rep, default=str))
    bad = [r for r in reports if not r['ok']]
    print(f'\nSUMMARY runs={len(reports)} complete={len(reports) - len(bad)} incomplete={len(bad)}')
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(reports, indent=2) + '\n')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
