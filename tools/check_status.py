#!/usr/bin/env python3
"""Check expected experiment artifacts; any missing or invalid unit exits nonzero."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
REGISTRY = json.loads((REPO / 'experiments/registry.json').read_text())
DIMS = ('V', 'A', 'D')


def manifest_holdout(seed, fold):
    root = ASSETS / 'Semi-Supervised-Regression-avrrc/outputs'
    path = root / ('p0' if seed == 0 else f'p0_s{seed}') / 'fold_manifest.json'
    return set(map(str, json.loads(path.read_text())['fold_ids'][str(fold)]))


def ccc(y, p):
    my, mp = y.mean(), p.mean()
    return float(2 * ((y - my) * (p - mp)).mean() / (y.var() + p.var() + (my - mp) ** 2 + 1e-12))


def unique_ids(values):
    ids = [str(x) for x in values]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate IDs')
    return ids


def check_teacher(seed, fold, fused=True, teacher=None):
    """Validate teacher arrays, split IDs, unlabeled order and fused mean."""
    issues = []
    try:
        split = json.loads((ASSETS / f'm1_v2_assets/splits/B_s{seed}_f{fold}/split.json').read_text())
        canonical = unique_ids(np.load(ASSETS / 'm1_v2_assets/u_ids.npy', allow_pickle=True))
        arrays = []
        for j in (range(4) if teacher is None else [teacher]):
            path = ASSETS / f'm1_hdt_gate/teachers/B_s{seed}_f{fold}_j{j}.npz'
            with np.load(path, allow_pickle=True) as z:
                if set(unique_ids(z['hold_ids'])) != set(map(str, split['groups'][j]['hold_ids'])):
                    issues.append(f'teacher {j}: hold IDs differ from split')
                if unique_ids(z['u_ids']) != canonical:
                    issues.append(f'teacher {j}: unlabeled IDs/order differ')
                p, v = z['u_probs'], z['u_pred']
                if p.shape != (len(canonical), 3, 20) or v.shape != (len(canonical), 3):
                    issues.append(f'teacher {j}: wrong shape')
                elif not np.isfinite(p).all() or not np.isfinite(v).all() or (p < 0).any() or not np.allclose(p.sum(-1), 1, atol=1e-3):
                    issues.append(f'teacher {j}: invalid probabilities or values')
                arrays.append(p)
        if fused:
            if teacher is not None:
                raise ValueError('fused validation requires all four teachers')
            with np.load(ASSETS / f'm1_hdt_gate/fused/B_s{seed}_f{fold}.npz', allow_pickle=True) as z:
                if unique_ids(z['u_ids']) != canonical:
                    issues.append('fused: unlabeled IDs/order differ')
                p, v = z['u_probs'], z['u_value']
                if p.shape != (len(canonical), 3, 20) or v.shape != (len(canonical), 3):
                    issues.append('fused: wrong shape')
                elif not np.isfinite(p).all() or not np.isfinite(v).all() or not np.allclose(p, np.mean(arrays, axis=0), atol=1e-5):
                    issues.append('fused: invalid or differs from teacher mean')
    except Exception as exc:  # malformed zip/npy payloads can raise several decoder errors
        issues.append(f'teacher cache: {exc}')
    return issues


def check(run, recompute=False, expected=None, seed=None, fold=None, teacher_guard=False, load_weight=True):
    run = Path(run)
    issues = []
    report = {'run': str(run), 'METRICS_VERIFIED': False, 'INFERENCE_READY': False, 'RUN_COMPLETE': False, 'notes': issues}
    if not run.is_dir():
        issues.append('missing run directory')
        return report
    result, oof, weight = (run / x for x in ('result.json', 'oof_predictions.parquet', 'readout.pth'))
    rep = None
    config_ok = False
    try:
        rep = json.loads(result.read_text())
        rs, rf = int(rep['seed']), int(rep['fold'])
        cfg = rep['config']
        if not isinstance(cfg, dict) or int(cfg['seed']) != rs or int(cfg['fold']) != rf:
            raise ValueError('config seed/fold mismatch')
        if (seed is not None and rs != seed) or (fold is not None and rf != fold):
            raise ValueError('seed/fold differ from expected unit')
        if expected:
            fields = {key: expected[key] for key in ('arm', 'implementation', 'lambda_u', 'n_layers', 'lam') if key in expected}
            if expected.get('protocol'):
                fields.update(REGISTRY['common_config'])
            for key, value in fields.items():
                if cfg.get(key, rep.get(key)) != value:
                    raise ValueError(f'config {key} differs from registry')
            if not any(k in cfg or k in rep for k in ('arm', 'implementation')) and 'implementation' not in expected:
                raise ValueError('missing arm')
        elif not any(k in cfg or k in rep for k in ('arm', 'implementation')):
            raise ValueError('missing arm; select documented --experiment')
        if not isinstance(rep.get('metrics'), dict) or 'mae' not in rep['metrics']:
            raise ValueError('missing metrics')
        config_ok = True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        issues.append(f'invalid result/config: {exc}')
    if rep is not None and 'seed' in rep and 'fold' in rep:
        try:
            df = pd.read_parquet(oof)
            columns = ['id'] + [f'{p}_{d}' for p in ('y', 'raw') for d in DIMS]
            if any(c not in df for c in columns):
                raise ValueError('missing OOF columns')
            ids = unique_ids(df['id'])
            hold = manifest_holdout(int(rep['seed']), int(rep['fold']))
            if len(df) != len(hold) or set(ids) != hold:
                raise ValueError(f'OOF IDs differ from fold manifest: rows={len(df)}, expected={len(hold)}')
            if not np.isfinite(df[columns[1:]].to_numpy(dtype=float)).all():
                raise ValueError('nonfinite labels or predictions')
            mae = float(np.mean([np.abs(df[f'y_{d}'] - df[f'raw_{d}']).mean() * 4 for d in DIMS]))
            score_ccc = float(np.mean([ccc(df[f'y_{d}'].to_numpy(), df[f'raw_{d}'].to_numpy()) for d in DIMS]))
            if abs(mae - float(rep['metrics']['mae'])) > 1e-4:
                raise ValueError('stored MAE differs from OOF recomputation')
            if 'ccc' in rep['metrics'] and abs(score_ccc - float(rep['metrics']['ccc'])) > 1e-4:
                raise ValueError('stored CCC differs from OOF recomputation')
            report['METRICS_VERIFIED'] = config_ok
            if recompute:
                report['mae'] = round(mae, 6)
                report['ccc'] = round(score_ccc, 6)
        except Exception as exc:  # parquet decoders use implementation-specific errors
            issues.append(f'invalid OOF: {exc}')
    if weight.is_file():
        try:
            if load_weight:
                import torch
                state = torch.load(weight, map_location='cpu', weights_only=True, mmap=True)
                if not isinstance(state, dict) or not state:
                    raise ValueError('invalid state dict')
                report['INFERENCE_READY'] = True
            else:
                report['INFERENCE_READY'] = None
        except Exception as exc:  # invalid checkpoint formats vary across torch versions
            issues.append(f'invalid weight: {exc}')
    else:
        issues.append('missing readout.pth')
    if teacher_guard and rep is not None:
        issues.extend(check_teacher(int(rep['seed']), int(rep['fold'])))
    report['RUN_COMPLETE'] = bool(report['METRICS_VERIFIED'] and report['INFERENCE_READY'] and not issues)
    report['ok'] = report['RUN_COMPLETE']
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--runs', nargs='+')
    group.add_argument('--experiment', choices=REGISTRY['models'])
    ap.add_argument('--version', default='historical')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--fold', type=int)
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--teacher-guard', action='store_true')
    ap.add_argument('--metrics-only', action='store_true')
    ap.add_argument('--no-weight-load', action='store_true')
    ap.add_argument('--json-out')
    a = ap.parse_args()
    jobs = []
    if a.experiment:
        model = dict(REGISTRY['models'][a.experiment])
        if a.version != 'historical' and 'new_arm' in model:
            model['arm'] = model.pop('new_arm')
        for s, f in REGISTRY['protocol_units'][model['protocol']]:
            if (a.seed is not None and s != a.seed) or (a.fold is not None and f != a.fold):
                continue
            pattern = model['historical'] if a.version == 'historical' else REGISTRY['new_run_template']
            jobs.append((ASSETS / pattern.format(version=a.version, model=a.experiment, s=s, f=f), model, s, f))
    else:
        for pattern in a.runs:
            path = Path(pattern)
            if not path.is_absolute():
                path = ASSETS / path
            matches = sorted(map(Path, glob.glob(str(path)))) if glob.has_magic(str(path)) else [path]
            jobs.extend((p, None, None, None) for p in (matches or [path]))
    reports = []
    for path, model, s, f in jobs:
        rep = check(path, a.recompute, model, s, f, a.teacher_guard or bool(model and model.get('teacher_guard')), not a.no_weight_load)
        reports.append(rep)
        print(json.dumps(rep, ensure_ascii=False))
    good = sum(r['METRICS_VERIFIED'] if a.metrics_only else r['RUN_COMPLETE'] for r in reports)
    print(f'SUMMARY runs={len(reports)} passed={good} failed={len(reports)-good}')
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(reports, indent=2, ensure_ascii=False) + '\n')
    return 0 if reports and good == len(reports) else 1


if __name__ == '__main__':
    sys.exit(main())
