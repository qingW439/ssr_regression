"""Registry-driven OOF table. Formal and development protocols are separate."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools'))
from check_status import REGISTRY, check, ccc, manifest_holdout, unique_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=REGISTRY['protocol_units'], default='formal15')
    ap.add_argument('--version', default='historical')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    assets = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
    rows, failures = [], []
    for name, model in REGISTRY['models'].items():
        if model['protocol'] != args.protocol:
            continue
        for seed, fold in REGISTRY['protocol_units'][args.protocol]:
            pattern = model['historical'] if args.version == 'historical' else REGISTRY['new_run_template']
            run = assets / pattern.format(version=args.version, model=name, s=seed, f=fold)
            result = check(run, expected=model, seed=seed, fold=fold, teacher_guard=False, load_weight=False)
            if not result['METRICS_VERIFIED']:
                failures.append(f'{name} s{seed} f{fold}: {result["notes"]}')
                continue
            d = pd.read_parquet(run / 'oof_predictions.parquet')
            per = {}
            for axis in ('V', 'A', 'D'):
                y, p = d[f'y_{axis}'].to_numpy(), d[f'raw_{axis}'].to_numpy()
                q = np.quantile(y, [0.2, 0.8])
                tail = (y < q[0]) | (y > q[1])
                per[axis] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p),
                             'tail_mae': float(np.abs(y[tail] - p[tail]).mean() * 4)}
            rows.append({'model': name, 'protocol': args.protocol, 'version': args.version,
                         'seed': seed, 'fold': fold, 'metrics_verified': True,
                         'weight_present': (run / 'readout.pth').is_file(),
                         **{k: float(np.mean([x[k] for x in per.values()])) for k in ('mae', 'ccc', 'tail_mae')}})
    if args.protocol == 'formal15' and args.version == 'historical':
        for name, spec in REGISTRY['rankup'].items():
            for seed, fold in REGISTRY['protocol_units']['formal15']:
                run = assets / spec['run_template'].format(s=seed, f=fold)
                try:
                    config = json.loads((run / 'run_config.json').read_text())
                    if config['data_seed'] != seed or config['fold'] != fold:
                        raise ValueError('seed/fold mismatch')
                    for key, value in spec['config'].items():
                        if config.get(key) != value:
                            raise ValueError(f'config {key} mismatch')
                    d = pd.read_parquet(run / spec['score'])
                    ids = unique_ids(d['id'])
                    if len(ids) != len(manifest_holdout(seed, fold)) or set(ids) != manifest_holdout(seed, fold):
                        raise ValueError('fold IDs mismatch')
                    numeric = d[[f'{part}_{axis}' for part in ('y', 'raw') for axis in ('V', 'A', 'D')]].to_numpy()
                    if not np.isfinite(numeric).all():
                        raise ValueError('nonfinite OOF')
                    per = {}
                    for axis in ('V', 'A', 'D'):
                        y, p = d[f'y_{axis}'].to_numpy(), d[f'raw_{axis}'].to_numpy()
                        q = np.quantile(y, [0.2, 0.8])
                        tail = (y < q[0]) | (y > q[1])
                        per[axis] = {'mae': float(np.abs(y - p).mean() * 4), 'ccc': ccc(y, p),
                                     'tail_mae': float(np.abs(y[tail] - p[tail]).mean() * 4)}
                    rows.append({'model': name, 'protocol': args.protocol, 'version': spec['version'],
                                 'seed': seed, 'fold': fold, 'metrics_verified': True,
                                 'weight_present': (run / spec['weight']).is_file(),
                                 **{k: float(np.mean([x[k] for x in per.values()])) for k in ('mae', 'ccc', 'tail_mae')}})
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    failures.append(f'{name} s{seed} f{fold}: {exc}')
    if failures:
        print('\n'.join(failures), file=sys.stderr)
        raise SystemExit(f'{len(failures)} expected units failed validation')
    out = args.out or assets / f'm1_runs_unified/{args.version}/evaluation/{args.protocol}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    print(frame.groupby('model')[['mae', 'ccc', 'tail_mae']].mean().round(6).to_string())
    print(json.dumps({'protocol': args.protocol, 'units_per_model': frame.groupby('model').size().to_dict(), 'out': str(out)}))


if __name__ == '__main__':
    main()
