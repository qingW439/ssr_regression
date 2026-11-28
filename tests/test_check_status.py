"""Failure and success cases for the artifact gate."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import check_status as status  # noqa: E402


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(status, 'ASSETS', tmp_path)
    manifest = tmp_path / 'Semi-Supervised-Regression-avrrc/outputs/p0/fold_manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'fold_ids': {'0': ['a', 'b', 'c']}}))
    run = tmp_path / 'run'
    run.mkdir()
    df = pd.DataFrame({'id': ['a', 'b', 'c'], **{f'y_{d}': [0.2, 0.4, 0.6] for d in status.DIMS},
                       **{f'raw_{d}': [0.3, 0.5, 0.7] for d in status.DIMS}})
    df.to_parquet(run / 'oof_predictions.parquet')
    (run / 'result.json').write_text(json.dumps({'seed': 0, 'fold': 0, 'arm': 'hs',
        'config': {'seed': 0, 'fold': 0, 'arm': 'hs'}, 'metrics': {'mae': 0.4, 'ccc': status.ccc(df.y_V.to_numpy(), df.raw_V.to_numpy())}}))
    return run, df


def test_complete_and_metrics_without_weights(artifact):
    run, _ = artifact
    result = status.check(run, expected={'arm': 'hs'}, seed=0, fold=0)
    assert result['METRICS_VERIFIED'] and not result['INFERENCE_READY'] and not result['RUN_COMPLETE']
    import torch
    torch.save({'model': {'x': torch.ones(1)}}, run / 'readout.pth')
    result = status.check(run, expected={'arm': 'hs'}, seed=0, fold=0)
    assert result['RUN_COMPLETE']


@pytest.mark.parametrize('change', ['missing_run', 'missing_oof', 'duplicate_ids', 'wrong_ids',
                                    'nan_label', 'nan_prediction', 'wrong_config', 'wrong_metric'])
def test_invalid_artifacts_fail(artifact, change):
    run, df = artifact
    if change == 'missing_run':
        run = run / 'absent'
    elif change == 'missing_oof':
        (run / 'oof_predictions.parquet').unlink()
    elif change in ('duplicate_ids', 'wrong_ids', 'nan_label', 'nan_prediction'):
        if change == 'duplicate_ids': df.loc[1, 'id'] = 'a'
        if change == 'wrong_ids': df.loc[1, 'id'] = 'z'
        if change == 'nan_label': df.loc[1, 'y_V'] = np.nan
        if change == 'nan_prediction': df.loc[1, 'raw_V'] = np.nan
        df.to_parquet(run / 'oof_predictions.parquet')
    else:
        result = json.loads((run / 'result.json').read_text())
        if change == 'wrong_config': result['config']['arm'] = 'hu'
        if change == 'wrong_metric': result['metrics']['mae'] = 0.1
        (run / 'result.json').write_text(json.dumps(result))
    assert not status.check(run, expected={'arm': 'hs'}, seed=0, fold=0)['METRICS_VERIFIED']


def test_teacher_missing_and_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(status, 'ASSETS', tmp_path)
    assert status.check_teacher(0, 0)
    root = tmp_path / 'm1_v2_assets'
    (root / 'splits/B_s0_f0').mkdir(parents=True)
    (root / 'splits/B_s0_f0/split.json').write_text(json.dumps({'groups': [{'hold_ids': ['h']}]*4}))
    np.save(root / 'u_ids.npy', np.array(['u']))
    directory = tmp_path / 'm1_hdt_gate/teachers'
    directory.mkdir(parents=True)
    for j in range(4):
        np.savez(directory / f'B_s0_f0_j{j}.npz', hold_ids=['h'], u_ids=['u'],
                 u_probs=np.full((1, 3, 20), 0.05), u_pred=np.full((1, 3), 0.5))
    assert not status.check_teacher(0, 0, fused=False)
    np.savez(directory / 'B_s0_f0_j0.npz', hold_ids=['h'], u_ids=['u'],
             u_probs=np.full((1, 3, 20), np.nan), u_pred=np.full((1, 3), 0.5))
    assert status.check_teacher(0, 0, fused=False)


def test_missing_glob_and_expected_formal_units(tmp_path, monkeypatch):
    monkeypatch.setattr(status, 'ASSETS', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['check_status.py', '--runs', 'missing*'])
    assert status.main() == 1
    monkeypatch.setattr(sys, 'argv', ['check_status.py', '--experiment', 'HDT-M', '--metrics-only'])
    assert status.main() == 1


def test_corrupt_oof_and_weight(artifact):
    run, _ = artifact
    (run / 'oof_predictions.parquet').write_bytes(b'corrupt')
    assert not status.check(run, expected={'arm': 'hs'})['METRICS_VERIFIED']
    (run / 'readout.pth').write_bytes(b'corrupt')
    assert not status.check(run, expected={'arm': 'hs'})['INFERENCE_READY']
