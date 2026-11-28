"""Phase 8 verification: load a formal HDT-M checkpoint, reproduce OOF predictions, run decoder tests."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/root/autodl-tmp/rankup_project')
sys.path.insert(0, str(ROOT / 'ssr_regression/src/hdtm'))
sys.path.insert(0, str(ROOT / 'Semi-Supervised-Regression-comp2'))
from train_hdt import HistVAD, predict_texts  # noqa: E402
from semilearn.datasets.nlp_datasets.emobank import read_gold  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def main():
    dev = torch.device('cpu')
    run = ROOT / 'm1_hdt_runs/hdtm_s0_f0'
    ck = torch.load(run / 'readout.pth', map_location='cpu', weights_only=False)
    model = HistVAD().to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    tok = AutoTokenizer.from_pretrained(str(ROOT / 'pretrained/roberta-base-simcse'),
                                        local_files_only=True, use_fast=True)
    lb = read_gold(ROOT / 'data/emobank_processed/lb500_s0/train_lb.parquet', expected_n=500)
    man = json.loads((ROOT / 'Semi-Supervised-Regression-avrrc/outputs/p0/fold_manifest.json').read_text())
    hold = set(man['fold_ids']['0'])
    ho = lb[lb['id'].isin(hold)].reset_index(drop=True)
    oof = pd.read_parquet(run / 'oof_predictions.parquet').set_index('id')
    texts = ho['text'].tolist()[:4]
    pred = predict_texts(model, texts, tok, dev, bs=4)
    ids = ho['id'].tolist()[:4]
    maxdiff = 0.0
    for i, tid in enumerate(ids):
        ref = np.array([oof.at[tid, 'raw_V'], oof.at[tid, 'raw_A'], oof.at[tid, 'raw_D']], dtype=np.float32)
        d = float(np.abs(pred[i] - ref).max())
        maxdiff = max(maxdiff, d)
        print(f'{tid[:28]:30s} pred={np.round(pred[i], 5).tolist()} oof={np.round(ref, 5).tolist()} maxdiff={d:.2e}')
    print('VERIFY_MAXDIFF', maxdiff)
    print('VERIFY_OK' if maxdiff < 1e-4 else 'VERIFY_FAIL')


if __name__ == '__main__':
    main()
