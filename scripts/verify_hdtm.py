import os
import sys
from pathlib import Path

REPO = Path('/root/autodl-tmp/rankup_project/ssr_regression')
sys.path.insert(0, str(REPO / 'src/hdtm'))
sys.path.insert(0, str(REPO))
from train_hdt import HistVAD, predict_texts  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from semilearn.datasets.nlp_datasets.emobank import read_gold  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

A = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
dev = torch.device('cuda')
run = A / 'm1_hdt_runs/hdtm_s0_f0'
ck = torch.load(run / 'readout.pth', map_location='cpu', weights_only=False)
model = HistVAD().to(dev)
model.load_state_dict(ck['model'])
model.eval()
tok = AutoTokenizer.from_pretrained(str(A / 'pretrained/roberta-base-simcse'), local_files_only=True, use_fast=True)
lb = read_gold(A / 'data/emobank_processed/lb500_s0/train_lb.parquet', expected_n=500)
man = json.loads((A / 'Semi-Supervised-Regression-avrrc/outputs/p0/fold_manifest.json').read_text())
ho = lb[lb['id'].isin(set(man['fold_ids']['0']))].reset_index(drop=True)
oof = pd.read_parquet(run / 'oof_predictions.parquet').set_index('id')
pred = predict_texts(model, ho['text'].tolist()[:4], tok, dev, bs=4)
md = 0.0
for i, tid in enumerate(ho['id'].tolist()[:4]):
    ref = np.array([oof.at[tid, 'raw_V'], oof.at[tid, 'raw_A'], oof.at[tid, 'raw_D']], dtype=np.float32)
    md = max(md, float(np.abs(pred[i] - ref).max()))
print('maxdiff', md, 'OK' if md < 1e-4 else 'FAIL')
