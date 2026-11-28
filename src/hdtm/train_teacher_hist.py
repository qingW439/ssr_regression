#!/usr/bin/env python3
"""Distribution-regression teacher (HDT): same cross-fit split/budget as the b2 teacher,
but with the 60-bin histogram head + CE + boundary-corrected decoding.

Matched to the original teacher recipe (m1_recovery/xfit): 3200 steps, batch 16,
lr 1e-5, layer decay 0.75, then the same 96-step supervised readout (CE here),
predicting holdout / query / unlabeled exactly like m1_v2/xfit_predict.py.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
ROOT = ASSETS
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src/hdtm'))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor, set_seed  # noqa: E402
from hist_loss import K, HistHead, decode_corrected, hist_loss  # noqa: E402

DATA = ROOT / 'data/emobank_processed'
UVR = ROOT / 'pretrained/roberta-base-simcse'
A = ROOT / 'm1_v2_assets'


class HistVAD(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        base = RobertaVADRegressor(str(UVR), dropout=dropout, offline=True)
        self.encoder = base.encoder
        self.dropout = base.dropout
        self.head = HistHead(self.encoder.config.hidden_size)

    def forward(self, **inputs):
        states = self.encoder(**inputs, return_dict=True).last_hidden_state
        m = inputs['attention_mask'].unsqueeze(-1).to(states.dtype)
        pooled = (states * m).sum(1) / m.sum(1).clamp_min(1)
        return self.head(self.dropout(pooled))


def param_groups(model, base_lr, decay):
    n = len(model.encoder.encoder.layer)
    groups = [{'params': list(model.encoder.embeddings.parameters()), 'lr': base_lr * (decay ** n)}]
    for i, layer in enumerate(model.encoder.encoder.layer):
        groups.append({'params': list(layer.parameters()), 'lr': base_lr * (decay ** (n - 1 - i))})
    groups.append({'params': list(model.head.parameters()), 'lr': base_lr})
    return groups


@torch.inference_mode()
def predict_texts(model, texts, tok, dev, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors='pt')
        b = {k: v.to(dev) for k, v in b.items()}
        out.append(decode_corrected(model(**b)).float().cpu())
    return torch.cat(out).numpy()


@torch.inference_mode()
def predict_probs(model, texts, tok, dev, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors='pt')
        b = {k: v.to(dev) for k, v in b.items()}
        out.append(torch.softmax(model(**b), dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--j', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=3200)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--layer-decay', type=float, default=0.75)
    ap.add_argument('--readout-steps', type=int, default=96)
    ap.add_argument('--readout-lr', type=float, default=5e-6)
    ap.add_argument('--readout-wd', type=float, default=0.01)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    set_seed(1000 * a.seed + a.fold)

    split = json.loads((A / 'splits' / f'B_s{a.seed}_f{a.fold}' / 'split.json').read_text())
    hold = set(split['outer_holdout']) | set(split['query_ids']) | set(split['groups'][a.j]['hold_ids'])
    lb = read_gold(DATA / f'lb500_s{a.seed}/train_lb.parquet', expected_n=500)
    tr = lb[~lb['id'].isin(hold)].reset_index(drop=True)
    ho = lb[lb['id'].isin(split['groups'][a.j]['hold_ids'])].reset_index(drop=True)
    q_ids = set(split['query_ids'])
    q = lb[lb['id'].isin(q_ids)].reset_index(drop=True) if q_ids else ho.iloc[0:0]
    print(f'teacher j={a.j} train={len(tr)} hold={len(ho)} query={len(q)}', flush=True)

    tok = AutoTokenizer.from_pretrained(str(UVR), local_files_only=True, use_fast=True)
    model = HistVAD().to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.layer_decay), weight_decay=a.weight_decay)
    dl = DataLoader(GoldData(tr), batch_size=a.batch_size, shuffle=True, collate_fn=TextCollator(tok, 128),
                    generator=torch.Generator().manual_seed(31_000 + 100 * a.seed + a.fold))
    it = iter(dl)
    for step in range(a.steps):
        model.train()
        try:
            b, y, _ = next(it)
        except StopIteration:
            it = iter(dl)
            b, y, _ = next(it)
        b = {k: v.to(dev) for k, v in b.items()}
        y = y.to(dev)
        loss = hist_loss(model(**b), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 800 == 0:
            print(json.dumps({'step': step + 1, 'ce': float(loss.detach())}), flush=True)

    model.train()
    opt2 = torch.optim.AdamW(model.parameters(), lr=a.readout_lr, weight_decay=a.readout_wd)
    ro_dl = DataLoader(GoldData(tr), batch_size=a.batch_size, shuffle=True, collate_fn=TextCollator(tok, 128),
                       generator=torch.Generator().manual_seed(17_000 + 100 * a.seed + a.fold))
    ro_it = iter(ro_dl)
    for _ in range(a.readout_steps):
        try:
            rb, ry, _ = next(ro_it)
        except StopIteration:
            ro_it = iter(ro_dl)
            rb, ry, _ = next(ro_it)
        rb = {k: v.to(dev) for k, v in rb.items()}
        ry = ry.to(dev)
        loss = hist_loss(model(**rb), ry)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()

    model.eval()
    ho_pred = predict_texts(model, ho['text'].tolist(), tok, dev)
    base_h = ho_pred  # base and final predictions coincide after the readout
    q_pred = predict_texts(model, q['text'].tolist(), tok, dev) if len(q) else np.zeros((0, 3), np.float32)
    u_ids = np.load(A / 'u_ids.npy', allow_pickle=True)
    u = pd.read_parquet(DATA / 'lb500_s0/train_ulb.parquet').set_index('id')
    u_texts = [u.at[i, 'text'] for i in u_ids]
    u_pred = predict_texts(model, u_texts, tok, dev)
    u_probs = predict_probs(model, u_texts, tok, dev)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, hold_ids=np.array(ho['id'].tolist(), dtype=object),
             hold_y=ho[['V_norm', 'A_norm', 'D_norm']].to_numpy(np.float32),
             hold_pred=ho_pred.astype(np.float32), base_hold_pred=base_h.astype(np.float32),
             query_ids=np.array(q['id'].tolist(), dtype=object),
             query_y=q[['V_norm', 'A_norm', 'D_norm']].to_numpy(np.float32) if len(q) else np.zeros((0, 3), np.float32),
             query_pred=q_pred.astype(np.float32),
             u_ids=u_ids, u_pred=u_pred.astype(np.float32), u_probs=u_probs.astype(np.float32))
    resid = ho[['V_norm', 'A_norm', 'D_norm']].to_numpy(np.float32) - ho_pred
    print('DONE', a.out, 'hold_mae', float(np.abs(resid).mean() * 4),
          'pred_sd', ho_pred.std(0).round(4).tolist(), flush=True)


if __name__ == '__main__':
    main()

