#!/usr/bin/env python3
"""RCUS-3D (official mechanism) on the U1 recipe.

U1 recipe unchanged: same splits, same optimizer/param groups, 3200 steps,
batch 16, same 96-step supervised readout and metrics. What is added is exactly
the SSR-RCUS ranking head and the Eq.(5)/(8) losses from rcus_loss.py:

    total = L1(pred, y) + w_u * Lu + w_r * Lr

No pseudo-label regression targets, no teacher, no UUL: this isolates the
official ranking mechanism. Diagnostic printouts include the RCUS confidence
mask fraction and per-term losses so a silent no-op is visible.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
ROOT = ASSETS
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src/baselines'))
sys.path.insert(0, str(REPO / 'src/hdtm'))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import set_seed  # noqa: E402
from rcus_loss import (labeled_ranking_loss, mixup_features, mixup_ranking_loss,  # noqa: E402
                       pseudo_rank)

DATA = ROOT / 'data/emobank_processed'
UVR = ROOT / 'pretrained/roberta-base-simcse'
AV = ROOT / 'Semi-Supervised-Regression-avrrc/outputs'
A = ROOT / 'm1_v2_assets'
DIMS = ('V', 'A', 'D')


class UTexts(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return i, self.texts[i]


class RobertaVADARC(nn.Module):
    """U1 encoder/head plus the official scalar ranking predictor H (per dim)."""

    def __init__(self, model_name, dropout=0.1, offline=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=offline)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))
        self.rank = nn.ModuleList([nn.Linear(h, 1) for _ in range(3)])

    def encode(self, **inputs):
        states = self.encoder(**inputs, return_dict=True).last_hidden_state
        m = inputs['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
        pooled = (states * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        return self.dropout(pooled)

    def scores(self, feat):
        return torch.cat([head(feat) for head in self.rank], dim=-1)   # (..., 3)

    def forward(self, **inputs):
        feat = self.encode(**inputs)
        return self.head(feat), feat


def param_groups(model, base_lr, decay):
    groups = [{'params': list(model.encoder.embeddings.parameters()), 'lr': base_lr * (decay ** 12)}]
    for i, layer in enumerate(model.encoder.encoder.layer):
        groups.append({'params': list(layer.parameters()), 'lr': base_lr * (decay ** (11 - i))})
    groups.append({'params': list(model.head.parameters()) + list(model.rank.parameters()), 'lr': base_lr})
    return groups


def metrics(Y, P):
    from scipy.stats import spearmanr
    per = {}
    for i, d in enumerate(DIMS):
        y, x = Y[:, i], P[:, i]
        q = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
        mid = (y >= q[1]) & (y <= q[2])
        tail = (y < q[0]) | (y > q[3])
        per[d] = {
            'mae': float(np.abs(x - y).mean() * 4),
            'pearson': float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 else 0.0,
            'ccc': float(2 * ((x - x.mean()) * (y - y.mean())).mean()
                         / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12)),
            'r2': float(1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            'srcc': float(spearmanr(x, y).statistic),
            'sd_ratio': float(x.std() / (y.std() + 1e-12)),
            'bias': float((x - y).mean() * 4),
            'mid_mae': float(np.abs(x[mid] - y[mid]).mean() * 4),
            'tail_mae': float(np.abs(x[tail] - y[tail]).mean() * 4),
        }
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'pearson': float(np.mean([v['pearson'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'r2': float(np.mean([v['r2'] for v in per.values()])),
            'srcc': float(np.mean([v['srcc'] for v in per.values()])),
            'sd_ratio': float(np.mean([v['sd_ratio'] for v in per.values()])),
            'bias': float(np.mean([v['bias'] for v in per.values()])),
            'mid_mae': float(np.mean([v['mid_mae'] for v in per.values()])),
            'tail_mae': float(np.mean([v['tail_mae'] for v in per.values()])),
            'per_dim': per}


@torch.inference_mode()
def predict_texts(model, texts, tok, dev, bs=64):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=128, return_tensors='pt')
        b = {k: v.to(dev) for k, v in b.items()}
        out.append(model(**b)[0].float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=3200)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--layer-decay', type=float, default=0.75)
    ap.add_argument('--w-u', type=float, default=0.1)
    ap.add_argument('--w-r', type=float, default=0.1)
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--tau', type=float, default=0.95)
    ap.add_argument('--readout-steps', type=int, default=96)
    ap.add_argument('--readout-lr', type=float, default=5e-6)
    ap.add_argument('--readout-wd', type=float, default=0.01)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    out = a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    set_seed(1000 * a.seed + a.fold)

    lb = read_gold(DATA / f'lb500_s{a.seed}/train_lb.parquet', expected_n=500)
    man = json.loads((AV / ('p0' if a.seed == 0 else f'p0_s{a.seed}') / 'fold_manifest.json').read_text())
    hold = set(man['fold_ids'][str(a.fold)])
    tr = lb[~lb['id'].isin(hold)].reset_index(drop=True)
    ho = lb[lb['id'].isin(hold)].reset_index(drop=True)
    u_ids = np.load(A / 'u_ids.npy', allow_pickle=True)
    u_all = pd.read_parquet(DATA / 'lb500_s0/train_ulb.parquet').set_index('id')
    u_texts = [u_all.at[i, 'text'] for i in u_ids]

    tok = AutoTokenizer.from_pretrained(str(UVR), local_files_only=True, use_fast=True)
    model = RobertaVADARC(str(UVR), dropout=0.1, offline=True).to(dev)
    n_rank = sum(p.numel() for p in model.rank.parameters())
    print(f'params total={sum(p.numel() for p in model.parameters())} rank_head={n_rank}', flush=True)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.layer_decay), weight_decay=a.weight_decay)
    lb_dl = DataLoader(GoldData(tr), batch_size=a.batch_size, shuffle=True, collate_fn=TextCollator(tok, 128),
                       generator=torch.Generator().manual_seed(31_000 + 100 * a.seed + a.fold))
    lb_it = iter(lb_dl)

    def collate(batch):
        idx = [i for i, _ in batch]
        t = tok([s for _, s in batch], padding=True, truncation=True, max_length=128, return_tensors='pt')
        return torch.tensor(idx), t

    u_dl = DataLoader(UTexts(u_texts), batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                      generator=torch.Generator().manual_seed(19_000 + 100 * a.seed + a.fold))
    u_it = iter(u_dl)
    logs = []
    for step in range(a.steps):
        model.train()
        try:
            b, y, _ = next(lb_it)
        except StopIteration:
            lb_it = iter(lb_dl)
            b, y, _ = next(lb_it)
        b = {k: v.to(dev) for k, v in b.items()}
        y = y.to(dev)
        pred_l, feat_l = model(**b)
        loss_l1 = F.l1_loss(pred_l, y)
        s_l = model.scores(feat_l)

        # Eq.(8) labeled ranking loss
        lr = labeled_ranking_loss(s_l, y)

        # Eq.(5) unlabeled ranking loss with manifold mixup
        try:
            idx, ub = next(u_it)
        except StopIteration:
            u_it = iter(u_dl)
            idx, ub = next(u_it)
        ub = {k: v.to(dev) for k, v in ub.items()}
        _, feat_u = model(**ub)
        s_u = model.scores(feat_u)
        lam = float(np.random.beta(a.alpha, a.alpha))
        lam = max(lam, 1.0 - lam)
        za, zb = mixup_features(feat_u, lam)
        t, m = pseudo_rank(s_u, tau=a.tau)
        if float(m.sum()) > 0:
            lu = mixup_ranking_loss(s_u, model.scores(za), model.scores(zb), t, m)
        else:
            lu = torch.zeros((), device=dev)
        mask_frac = float(m.mean())

        loss = loss_l1 + a.w_u * lu + a.w_r * lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 400 == 0:
            logs.append({'step': step + 1, 'l1': float(loss_l1.detach()), 'lu': float(lu.detach()),
                         'lr': float(lr.detach()), 'mask_frac': mask_frac, 'lam': lam})
            print(json.dumps(logs[-1]), flush=True)

    model.eval()
    base_ho = predict_texts(model, ho['text'].tolist(), tok, dev)
    torch.save({'model': model.state_dict(), 'seed': a.seed, 'fold': a.fold}, out / 'base.pth')

    ro = copy.deepcopy(model).to(dev)
    ro.train()
    opt2 = torch.optim.AdamW(ro.parameters(), lr=a.readout_lr, weight_decay=a.readout_wd)
    ro_dl = DataLoader(GoldData(tr), batch_size=a.batch_size, shuffle=True, collate_fn=TextCollator(tok, 128),
                       generator=torch.Generator().manual_seed(17_000 + a.fold))
    ro_it = iter(ro_dl)
    for _ in range(a.readout_steps):
        try:
            rb, ry, _ = next(ro_it)
        except StopIteration:
            ro_it = iter(ro_dl)
            rb, ry, _ = next(ro_it)
        rb = {k: v.to(dev) for k, v in rb.items()}
        ry = ry.to(dev)
        loss = F.l1_loss(ro(**rb)[0], ry)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ro.parameters(), 1.0)
        opt2.step()
    ro.eval()
    ro_ho = predict_texts(ro, ho['text'].tolist(), tok, dev)
    torch.save({'model': ro.state_dict(), 'seed': a.seed, 'fold': a.fold}, out / 'readout.pth')

    Y = ho[['V_norm', 'A_norm', 'D_norm']].to_numpy(np.float32)
    rep = {'implementation': 'SSR-RCUS-3D official mechanism (Eq.3-5,8), scalar rank head per dim',
           'seed': a.seed, 'fold': a.fold,
           'base_metrics': metrics(Y, base_ho), 'metrics': metrics(Y, ro_ho),
           'rank_head_params': n_rank, 'w_u': a.w_u, 'w_r': a.w_r, 'alpha': a.alpha, 'tau': a.tau,
           'train_log': logs,
           'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    (out / 'result.json').write_text(json.dumps(rep, indent=2) + '\n')
    df = pd.DataFrame({'id': ho['id'], 'fold': a.fold})
    for i, d in enumerate(DIMS):
        df[f'y_{d}'] = Y[:, i]
        df[f'base_{d}'] = base_ho[:, i]
        df[f'raw_{d}'] = ro_ho[:, i]
    df.to_parquet(out / 'oof_predictions.parquet', index=False)
    print(f'DONE RCUS2 seed={a.seed} fold={a.fold} base={rep["base_metrics"]["mae"]:.6f} '
          f'readout={rep["metrics"]["mae"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
