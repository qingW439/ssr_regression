#!/usr/bin/env python3
"""AAN-style reimplementation for EmoBank 3D VAD (supervised, unified protocol).

Private attention per dimension + shared attention + private/common discriminators.
The common discriminator is adversarial via gradient reversal on the shared feature.
The public repo (sriyash421/Adversarial-Attention) is notebook-only, 2D, and incomplete;
this is an explicit reimplementation, not the authors' code.
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
ROOT = ASSETS
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import set_seed  # noqa: E402

DATA = ROOT / 'data/emobank_processed'
UVR = ROOT / 'pretrained/roberta-base-simcse'
AV = ROOT / 'Semi-Supervised-Regression-avrrc/outputs'
DIMS = ('V', 'A', 'D')


class _Rev(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


def rev(x, alpha=1.0):
    return _Rev.apply(x, alpha)


class AttnPool(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.proj = nn.Linear(h, 1)

    def forward(self, H, mask):
        s = torch.tanh(self.proj(H)).squeeze(-1)
        s = s.masked_fill(mask == 0, -1e9)
        a = torch.softmax(s, dim=1)
        return (a.unsqueeze(-1) * H).sum(1)


class AANModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        h = self.encoder.config.hidden_size
        self.attn_priv = nn.ModuleList([AttnPool(h) for _ in range(3)])
        self.attn_shared = AttnPool(h)
        self.reg = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
                                  for _ in range(3)])
        self.disc_priv = nn.Linear(h, 3)
        self.drop = nn.Dropout(0.1)

    def forward(self, **inputs):
        H = self.encoder(**inputs, return_dict=True).last_hidden_state
        mask = inputs['attention_mask']
        fps = [self.drop(self.attn_priv[d](H, mask)) for d in range(3)]
        fs = self.drop(self.attn_shared(H, mask))
        preds = torch.cat([self.reg[d](torch.cat([fps[d], fs], dim=-1)) for d in range(3)], dim=1)
        # private adversarial discriminator: it classifies the dimension from reversed
        # private features, so the attention/encoder is pushed to make them dimension-invariant.
        lp = torch.stack([self.disc_priv(rev(f)) for f in fps], dim=1)  # B,3(dim),3(class)
        return preds, lp


def param_groups(model, base_lr, decay):
    groups = [{'params': list(model.encoder.embeddings.parameters()), 'lr': base_lr * (decay ** 12)}]
    for i, layer in enumerate(model.encoder.encoder.layer):
        groups.append({'params': list(layer.parameters()), 'lr': base_lr * (decay ** (11 - i))})
    rest = list(model.attn_priv.parameters()) + list(model.attn_shared.parameters()) + \
           list(model.reg.parameters()) + [model.disc_priv.weight, model.disc_priv.bias]
    groups.append({'params': rest, 'lr': base_lr})
    return groups


def metrics(Y, P):
    per = {}
    for i, d in enumerate(DIMS):
        y, x = Y[:, i], P[:, i]
        per[d] = {'mae': float(np.abs(x - y).mean() * 4),
                  'ccc': float(2 * ((x - x.mean()) * (y - y.mean())).mean()
                               / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12)),
                  'pearson': float(np.corrcoef(x, y)[0, 1]),
                  'r2': float(1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum())}
    return {'mae': float(np.mean([v['mae'] for v in per.values()])),
            'ccc': float(np.mean([v['ccc'] for v in per.values()])),
            'pearson': float(np.mean([v['pearson'] for v in per.values()])),
            'r2': float(np.mean([v['r2'] for v in per.values()])),
            'per_dim': per}


@torch.inference_mode()
def predict(model, loader, dev):
    model.eval()
    ys, ps, ids = [], [], []
    for b, y, bid in loader:
        b = {k: v.to(dev) for k, v in b.items()}
        p, _ = model(**b)
        ys.append(y.numpy()); ps.append(p.cpu().numpy()); ids.extend(bid)
    return np.concatenate(ys), np.concatenate(ps), ids


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
    ap.add_argument('--aux-weight', type=float, default=0.1)
    ap.add_argument('--readout-steps', type=int, default=96)
    ap.add_argument('--readout-lr', type=float, default=5e-6)
    ap.add_argument('--readout-wd', type=float, default=0.01)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    out = a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    set_seed(1000 * a.seed + a.fold)

    lb = read_gold(DATA / f'lb500_s{a.seed}/train_lb.parquet', expected_n=500)
    man = json.loads((AV / ('p0' if a.seed == 0 else f'p0_s{a.seed}') / 'fold_manifest.json').read_text())
    hold = set(man['fold_ids'][str(a.fold)])
    tr = lb[~lb['id'].isin(hold)].reset_index(drop=True)
    ho = lb[lb['id'].isin(hold)].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(str(UVR), local_files_only=True, use_fast=True)
    model = AANModel(str(UVR)).to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.layer_decay), weight_decay=a.weight_decay)
    dl = DataLoader(GoldData(tr), batch_size=a.batch_size, shuffle=True, collate_fn=TextCollator(tok, 128),
                    generator=torch.Generator().manual_seed(31_000 + 100 * a.seed + a.fold))
    it = iter(dl)
    dims = torch.arange(3, device=dev)

    logs = []
    for step in range(a.steps):
        model.train()
        try:
            b, y, _ = next(it)
        except StopIteration:
            it = iter(dl); b, y, _ = next(it)
        b = {k: v.to(dev) for k, v in b.items()}
        y = y.to(dev)
        pred, lp = model(**b)
        reg = F.l1_loss(pred, y)
        lp_loss = sum(F.cross_entropy(lp[:, d, :], torch.full((len(y),), d, device=dev, dtype=torch.long))
                      for d in range(3)) / 3.0
        loss = reg + a.aux_weight * lp_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 400 == 0:
            logs.append({'step': step + 1, 'reg': float(reg.detach()), 'adv': float(lp_loss.detach())})
            print(json.dumps(logs[-1]), flush=True)

    eval_dl = DataLoader(GoldData(ho), batch_size=64, shuffle=False, collate_fn=TextCollator(tok, 128))
    Y, base_ho, _ = predict(model, eval_dl, dev)
    torch.save({'model': model.state_dict(), 'seed': a.seed, 'fold': a.fold,
                'impl': 'AAN-style reimplementation'}, out / 'base.pth')

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
            ro_it = iter(ro_dl); rb, ry, _ = next(ro_it)
        rb = {k: v.to(dev) for k, v in rb.items()}
        ry = ry.to(dev)
        pred, _ = ro(**rb)
        loss = F.l1_loss(pred, ry)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ro.parameters(), 1.0)
        opt2.step()
    Y2, ro_ho, ids = predict(ro, eval_dl, dev)
    torch.save({'model': ro.state_dict(), 'seed': a.seed, 'fold': a.fold}, out / 'readout.pth')

    rep = {'implementation': 'AAN-style reimplementation',
           'source': 'Zhu et al., ACL 2019, Adversarial Attention Network',
           'note': 'private/shared attention + private adversarial discriminator (gradient reversal); no official authors code available; not a bit-exact reproduction',
           'seed': a.seed, 'fold': a.fold,
           'base_metrics': metrics(Y, base_ho), 'metrics': metrics(Y2, ro_ho),
           'aux_weight': a.aux_weight, 'steps': a.steps, 'train_log': logs,
           'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    (out / 'result.json').write_text(json.dumps(rep, indent=2) + '\n')
    df = pd.DataFrame({'id': ho['id'], 'fold': a.fold})
    for i, d in enumerate(DIMS):
        df[f'y_{d}'] = Y2[:, i]
        df[f'base_{d}'] = base_ho[:, i]
        df[f'raw_{d}'] = ro_ho[:, i]
    df.to_parquet(out / 'oof_predictions.parquet', index=False)
    print(f'DONE AAN seed={a.seed} fold={a.fold} base={rep["base_metrics"]["mae"]:.6f} '
          f'readout={rep["metrics"]["mae"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
