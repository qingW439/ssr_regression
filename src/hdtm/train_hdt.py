#!/usr/bin/env python3
"""HDT students: distribution-teacher information transfer on the HS/HU recipe.

HDT-M: unlabeled target = Gaussian distribution around the decoded fused value
       (i.e., the old numeric pathway but with the new distribution teachers).
HDT-D: unlabeled target = the fused 3x20 probability distribution itself (CE).

Everything else (encoder, head, labelled CE, 3200 steps, 96-step readout, 0.25
unlabeled weight) is identical to H-S/H-U.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path(os.environ.get('SSR_ASSETS', '/root/autodl-tmp/rankup_project'))
ROOT = ASSETS
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src/hdtm'))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor, set_seed  # noqa: E402
from hist_loss import K, HistHead, decode_corrected, gaussian_target, hist_loss  # noqa: E402

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


def metrics(Y, P_):
    from scipy.stats import spearmanr
    per = {}
    for i, d in enumerate(DIMS):
        y, x = Y[:, i], P_[:, i]
        q = np.quantile(y, [0.2, 0.4, 0.6, 0.8])
        mid = (y >= q[1]) & (y <= q[2])
        tail = (y < q[0]) | (y > q[3])
        per[d] = {'mae': float(np.abs(x - y).mean() * 4),
                  'pearson': float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 else 0.0,
                  'ccc': float(2 * ((x - x.mean()) * (y - y.mean())).mean()
                               / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12)),
                  'r2': float(1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                  'srcc': float(spearmanr(x, y).statistic),
                  'sd_ratio': float(x.std() / (y.std() + 1e-12)),
                  'bias': float((x - y).mean() * 4),
                  'mid_mae': float(np.abs(x[mid] - y[mid]).mean() * 4),
                  'tail_mae': float(np.abs(x[tail] - y[tail]).mean() * 4)}
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
        out.append(decode_corrected(model(**b)).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=('hdtm', 'hdtd', 'acl'), required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--fused', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--lam', type=float, default=0.25)
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
    z = np.load(a.fused, allow_pickle=True)
    assert list(z['u_ids']) == list(u_ids), 'U id order mismatch'
    q_prob = torch.tensor(z['u_probs'], device=dev)          # (N,3,K)
    q_val = torch.tensor(z['u_value'], device=dev)           # (N,3)

    tok = AutoTokenizer.from_pretrained(str(UVR), local_files_only=True, use_fast=True)
    model = HistVAD().to(dev)
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
        ce_s = hist_loss(model(**b), y)
        try:
            idx, ub = next(u_it)
        except StopIteration:
            u_it = iter(u_dl)
            idx, ub = next(u_it)
        ub = {k: v.to(dev) for k, v in ub.items()}
        logits_u = model(**ub)
        if a.arm == 'hdtm':
            t_u = gaussian_target(q_val[idx.to(dev)])
            l_u = -(t_u * F.log_softmax(logits_u, dim=-1)).sum(-1).mean()
        elif a.arm == 'acl':
            # A-CL: labelled CE + unlabeled L1 on the decoded continuous values
            l_u = F.l1_loss(decode_corrected(logits_u), q_val[idx.to(dev)])
        else:
            l_u = -(q_prob[idx.to(dev)] * F.log_softmax(logits_u, dim=-1)).sum(-1).mean()
        loss = ce_s + a.lam * l_u
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        if (step + 1) % 400 == 0:
            logs.append({'step': step + 1, 'ce_s': float(ce_s.detach()), 'ce_u': float(l_u.detach()),
                         'gnorm': gnorm})
            print(json.dumps(logs[-1]), flush=True)

    model.eval()
    base_ho = predict_texts(model, ho['text'].tolist(), tok, dev)
    torch.save({'model': model.state_dict(), 'arm': a.arm, 'seed': a.seed, 'fold': a.fold}, out / 'base.pth')

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
        loss = hist_loss(ro(**rb), ry)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ro.parameters(), 1.0)
        opt2.step()
    ro.eval()
    ro_ho = predict_texts(ro, ho['text'].tolist(), tok, dev)
    torch.save({'model': ro.state_dict(), 'arm': a.arm, 'seed': a.seed, 'fold': a.fold}, out / 'readout.pth')

    Y = ho[['V_norm', 'A_norm', 'D_norm']].to_numpy(np.float32)
    rep = {'arm': a.arm, 'seed': a.seed, 'fold': a.fold, 'lam': a.lam,
           'base_metrics': metrics(Y, base_ho), 'metrics': metrics(Y, ro_ho),
           'train_log': logs, 'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    (out / 'result.json').write_text(json.dumps(rep, indent=2) + '\n')
    df = pd.DataFrame({'id': ho['id'], 'fold': a.fold})
    for i, d in enumerate(DIMS):
        df[f'y_{d}'] = Y[:, i]
        df[f'base_{d}'] = base_ho[:, i]
        df[f'raw_{d}'] = ro_ho[:, i]
    df.to_parquet(out / 'oof_predictions.parquet', index=False)
    print(f'DONE {a.arm} seed={a.seed} fold={a.fold} base={rep["base_metrics"]["mae"]:.6f} '
          f'readout={rep["metrics"]["mae"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
