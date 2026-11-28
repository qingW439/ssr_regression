#!/usr/bin/env python3
"""HPL (Heteroscedastic Pseudo-Labels) ported to the EmoBank 3D VAD protocol.

Faithful to sxq11/Heteroscedastic-Pseudo-Labels (sts/main_ours.py):

  * shared encoder + prediction head, 3 outputs (V/A/D);
  * weak view = clean text, strong view = random token masking;
  * unsupervised term: exp(-u)/2 * (pred_strong - pred_weak)^2,
    u = UncertaintyLearner([err, pred]) with the input detached;
  * bilevel update every 5 iterations: one inner Adam step on the prediction
    head (higher.innerloop_ctx, unrolled and differentiable), followed by an
    uncertainty-learner step on held-out labeled samples:
        loss_unc = MSE(fc_adapted(feat_unc), y_unc) - lambda2 * mean(u)

Protocol parity with U1: same splits, same 500/400 labeled rows, same 6657
unlabeled texts, batch 16, 3200 outer steps, same 96-step supervised readout.
The only differences are the HPL objective and the strong-view forward pass.
No unlabeled label is ever used.
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import higher
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
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import set_seed  # noqa: E402

DATA = ROOT / 'data/emobank_processed'
UVR = ROOT / 'pretrained/roberta-base-simcse'
AV = ROOT / 'Semi-Supervised-Regression-avrrc/outputs'
A = ROOT / 'm1_v2_assets'
DIMS = ('V', 'A', 'D')
MASK_ID = 50264  # roberta-base <mask>


class UTexts(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return i, self.texts[i]


class UncertaintyLearner(nn.Module):
    """Official architecture: FC(in,128) + ReLU + FC(128,out), Xavier init."""

    def __init__(self, input_dim=6, output_dim=3, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, output_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        return self.fc(x)


class HPLModel(nn.Module):
    def __init__(self, model_name, dropout=0.1, offline=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=offline)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))

    def encode(self, **inputs):
        states = self.encoder(**inputs, return_dict=True).last_hidden_state
        m = inputs['attention_mask'].unsqueeze(-1).to(dtype=states.dtype)
        pooled = (states * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        return self.dropout(pooled)

    def forward(self, **inputs):
        return self.head(self.encode(**inputs))


def strong_view(batch, gen, p=0.15):
    """Random token masking as the strong augmentation (weak view is clean)."""
    ids = batch['input_ids'].clone()
    am = batch['attention_mask']
    special = ids <= 2
    cand = (am.bool() & ~special)
    r = torch.rand(ids.shape, generator=gen).to(ids.device)
    ids[cand & (r < p)] = MASK_ID
    return {'input_ids': ids, 'attention_mask': am}


def unc_input(err, pred):
    return torch.cat([err.detach(), pred.detach()], dim=-1)


def bilevel_update(head, unc, opt_unc, opt_head, feat_lb, y_lb, feat_w, feat_s,
                   feat_unc, y_unc, w_ulb, lambda2):
    """One official bilevel step. Returns diagnostics.

    opt_head is the same optimizer object used for the outer head updates,
    matching the official single-optimizer `optim_fc` (higher patches it in place).
    """
    with higher.innerloop_ctx(head, opt_head, copy_initial_weights=False) as (fhead, diffopt):
        preds_lb = fhead(feat_lb)
        loss_lb = F.mse_loss(preds_lb, y_lb)
        with torch.no_grad():
            preds_w = fhead(feat_w)
        preds_s = fhead(feat_s)
        u = unc(unc_input(preds_s - preds_w, preds_s))
        loss_ulb = (torch.exp(-u) / 2.0 * (preds_s - preds_w) ** 2).mean()
        diffopt.step(loss_lb + w_ulb * loss_ulb)

        preds_unc = fhead(feat_unc.detach())
        u2 = unc(unc_input(preds_unc - y_unc, preds_unc))
        loss_unc = F.mse_loss(preds_unc, y_unc) - lambda2 * u2.mean()
        opt_unc.zero_grad(set_to_none=True)
        loss_unc.backward()
        opt_unc.step()
    return {'inner_loss': float((loss_lb + w_ulb * loss_ulb).detach()),
            'unc_loss': float(loss_unc.detach()), 'u_mean': float(u.mean().detach())}


def param_groups(model, base_lr, decay, head_lr):
    groups = [{'params': list(model.encoder.embeddings.parameters()), 'lr': base_lr * (decay ** 12)}]
    for i, layer in enumerate(model.encoder.encoder.layer):
        groups.append({'params': list(layer.parameters()), 'lr': base_lr * (decay ** (11 - i))})
    groups.append({'params': list(model.head.parameters()), 'lr': head_lr})
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
        out.append(model(**b).float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=3200)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--fc-lr', type=float, default=1e-4)
    ap.add_argument('--unc-lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--layer-decay', type=float, default=0.75)
    ap.add_argument('--w-ulb', type=float, default=1.0)
    ap.add_argument('--lambda2', type=float, default=1.0)
    ap.add_argument('--mask-p', type=float, default=0.15)
    ap.add_argument('--inner-every', type=int, default=5)
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
    model = HPLModel(str(UVR), dropout=0.1, offline=True).to(dev)
    unc = UncertaintyLearner(6, 3).to(dev)
    print(f'params total={sum(p.numel() for p in model.parameters())} '
          f'unc={sum(p.numel() for p in unc.parameters())}', flush=True)
    opt = torch.optim.AdamW(param_groups(model, a.lr, a.layer_decay, a.fc_lr),
                            weight_decay=a.weight_decay)
    # official split: feature optimizer + prediction-head optimizer
    head_ids = {id(p) for p in model.head.parameters()}
    feat_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt_feat = torch.optim.AdamW(feat_params, lr=a.lr, weight_decay=a.weight_decay)
    opt_head = torch.optim.Adam(model.head.parameters(), lr=a.fc_lr, weight_decay=1e-5)
    opt_unc = torch.optim.Adam(unc.parameters(), lr=a.unc_lr, weight_decay=1e-5)

    lb_dl = DataLoader(GoldData(tr), batch_size=a.batch_size, shuffle=True, collate_fn=TextCollator(tok, 128),
                       generator=torch.Generator().manual_seed(31_000 + 100 * a.seed + a.fold))
    lb_it = iter(lb_dl)
    unc_dl = DataLoader(GoldData(tr), batch_size=a.batch_size, shuffle=True, collate_fn=TextCollator(tok, 128),
                        generator=torch.Generator().manual_seed(41_000 + 100 * a.seed + a.fold))
    unc_it = iter(unc_dl)

    def collate(batch):
        idx = [i for i, _ in batch]
        t = tok([s for _, s in batch], padding=True, truncation=True, max_length=128, return_tensors='pt')
        return torch.tensor(idx), t

    u_dl = DataLoader(UTexts(u_texts), batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                      generator=torch.Generator().manual_seed(19_000 + 100 * a.seed + a.fold))
    u_it = iter(u_dl)
    aug_gen = torch.Generator().manual_seed(23_000 + 100 * a.seed + a.fold)
    n_strong_masked = 0
    logs = []
    for step in range(a.steps):
        model.train()
        unc.train()
        try:
            b, y, _ = next(lb_it)
        except StopIteration:
            lb_it = iter(lb_dl)
            b, y, _ = next(lb_it)
        b = {k: v.to(dev) for k, v in b.items()}
        y = y.to(dev)
        try:
            idx, ub = next(u_it)
        except StopIteration:
            u_it = iter(u_dl)
            idx, ub = next(u_it)
        ub = {k: v.to(dev) for k, v in ub.items()}
        sb = strong_view(ub, aug_gen, a.mask_p)
        sb = {k: v.to(dev) for k, v in sb.items()}
        n_strong_masked += int((sb['input_ids'] != ub['input_ids']).sum())

        feat_lb = model.encode(**b)
        with torch.no_grad():
            feat_w = model.encode(**ub)
        feat_s = model.encode(**sb)

        diag = None
        if step % a.inner_every == 0:
            try:
                ub2, y2, _ = next(unc_it)
            except StopIteration:
                unc_it = iter(unc_dl)
                ub2, y2, _ = next(unc_it)
            ub2 = {k: v.to(dev) for k, v in ub2.items()}
            y2 = y2.to(dev)
            with torch.no_grad():
                feat_unc = model.encode(**ub2)
            diag = bilevel_update(model.head, unc, opt_unc, opt_head, feat_lb.detach(), y,
                                  feat_w.detach(), feat_s.detach(), feat_unc, y2,
                                  a.w_ulb, a.lambda2)

        preds_lb = model.head(feat_lb)
        loss_lb = F.mse_loss(preds_lb, y)
        preds_s = model.head(feat_s)
        with torch.no_grad():
            preds_w = model.head(feat_w)
            u = unc(unc_input(preds_s - preds_w, preds_s))
        loss_ulb = (torch.exp(-u) / 2.0 * (preds_s - preds_w) ** 2).mean()
        loss = loss_lb + a.w_ulb * loss_ulb
        opt_feat.zero_grad(set_to_none=True)
        opt_head.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_head.step()
        opt_feat.step()

        if (step + 1) % 400 == 0:
            rec = {'step': step + 1, 'lb': float(loss_lb.detach()), 'ulb': float(loss_ulb.detach()),
                   'u_mean': float(u.mean().detach()), 'u_sd': float(u.std().detach())}
            if diag:
                rec.update({'inner': diag['inner_loss'], 'unc': diag['unc_loss']})
            logs.append(rec)
            print(json.dumps(rec), flush=True)

    model.eval()
    base_ho = predict_texts(model, ho['text'].tolist(), tok, dev)
    torch.save({'model': model.state_dict(), 'unc': unc.state_dict(),
                'seed': a.seed, 'fold': a.fold}, out / 'base.pth')

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
        loss = F.l1_loss(ro(**rb), ry)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ro.parameters(), 1.0)
        opt2.step()
    ro.eval()
    ro_ho = predict_texts(ro, ho['text'].tolist(), tok, dev)
    torch.save({'model': ro.state_dict(), 'seed': a.seed, 'fold': a.fold}, out / 'readout.pth')

    Y = ho[['V_norm', 'A_norm', 'D_norm']].to_numpy(np.float32)
    rep = {'implementation': 'HPL official mechanism ported to EmoBank 3D (heteroscedastic weights + bilevel)',
           'seed': a.seed, 'fold': a.fold,
           'base_metrics': metrics(Y, base_ho), 'metrics': metrics(Y, ro_ho),
           'strong_tokens_masked': n_strong_masked,
           'train_log': logs,
           'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    (out / 'result.json').write_text(json.dumps(rep, indent=2) + '\n')
    df = pd.DataFrame({'id': ho['id'], 'fold': a.fold})
    for i, d in enumerate(DIMS):
        df[f'y_{d}'] = Y[:, i]
        df[f'base_{d}'] = base_ho[:, i]
        df[f'raw_{d}'] = ro_ho[:, i]
    df.to_parquet(out / 'oof_predictions.parquet', index=False)
    print(f'DONE HPL seed={a.seed} fold={a.fold} base={rep["base_metrics"]["mae"]:.6f} '
          f'readout={rep["metrics"]["mae"]:.6f}', flush=True)


if __name__ == '__main__':
    main()
