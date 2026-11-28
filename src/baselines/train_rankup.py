#!/usr/bin/env python3
"""RankUp-3D official-recipe adaptation (one fold).

Arms: b0 raw pure, b1 raw ARC, b2 UVR pure, b3 UVR ARC, b4 UVR labeled-only ARC.
Official ingredients: lr 1e-5, wd 5e-4, layer decay 0.75, ARC ratios 0.2/1.0,
T=0.5, p_cutoff=0.95. Strong unlabeled view is token masking (text-level), because
the official precomputed augmented corpus is unavailable. 96-step readout remains
the frozen pure-supervised script for all arms.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from semilearn.datasets.nlp_datasets.emobank import GoldData, TextCollator, read_gold  # noqa: E402
from scripts.vad_research.train_roberta_vad_supervised import RobertaVADRegressor, set_seed  # noqa: E402
from scripts.vad_research.train_acvr import mask_tokens  # noqa: E402

ROOT = Path("/root/autodl-tmp/rankup_project")
DATA = ROOT / "data/emobank_processed"
RAW = ROOT / "pretrained/roberta-base"
UVR = ROOT / "pretrained/roberta-base-simcse"


class Texts(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]


def pool(states, mask):
    m = mask.unsqueeze(-1).to(dtype=states.dtype)
    return (states * m).sum(1) / m.sum(1).clamp_min(1)


def forward_arc(model, arc, batch, dropout):
    states = model.encoder(**batch, return_dict=True).last_hidden_state
    feat = pool(states, batch["attention_mask"])
    pred = model.head(dropout(feat))
    logits = arc(feat).view(-1, 3, 2)
    return pred, logits


def soft_rank_loss(logits, targets):
    diff = targets[:, None, :] - targets[None, :, :]
    p0 = 0.5 * (1.0 + torch.sign(diff))
    eye = torch.eye(len(targets), dtype=torch.bool, device=targets.device)[:, :, None]
    p0 = torch.where(eye, torch.zeros_like(p0), p0)
    valid = ~eye
    logp = F.log_softmax(logits, dim=-1)
    loss = -(p0 * logp[..., 0] + (1 - p0) * logp[..., 1])
    return loss[valid.expand_as(loss)].mean()


def unlabeled_rank_loss(weak, strong, temp=0.5, cutoff=0.95):
    wp = weak.unsqueeze(1) - weak.unsqueeze(0)
    sp = strong.unsqueeze(1) - strong.unsqueeze(0)
    with torch.no_grad():
        conf = wp.softmax(-1).max(-1).values
        pseudo = (wp / temp).softmax(-1).argmax(-1)
        eye = torch.eye(len(weak), dtype=torch.bool, device=weak.device)[:, :, None]
        mask = (conf >= cutoff) & ~eye.expand(-1, -1, 3)
    if mask.sum() == 0:
        return sp.sum() * 0.0, 0.0
    loss = F.cross_entropy(sp[mask], pseudo[mask])
    return loss, float(mask.float().mean())




def labeled_rrc(pred, targets, margin):
    diff = pred[:, None, :] - pred[None, :, :]
    sign = torch.sign(targets[:, None, :] - targets[None, :, :])
    eye = torch.eye(len(targets), dtype=torch.bool, device=targets.device)[:, :, None]
    hinge = torch.relu(margin - diff * sign)
    return hinge[~eye.expand_as(hinge)].mean()


def unlabeled_rrc(pred, weak_arc, margin):
    wp = weak_arc.unsqueeze(1) - weak_arc.unsqueeze(0)
    with torch.no_grad():
        conf = wp.softmax(-1).max(-1).values
        pseudo = (wp / 0.5).softmax(-1).argmax(-1)
        eye = torch.eye(len(pred), dtype=torch.bool, device=pred.device)[:, :, None]
        mask = (conf >= 0.95) & ~eye.expand(-1, -1, 3)
    if mask.sum() == 0:
        return pred.sum() * 0.0
    sign = torch.where(pseudo == 0, 1.0, -1.0)
    diff = pred[:, None, :] - pred[None, :, :]
    hinge = torch.relu(margin - diff * sign)
    return hinge[mask].mean()


def param_groups(model, base_lr, decay):
    groups = []
    emb = list(model.encoder.embeddings.parameters())
    groups.append({"params": emb, "lr": base_lr * (decay ** 12)})
    for i, layer in enumerate(model.encoder.encoder.layer):
        groups.append({"params": list(layer.parameters()), "lr": base_lr * (decay ** (11 - i))})
    groups.append({"params": list(model.head.parameters()), "lr": base_lr})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--data-seed", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--arm", choices=("b0", "b1", "b2", "b3", "b4"), required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--arc-weight", type=float, default=0.2)
    ap.add_argument("--arc-ulb-weight", type=float, default=1.0)
    ap.add_argument("--rrc-weight", type=float, default=0.0)
    ap.add_argument("--rrc-margin", type=float, default=0.05)
    ap.add_argument("--strong-view", choices=("dropout", "mask"), default="mask")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    use_arc = a.arm in ("b1", "b3", "b4")
    use_unlabeled = a.arm in ("b1", "b3")
    init_path = RAW if a.arm in ("b0", "b1") else UVR
    set_seed(1000 * a.data_seed + a.fold)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = a.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    lb = read_gold(DATA / f"lb500_s{a.data_seed}" / "train_lb.parquet", expected_n=500)
    man = json.loads(a.manifest.read_text()); hold = set(man["fold_ids"][str(a.fold)])
    tr = [i for i in lb["id"] if i not in hold]
    lb_tr = lb[lb["id"].isin(tr)].reset_index(drop=True)
    lb_ho = lb[lb["id"].isin(hold)].reset_index(drop=True)
    sets, texts = [], {}
    for s in (0, 1, 2):
        u = pd.read_parquet(DATA / f"lb500_s{s}" / "train_ulb.parquet")
        sets.append(set(u["id"])); texts.update(dict(zip(u["id"], u["text"])))
    corpus = [texts[i] for i in sorted(set.intersection(*sets))]
    tok = AutoTokenizer.from_pretrained(str(init_path), local_files_only=True, use_fast=True)
    model = RobertaVADRegressor(str(init_path), dropout=0.1, offline=True).to(dev)
    arc = torch.nn.Linear(768, 6).to(dev) if use_arc else None
    params = param_groups(model, a.lr, a.layer_decay)
    if arc is not None:
        params.append({"params": list(arc.parameters()), "lr": a.lr})
    opt = torch.optim.AdamW(params, weight_decay=a.weight_decay)
    lb_dl = DataLoader(GoldData(lb_tr), batch_size=a.batch_size, shuffle=True,
                       collate_fn=TextCollator(tok, 128),
                       generator=torch.Generator().manual_seed(31_000 + 100 * a.data_seed + a.fold))
    lb_it = iter(lb_dl)
    if use_unlabeled:
        ulb_dl = DataLoader(Texts(corpus), batch_size=a.batch_size, shuffle=True,
                            collate_fn=lambda z: tok(list(z), padding=True, truncation=True,
                                                     max_length=128, return_tensors="pt"),
                            generator=torch.Generator().manual_seed(19_000 + 100 * a.data_seed + a.fold))
        ulb_it = iter(ulb_dl)
    for step in range(a.steps):
        model.train()
        try: b, y, _ = next(lb_it)
        except StopIteration: lb_it = iter(lb_dl); b, y, _ = next(lb_it)
        b = {k: v.to(dev) for k, v in b.items()}; y = y.to(dev)
        if use_arc:
            pred, logits = forward_arc(model, arc, b, model.dropout)
        else:
            pred = model(**b)
        loss_reg = F.l1_loss(pred, y)
        if use_arc:
            loss_arc = soft_rank_loss(logits, y)
            if use_unlabeled:
                try: ub = next(ulb_it)
                except StopIteration: ulb_it = iter(ulb_dl); ub = next(ulb_it)
                ub = {k: v.to(dev) for k, v in ub.items()}
                pred_w, weak = forward_arc(model, arc, ub, model.dropout)
                strong_tok = mask_tokens(ub, tok, 0.15) if a.strong_view == "mask" else ub
                pred_s, strong = forward_arc(model, arc, strong_tok, model.dropout)
                loss_u, cov = unlabeled_rank_loss(weak, strong)
                loss_arc = loss_arc + a.arc_ulb_weight * loss_u
            if a.rrc_weight > 0:
                rrc = labeled_rrc(pred, y, a.rrc_margin)
                if use_unlabeled:
                    rrc = rrc + unlabeled_rrc(pred_s, weak, a.rrc_margin)
                loss = loss_reg + a.arc_weight * loss_arc + a.rrc_weight * rrc
            else:
                loss = loss_reg + a.arc_weight * loss_arc
        else:
            loss = loss_reg
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(arc.parameters()) if arc else []), 1.0)
        opt.step()
        if (step + 1) % 320 == 0:
            print(f"step {step+1}/{a.steps} reg={float(loss_reg):.6f}", flush=True)

    model.eval()
    dl = DataLoader(GoldData(lb_ho), batch_size=32, shuffle=False, collate_fn=TextCollator(tok, 128))
    ys, ps, ids = [], [], []
    with torch.inference_mode():
        for b, y, bid in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            ps.append(model(**b).float().cpu().numpy()); ys.append(y.numpy()); ids.extend(bid)
    Y, P = np.concatenate(ys), np.concatenate(ps)
    pd.DataFrame({"id": ids, "fold": a.fold, "y_V": Y[:, 0], "y_A": Y[:, 1], "y_D": Y[:, 2],
                  "p_V": P[:, 0], "p_A": P[:, 1], "p_D": P[:, 2]}).to_parquet(
        out / "sup_predictions.parquet", index=False)
    torch.save({"model": model.state_dict(), "steps": a.steps, "lr": a.lr, "batch_size": a.batch_size,
                "fold": a.fold, "data_seed": a.data_seed, "arm": a.arm}, out / "base.pth")
    (out / "run_config.json").write_text(json.dumps({**vars(a), "out_dir": str(out)}, indent=2, default=str) + "\n")
    print(f"BASE {a.arm} fold={a.fold} MAE={float(np.abs(P-Y).mean()*4):.6f}", flush=True)


if __name__ == "__main__":
    main()
