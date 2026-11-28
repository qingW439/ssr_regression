# BASELINE_README — compared models: source, config, how to run, results

Every entry: code location → exact command → 15-fold (or 3-unit) result.
All numbers recomputed from OOF with `src/hdtm/evaluate.py` unless marked "reported".

## How to run any baseline

```
bash scripts/run_baseline.sh <ARM> <seed> <fold>
# ARM ∈ U0 U1 THS H-S H-U C-S C-U HPL RCUS AAN SUP6 UUL6 HDT-D HDT-M
```
New run dirs go to `$SSR_ASSETS/m1_runs_unified/v0.1.2-hdtm-ready/`. A run only counts as complete when
`tools/check_status.py --experiment <MODEL> --version v0.1.2-hdtm-ready --seed <seed> --fold <fold>` passes.

## RankUp family (external baseline)

- Code: `src/baselines/train_rankup.py` (originally `train_rankup_official.py`).
- Arms: `b1/b3/b4` add ARC ranking + RRC; `b2` is the plain 3-D regressor used as the cross-fit teacher.
- Protocol: 3200 steps, batch 16, lr 1e-5, decay 0.75, 96-step readout; EmoBank 500-label budget.
- RankUp-3D v2 MAE 0.186333 and Full v2 (RDA) MAE 0.185048 recompute from the 15
  readout-stage `sup96/oof_predictions.parquet` files for `off_b3v2` and `off_b3rdav2`.
  Each has 100 unique IDs per fold matching the fold manifest; both use VAD MAE ×4 averaged
  over 15 units. The earlier `sup_predictions.parquet` files are a different training stage.
- RankUp "raw" nested-OOF MAE 0.181740 / CCC 0.426728 uses 500-sample nested OOF
  (`Semi-Supervised-Regression-rankup-m1/outputs/oof/oof_metrics.json`), a separate protocol.

## S (strong supervised scalar)

- Code: `src/baselines/train_uul.py --arm off` (no unlabelled term), 400 labels/fold.
- Three-seed summary (`M1_BASELINE_V2_3SEED.csv`, `S_v1` mean row): MAE 0.180800.
  This summary is a separate protocol from the formal 15-fold OOF table.

## U0 / U1 / T-HS (scalar student)

- Code: `src/baselines/train_uul.py`; U0 = `--arm off`, U1 = `--arm raw --lambda-u 0.25`,
  T-HS = same as U1 but `--targets m1_hdt_gate/targets_th/B_s{seed}_f{fold}.npz` (distribution-teacher values).
- Formal 15 folds: U0 0.180315 / 0.416833, U1 0.178130 / 0.414510 (MAE / CCC).
- T-HS (3 dev units): 0.173546 / 0.422919.

## HPL-3D

- Code: `src/baselines/train_hpl.py` (weak/strong consistency with learned uncertainty, bilevel `higher`).
- Formal 15 folds: 0.177314 / 0.433869.

## SSR-RCUS-3D (ranking correction)

- Code: `src/baselines/train_rcus.py` + `src/baselines/rcus_loss.py` (per-dim rank scores, BCE pair loss,
  manifold mixup, τ=0.95 mask). The "fixed" variant is `train_rcus2.py` → `src/baselines/train_rcus.py`.
- Formal 15 folds: 0.196126 / 0.494020 (best rank correlation, worst MAE).

## AAN (3-D adaptation)

- Code: `src/baselines/train_aan.py`.
- Formal 15 folds: **0.188113 / 0.482730** (recomputed from OOF); weights were removed in the
  consolidation cleanup, so inference requires retraining. Metrics remain reproducible.

## H-S / H-U / C-S / C-U (histogram student)

- Code: `src/baselines/train_hist.py` (+ `src/hdtm/hist_loss.py`).
- Formal 15 folds: H-S 0.175208 / 0.470203, H-U 0.174910 / 0.429681.
- Controls (3 units): C-S 0.174036 / 0.418093, C-U 0.176501 / 0.396894.

## SUP-6 / UUL-6 (6-layer depth study)

- Code: `src/baselines/train_shallow.py` (`sup6`, `uul6`). **UUL-6 is a 6-layer encoder semi-supervised
  model, not "six unlabelled batches per step"** (the earlier wording was wrong).
- Formal 15 folds: SUP-6 0.187393 / 0.397537, UUL-6 0.180128 / 0.411744; weights removed
  (metrics reproducible, inference requires retraining).

## HDT-M / HDT-D

- Code: `src/hdtm/train_hdt.py` (student) + `src/hdtm/train_teacher_hist.py` (teachers) +
  `src/hdtm/make_fused_fold.py` (fusion).
- HDT-M formal 15 folds: **0.173036 / 0.465931** (weights kept, inference verified).
- HDT-D (3 units): 0.173016 / 0.445523 (direct distribution transfer, no independent gain).

## Missing / degraded

| asset | state |
| --- | --- |
| AAN, SUP-6, UUL-6 readout weights | all 45 restored 2026-11-28; paths, sizes and SHA256 recorded in `docs/ASSET_MANIFEST.json`. Their pre-readout `base.pth` files remain on the preserved second host. |
| P1 studies (`m1_p1_runs`) | result.json only, no OOF; superseded |

## Current host capacity and replay

On 2026-11-28 the connected container reported no usable CUDA device (`torch.cuda.is_available() = False`; `nvidia-smi` permission denied). Its cgroup memory limit was 2 GiB even though host-wide memory was much larger. Loading one checkpoint for artifact validation succeeded. A fixed-input replay that also instantiates RoBERTa can exceed this 2 GiB container limit; this is the likely cause of the earlier killed process, but no replay-specific OOM log was found and the current cgroup `oom_kill` counter was zero. Repeat the inference replay only with a GPU or a larger container limit. No model was retrained during this audit.
