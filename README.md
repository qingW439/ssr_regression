# SSR-regression — semi-supervised text multi-dimensional emotion regression

Consolidated research repository for the **M1 / M2 / M3** line of work
(M1: semi-supervised VAD regression; M2: pseudo-label reliability; M3: cross-lingual domain adaptation).

Current main model: **HDT-M (Histogram Distribution Transfer)** — EmoBank 500-label budget,
3 seeds × 5 folds, formal result **MAE 0.173036**, CCC 0.465931, R² 0.257238.

VAD = Valence (how positive), Arousal (how excited), Dominance (how in control), each in [0,1].
MAE below is ×4 so that it reads on the original 1..5 scale.

## Layout

```
src/hdtm/          actual HDT-M / teacher / fusion / evaluation code (moved verbatim)
configs/hdtm/      teacher and student hyperparameters
scripts/           entry points (teacher, student, fusion, formal queue)
experiments/       formal table (15 folds) and summary
docs/              architecture, roadmap, ledger, failures, branch index, asset manifest
tests/             numeric checks (histogram loss, decoder/boundary correction)
archive/extracted/ historical text records from the consolidated branches
semilearn/         baseline framework (RankUp reproduction, unchanged)
```

## Reproduce (server paths)

```
# 1. teachers for one outer unit (4 teachers, same cross-fit split as the scalar teachers)
python src/hdtm/train_teacher_hist.py --seed 0 --fold 0 --j 0 --out teachers/B_s0_f0_j0.npz

# 2. fuse teacher probabilities -> continuous pseudo labels
python src/hdtm/make_fused_fold.py 0 0

# 3. student (HDT-M)
python src/hdtm/train_hdt.py --arm hdtm --seed 0 --fold 0 \
  --fused m1_hdt_gate/fused/B_s0_f0.npz --out-dir m1_hdt_runs/hdtm_s0_f0

# 4. full 15-fold queue (resume-safe)
bash scripts/hdtm_run_formal.sh
```

Data, weights and teacher caches stay outside git; see `docs/ASSET_MANIFEST.json` for their
server locations and hashes.

## Baselines

RankUp-3D v2 reproduction lives in `semilearn/`, `train.py`, `config/` (see `docs/BASELINE_README.md`).
HPL / RCUS / U0 / U1 / H-S / H-U results are in `docs/EXPERIMENT_LEDGER.md`.

## Rules

- One branch (`wq-main`), tags for milestones; no per-experiment branches.
- Never touch EmoBank DEV/TEST; never fit anything on outer holdout labels.
- Failures are recorded in `docs/FAILURE_SUMMARY.md`, not deleted.
