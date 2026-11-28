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

## Run

```
export SSR_ASSETS=/root/autodl-tmp/rankup_project   # data / weights / caches live here

# any compared model, one unit
bash scripts/run_baseline.sh HDT-M 0 0
bash scripts/run_baseline.sh U1 0 0          # U0, U1, THS, H-S, H-U, C-S, C-U,
                                             # HPL, RCUS, AAN, SUP6, UUL6, HDT-D

# HDT-M full 15 folds (teachers -> fused targets -> student), resume-safe
bash scripts/hdtm_run_formal.sh

# completion check (weights + result + OOF ids + config + teacher caches)
python tools/check_status.py --runs "$SSR_ASSETS/m1_hdt_runs/hdtm_s*_f*" --recompute --teacher-guard
```

All model code is imported from `src/`; the old `m1_*_gate` directories are historical copies and are
never used by the entrypoints. See `docs/SOURCE_MAP.md` for model → source → origin.

## Baselines

RankUp-3D v2 reproduction lives in `semilearn/`, `train.py`, `config/` (see `docs/BASELINE_README.md`).
HPL / RCUS / U0 / U1 / H-S / H-U results are in `docs/EXPERIMENT_LEDGER.md`.

## Rules

- One branch (`wq-main`), tags for milestones; no per-experiment branches.
- Never touch EmoBank DEV/TEST; never fit anything on outer holdout labels.
- Failures are recorded in `docs/FAILURE_SUMMARY.md`, not deleted.
