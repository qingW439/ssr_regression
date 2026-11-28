# CHANGELOG

## v0.1.1-hdtm-repro — 2026-11-28

- Restored baseline implementations into the repo: U0/U1/T-HS (`src/baselines/train_uul.py`),
  H-S/H-U/C-S/C-U (`train_hist.py`), HPL (`train_hpl.py`), RCUS (`train_rcus.py` + `rcus_loss.py`),
  SUP-6/UUL-6 (`train_shallow.py`), AAN (`train_aan.py`), RankUp-3D/Full (`train_rankup.py`).
- Removed the dependency on old project directories: entrypoints now import from `src/` only,
  assets are resolved through `SSR_ASSETS` (`src/common/paths.py`).
- Added `tools/check_status.py`: completion requires weights + result + OOF + id match + config
  (+ teacher caches/fused target for HDT-M); incomplete runs are reported, not silently accepted.
- Added `scripts/run_baseline.sh` and rewrote `scripts/hdtm_*.sh` to call repo code.
- Audit of the consolidation cleanup: AAN / SUP-6 / UUL-6 readout weights were deleted
  (OOF + result.json kept; metrics reproducible, inference needs retraining). RankUp-3D v2 15-fold
  OOF was never in the tree. Recorded in `docs/RESEARCH_ROADMAP.md`.
- Fixed documentation: UUL-6 is a 6-layer model; teacher–student correlation sign explained;
  baseline README rewritten as a real comparison document; source/version map added.

## v0.1.0-hdtm — 2026-11-28

First consolidated research baseline.

- Consolidated 7 historical branches into `wq-main`; historical text records extracted to
  `archive/extracted/`; Git bundle backup created before deletion.
- Added the actual HDT-M pipeline under `src/hdtm/` (histogram head, Gaussian target, boundary
  corrected decoder, distribution teachers, teacher fusion, student training, evaluation).
- Formal 15-fold result: **MAE 0.173036**, CCC 0.465931, Pearson 0.518222, SRCC 0.482137,
  R² 0.257238, tail MAE 0.264927 (best MAE and R² of all compared models; 12/15 folds beat H-S).
- Distribution teachers: MAE 0.182044 → 0.175681, CCC 0.405889 → 0.464975 (11/12 and 12/12 wins).
- Recorded rejected hypotheses: teacher-student expression matching (2×2 interaction −0.002288),
  HDT-D direct distribution transfer (KL 0.008–0.027 nat), HC-M/HC-D consistency learning.
- Corrected a previously mis-reported T-HS mean (0.174946 → 0.173546).

## Prior history

The RankUp reproduction baseline and earlier M1 experiments are kept in
`docs/BASELINE_README.md`, `archive/extracted/` and the backup bundle
(`backups/ssr_pre_consolidation_20261128.bundle`).
