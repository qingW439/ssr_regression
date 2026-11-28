# RESEARCH_ROADMAP — M1 / M2 / M3

## M1 (current): high-performance semi-supervised text multi-dimensional VAD regression

Established:
1. **Histogram distribution regression** (head + Gaussian target + CE + boundary-corrected decoding)
   is the strongest regression structure: H-S 0.175208 MAE / 0.470203 CCC (15 folds).
2. **Distribution-regression teachers** beat scalar teachers on the same cross-fit splits:
   MAE 0.182044 → 0.175681, CCC 0.405889 → 0.464975 (12 teachers, 11/12 MAE, 12/12 CCC).
3. **HDT-M** (distribution teachers + histogram student, λ=0.25) is the current candidate:
   **0.173036 MAE**, 0.465931 CCC, 0.257238 R² (15 folds, best MAE in all 3 seeds).

Open issues:
- CCC/tail still slightly below H-S (ΔCCC −0.0043, Δtail +0.0022); the gain is concentrated in MAE/R².
- Teacher-student "expression matching" is **not** a valid mechanism claim (interaction −0.0023).
- The independent novelty of the teacher-side change still needs a separate argument; the pipeline
  is an effective system, not yet a proven new algorithm.

Next (not started): localise the remaining tail/CCC cost; decide whether a new unlabelled signal is
needed. C-S/C-U (60-dim head + L1) formal 15-fold is an open control (dev MAE 0.174036).

## M2: continuous pseudo-label reliability

Untouched. Must stay separate from M1: no uncertainty weighting, confidence filtering or amplitude
calibration has been used in any M1 candidate.

## M3: Chinese-English tourism-domain adaptation

Not started. Requires new data; DEV/TEST of EmoBank remain untouched.

## Rules

- One long-lived branch `wq-main`; no per-experiment branches.
- Every improvement: source change → validation → config → ledger → commit → tag on major versions.
- Never access EmoBank DEV/TEST; never fit weights/calibration on outer holdout labels.


## History preserved (do not erase)

- HPL-3D (weak/strong consistency with learned uncertainty, bilevel `higher`) was implemented and
  formally evaluated (15 folds, 0.177314 / 0.433869).
- A mean-teacher component was used in earlier M1 work (`P1_v2 + mean-teacher raw + edit_V`);
  EMA teachers are therefore *explored history*, not an untouched idea.
- JACF / RDA / UVR experiments and reports are archived in `archive/extracted/` and the bundle.

## Known asset gaps (2026-11-28 audit)

| asset | state |
| --- | --- |
| AAN, SUP-6, UUL-6 readout weights | **recovered 2026-11-28** from the preserved second host (45 readouts restored, all sizes match, 6 sampled sha256 identical; base.pth of these runs remain only on that host) |
| RankUp-3D v2 15-fold OOF | not present; only reported numbers |
| P1 study weights (`m1_p1_runs`) | removed earlier (result.json only, superseded) |
