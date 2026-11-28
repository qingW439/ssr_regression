# FAILURE_SUMMARY — closed lines and the evidence

Each line: what was changed, where it failed, evidence, remaining hypothesis.

## Closed (no further work)

| line | change | evidence | status |
| --- | --- | --- | --- |
| Multi-layer feature fusion (F1/F2/F1-LR) | fuse encoder layers before the head | 15-fold MAE worse than U1 on every seed | permanently closed |
| ModernBERT encoder | swap RoBERTa for ModernBERT | 3200 steps: 0.191977; 6400 steps: 0.197159 (+0.0177 vs U1) while train L1 0.00982 < U1 0.01165 | closed after single-factor fix |
| TARE token editing | edit emotion tokens | no gain; weights deleted, evidence only in logs | closed |
| Shallow/deep auxiliary supervision (DS-L/U/LU) | aux head at layer 6 | aux learns (MAE 0.180–0.198) and gradients reach layers 1–6, but main path unchanged (ΔMAE +0.0011/−0.0004/+0.0004) | closed |
| RCUS ranking | pairwise rank loss + mixup | CCC 0.494 but MAE 0.196126 (15 folds); loses mid-range | kept only as a comparison |
| HC-M (value consistency) | weak/strong L1 consistency, λ matched by encoder gradient | 47% ratio diverged (MAE 0.48–0.59); at 4.7%: MAE 0.183754, sd_ratio 0.811 (over-dispersed) | closed |
| HC-D (distribution consistency) | KL between weak/strong 3×20 distributions | diverged even at 4.7% (MAE 0.294426, sd_ratio 1.35) | closed |
| HDT-D (direct distribution transfer) | unlabelled target = fused teacher probabilities | KL(Q‖Gaussian) only 0.008–0.027 nat, width gap 0.003 → no extra information; MAE 0.173016 vs HDT-M 0.172451 | paused |
| Teacher-student "expression matching" | 2×2 cross | interaction −0.002288 (all 3 units negative) | hypothesis rejected |

## Not tested / open

- Mean-teacher style EMA target for the unlabelled branch (the project did use a mean-teacher
  component historically in `P1_v2 + mean-teacher raw + edit_V`; it was not re-tested here).
- Anchored consistency (consistency + small pseudo-label anchor).
- Feature-level consistency instead of output-level.
- PP-SSL (NeurIPS 2025) gradient correction — potential overlap with M2.
- A-LL / A-CL / A-LC asymmetric supervision — not started.

## Lessons that are now binding

- Loss-scale matching at initialisation does not hold during training; measure, don't assume.
- Over-dispersion and under-dispersion are different failures; sd_ratio alone is not a target.
- Any "synergy" claim needs the 2×2 cross, not a single replacement.
