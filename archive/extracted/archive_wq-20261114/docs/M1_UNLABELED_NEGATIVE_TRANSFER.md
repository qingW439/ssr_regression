# Unlabeled-Training Negative Transfer

## ARC (RankUp)

Evidence from B2 vs B3, same UVR/Strong recipe and same fold:

| Fold | B2 MAE | B3 MAE | B2 Pearson | B3 Pearson |
|---:|---:|---:|---:|---:|
| 0 | 0.17220 | 0.17100 | 0.4621 | 0.5644 |
| 1 | 0.16277 | 0.18826 | 0.5482 | 0.5097 |

- unlabeled pair coverage: 0% → 10.2% → 14.1% → 13.5%;
- regression grad norm: 4.51 → 0.68;
- unlabeled ARC grad norm at step 800: **21.62**, regression **1.21**;
- supervised ARC grad 2.56–5.50.

The unlabeled ranking gradient is not calibrated to the continuous regression scale. Fold0 Pearson gain did not replicate; fold1 MAE and Pearson both degrade. **VERIFIED negative transfer** for this implementation.

## Noisy Student

Same samples, B2 vs student:

| Arm | MAE | Pearson | CCC | R² |
|---|---:|---:|---:|---:|
| B2 | **0.16762** | **0.54349** | **0.43767** | **0.30421** |
| Noisy Student | 0.17085 | 0.51305 | 0.39718 | 0.26981 |

ΔMAE +0.00323, CI [+0.00079, +0.00569]. Teacher macro bias is −0.0122 and SD ratio 0.498. The student copies the teacher’s compressed target; fold0 gain does not replicate.

## CPMD

The corrected holdout audit (not the old in-sample 0.946 statistic):

- valid query Spearman ≈ 0.743; 1-NN VAD distance 0.0868 vs random 0.1509;
- kNN MAE is worse than B2 by +0.00288 over five folds;
- residual correlations between kNN and B2 error are not consistently positive.

CPMD therefore has no stable incremental information to distill; **VERIFIED**.

## QPA

Expanding pseudo-label quantiles:

| Stage | B2 | QPA |
|---|---:|---:|
| base | 0.15186 | **0.19000** |
| 96-step readout | 0.15104 | **0.18593** |

`QPA = NO-GO`; distribution expansion without conditional evidence is destructive.

## Diagnosis

The repeated pattern is not “unlabeled data is useless”. It is:

1. the unlabeled objective is optimized on a **different scale** from continuous VAD regression;
2. teacher/pseudo-targets carry the same **conditional-mean compression** as the base model;
3. adding a ranking/distribution gradient changes the encoder in a direction that helps one fold but not the five-fold ranking;
4. the base UVR gain itself is mostly a ranking gain, not a variance fix.

The next mechanism must not add another global pseudo-label/ranking gradient. It must use unlabeled information in a way that is **scale-aware and conditionally local**, or it must avoid changing the encoder’s regression geometry at all.

## Candidate decision after this audit

The strongest remaining measured failure is **D/A underprediction and tail compression**, but the unlabeled-training evidence says that global pseudo-target expansion is unsafe. A safe next mechanism must therefore:

- operate on the **output/decision stage**, not add another encoder-wide pseudo-label gradient;
- use unlabeled information only through a **conditional, low-capacity** correction;
- be ablated by removing the unlabeled pool.

The first candidate in this direction was QPA; it failed because it expanded the marginal unconditionally. The next candidate must condition the correction on **local V/A predictions and text**, not on the global label quantiles.

That candidate has not yet been implemented; this document only closes the negative-transfer diagnosis.
