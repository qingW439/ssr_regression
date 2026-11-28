# Local Geometry Root-Cause Decision

## 1. Why the earlier 0.946 / 0.029 claim was invalid

The earlier CPMD/retrieval probe used `features_labeled.npy` from the same fold’s supervised model and computed pairwise distances over those 400 training samples. Those features are **in-sample**: the encoder was trained on the labels attached to them. Treating all pairwise distances as independent also overstates precision.

The corrected audit uses:

- **query = fold holdout (100 samples)**, whose labels were not used to train that fold’s encoder;
- neighbours = only the fold’s legal 400 TRAIN labels;
- no self-retrieval and no ID overlap;
- per-query Spearman correlation, not pooled pair correlation;
- 100 random-label permutations per query;
- five folds.

## 2. Corrected local-geometry statistics

| Statistic | Earlier claim | Corrected holdout audit |
|---|---:|---:|
| Feature–VAD distance correlation | 0.946 Pearson | **0.743 mean per-query Spearman** (query CI 0.303–0.952) |
| 1-NN VAD distance | 0.029 | **0.0868** |
| Random-label VAD distance | 0.153 | **0.151** |
| kNN MAE vs B2 | −0.0003 (fold0) | **+0.00288 mean over 5 folds** |
| kNN residual correlation | positive folded signal | **near zero or negative** for V/A/D |

Local structure is real, but **weaker than claimed** and much less precise.

## 3. Does the neighbourhood carry incremental residual information?

No stable evidence:

- kNN MAE is worse than B2 in every fold (Δ +0.0014 to +0.0052; mean +0.00288).
- The residual correlation between kNN and B2 error is not consistently positive:
  - V: −0.20, −0.16, −0.13, −0.21, −0.20;
  - A: +0.14, 0.00, −0.12, 0.00, −0.09;
  - D: +0.03, +0.14, +0.03, +0.26, +0.06.
- This is consistent with **re-smoothing the teacher/B2 prediction**, not adding new residual information.

## 4. Decision

**B: terminate the local-retrieval hypothesis.**  
RARU and CPMD are not rescued by more neighbours, blend weights or calibration. No seed1/2, no random-neighbour escalation.

The only retained baseline failure with strong, repeated evidence is **tail compression / conditional-mean shrinkage**, especially in D and A.

## 5. Follow-up candidate tested: Quantile Pseudo-label Alignment (QPA)

QPA rank-maps teacher pseudo-VAD onto the TRAIN label quantiles and trains a student with tail-weighted pseudo-L1 + labeled ARC.

| Model | Stage | MAE |
|---|---|---:|
| B2 | base | 0.15186 |
| QPA | base | **0.19000** |
| B2 | 96-step readout | 0.15104 |
| QPA | 96-step readout | **0.18593** |

`QPA = NO-GO`. Expanding the pseudo-label marginal without reliable conditional information destroys the regression; it is not a valid tail fix.

## 6. Current state

- `LOCAL_RETRIEVAL = NO-GO`
- `RARU = NO-GO`
- `CPMD = NO-GO`
- `QPA = NO-GO`
- `M1 = NOT_COMPLETED`

Next work must return to the failure atlas and target a different mechanism; no more retrieval/neighbour scans.
