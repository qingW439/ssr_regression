# M1 Failure Root Causes

Scope: existing seed0 B0/B2 OOF predictions, ARC diagnostics, Noisy Student, CPMD, DRILL/DDA logs. No DEV/TEST.

## 1. RankUp ARC

**VERIFIED phenomena**

- unlabeled pair coverage: 0% at step 0, 10.2% at 320, 14.1% at 800, 13.5% at 1600;
- regression gradient norm falls from 4.51 to 0.68;
- ARC supervised gradient stays 2.6–5.5; unlabeled ARC gradient spikes to 21.62 at step 800;
- B3 5-fold Pearson is not stable: fold0 0.5644 vs B2 0.4621, fold1 0.5097 vs B2 0.5482.

**Root cause:** the ranking objective is not merely “missing information”. Its gradient becomes large relative to the shrinking regression gradient, and the pseudo-pair coverage is low and class-imbalanced. The shared encoder is pulled toward a pair-classification solution that does not preserve the continuous VAD metric. This is an **optimization/coordination failure**, not proof that ranking is universally useless.

**Evidence strength:** VERIFIED for gradient/coverage; PLAUSIBLE for the exact causal share of information vs optimization.

## 2. Noisy Student

**VERIFIED phenomena**

- fold0 base MAE 0.14872 vs B2 base 0.15186;
- five-fold final MAE 0.17085 vs B2 0.16762; Δ +0.00323, CI [+0.00079, +0.00569];
- teacher bias is already −0.0122 macro and SD ratio 0.498; the teacher compresses variance.

**Root cause:** the student is trained to copy a compressed teacher. On fold0 this acts as a regularizer; across folds it accumulates the teacher’s conditional-mean bias. This is a **teacher-error-copying** failure, not evidence that unlabeled text is useless.

**Evidence strength:** VERIFIED.

## 3. CPMD

**VERIFIED phenomenon**

On TRAIN labeled UVR features:

- pairwise feature distance vs pairwise VAD distance: Pearson **0.946**, Spearman **0.958**;
- mean 1-NN VAD distance **0.0290**, 5-NN **0.0361**, 20-NN **0.0505**, random pair **0.1533**.

**Conclusion:** the local-continuity assumption is **strongly supported** by the feature geometry. CPMD did not fail because neighbours are semantically unrelated. Its failure is more likely due to **feature-space mixup creating points that do not correspond to a valid text**, and/or mixing teacher labels with a linear rule that is not calibrated at the tails.

**Evidence strength:** VERIFIED for geometry; PLAUSIBLE for the mixup failure mechanism.

## 4. DRILL / DDA

**VERIFIED phenomena**

- fold0 base MAE 0.14923 vs B2 base 0.15186;
- five-fold base MAE 0.17032 vs B2 base 0.16922, Δ +0.00109, CI [−0.00125, +0.00343];
- final readout degrades more than base.

**Root cause:** the fold0 signal is a fold-specific distribution/optimization effect, not a stable mechanism. The ordinal bucket head plus teacher distribution changes the output scale and tail behavior; across folds the student does not preserve the B2 ranking.

**Evidence strength:** VERIFIED instability; PLAUSIBLE for the exact distributional cause.

## 5. Shared baseline failure: tail compression

**VERIFIED phenomena**

- B2 SD ratio: V 0.583, A 0.455, D 0.455;
- low-V bin bias +0.288, high-V bin bias −0.301;
- low-D bin bias +0.207, high-D bin bias −0.255;
- D Pearson 0.403 (lowest), A 0.491, V 0.737.

This is a direct conditional-mean compression. It is a **prediction-target/decision-rule failure**, not necessarily an unlabeled-information failure.

## 6. What is actually underused

The UVR feature geometry contains strong local VAD information (feature–VAD distance correlation 0.946), but the parametric head does not use the labeled/unlabeled neighbourhood at inference. A retrieval probe using labeled + pseudo-labeled unlabeled neighbours reaches fold0 MAE 0.1507 and Pearson 0.6875, versus B2 0.1510/0.6798. This is the most concrete untapped signal found so far.

**Caveat:** the retrieval probe is single-fold and must be validated with five folds and an unlabeled-pool ablation. It is a candidate mechanism, not yet a verified component.
