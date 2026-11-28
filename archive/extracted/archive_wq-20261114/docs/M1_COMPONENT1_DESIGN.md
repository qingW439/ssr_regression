# M1 Component-One Design: Retrieval-Augmented RankUp (RARU)

## 1. What is borrowed

- kNN regression / label propagation: non-parametric prediction from a similarity-weighted neighbourhood.
- Retrieval-augmented NLP: keep an external memory and combine it with a parametric prediction.
- RankUp: shared text encoder, continuous VAD head, auxiliary ranking-derived representation.

RARU is **not** a re-run of CPMD. CPMD synthesised mixed features and trained on mixed targets; RARU never creates synthetic points. It uses real labeled examples and real unlabeled texts with teacher pseudo-VAD, and only changes the inference decision rule.

## 2. Insertion point

```
text x
  -> UVR/RankUp encoder f_theta
  -> parametric VAD f_theta(x)
  -> retrieve neighbours from memory M
  -> weighted VAD y_hat_M(x)
  -> p(x) = (1-lambda) f_theta(x) + lambda y_hat_M(x)
```

Memory `M`:

- labeled 400 training texts: true VAD;
- unlabeled 6657 pool texts: frozen B2 teacher VAD (pseudo-labels);
- optional: no unlabeled for the ablation.

No gradient flows through the memory. The encoder/head are unchanged, so any gain is attributable to the decision rule and the external unlabeled memory.

## 3. Mathematical definition

For dimension \(d\in\{V,A,D\}\):

\[
\hat y_d(x) = (1-\lambda_d) f_{\theta,d}(x) + \lambda_d
\frac{\sum_{i\in \mathcal N_k(x)} \exp(-d_i/\tau_d)\, y_{i,d}}
{\sum_{i\in \mathcal N_k(x)} \exp(-d_i/\tau_d)}
\]

- \(\mathcal N_k(x)\): top-\(k\) cosine neighbours in the UVR embedding;
- \(d_i = 1-\cos(h_x,h_i)\);
- labeled neighbours use true \(y_i\);
- unlabeled neighbours use teacher pseudo-\(\tilde y_i\);
- \(\lambda_d,\tau_d,k\) are selected on TRAIN via inner cross-fitting; the outer holdout is never used.

Initial fixed values for the screening experiment: \(k=50\), \(\tau\) from distances, \(\lambda=0.5\), and a pure-retrieval variant \(\lambda=1\).

## 4. Why this could improve MAE and Pearson

- The parametric head compresses tails; neighbor labels preserve local extremeness.
- The UVR geometry is strongly VAD-aligned (pairwise distance correlation 0.946), so neighbors are informative.
- Averaging over 50 neighbors reduces pseudo-label noise; the unlabeled pool increases coverage of rare text types.
- Pearson can improve because neighbor labels preserve local ordering even when the global scale is compressed.

## 5. Differences from existing work

| Method | Unlabeled use | Decision rule | Synthetic points |
|---|---|---|---|
| RankUp ARC | pseudo-pair ranking | parametric | no |
| CPMD | neighbor mixup targets | parametric | yes |
| DRILL/DDA | distribution distillation | parametric | no |
| Noisy Student | pseudo-VAD self-training | parametric | no |
| **RARU** | pseudo-VAD memory | **non-parametric retrieval + parametric** | no |

The contribution is the **memory-based decision rule on a RankUp/UVR encoder**, not kNN itself.

## 6. Training / selection protocol

1. Train or reuse the matched B2 base (no new encoder training for the screen).
2. Build memory from the 400 TRAIN labels and the 6657 unlabeled pool.
3. Select \(k,\lambda,\tau\) by inner cross-fitting on the 400 TRAIN labels (leave-one-out or 5 inner folds). No outer holdout labels.
4. Freeze the rule and evaluate the outer holdout.
5. Seed0 five folds first; if pass, seeds 1/2.

## 7. Ablations / negative controls

- **C0**: B2 parametric prediction.
- **C1**: B2 + retrieval with labeled + unlabeled memory.
- **C2**: B2 + retrieval with **labeled-only** memory (unlabeled ablation).
- **C3**: B2 + retrieval with **random memory** / shuffled neighbor labels.
- **C4**: B2 + retrieval from **raw RoBERTa** features (geometry control).

## 8. Failure risks

- Pseudo-label noise in the unlabeled memory;
- retrieval gain may be labeled-only, not unlabeled;
- transductive memory may not generalize to new corpora;
- kNN memory cost at inference;
- fold-specific gain.

If C2 removes the gain, the component is a labeled retrieval method, not an unlabeled component.

## 9. Screening result (added after the pre-registered run)

RARU was run on seed0 five folds with a 50-neighbour memory and fixed blend:

| Arm | MAE | Pearson | CCC | R² |
|---|---:|---:|---:|---:|
| B2 parametric | **0.16762** | **0.54349** | **0.43767** | **0.30421** |
| RARU labeled+unlabeled, λ=0.5 | 0.16788 | 0.54276 | 0.40759 | 0.29432 |
| RARU labeled only, λ=0.5 | 0.16835 | 0.54370 | 0.39237 | 0.28991 |
| RARU labeled+unlabeled, pure retrieval | 0.17049 | 0.52951 | 0.37065 | 0.26916 |

The retrieval branch **does not beat B2** on five folds: MAE +0.00026, Pearson −0.0007, CCC −0.030. The fold0 Pearson gain did not replicate. Random-neighbour and seed1/2 controls were therefore not escalated.

`RARU = NO-GO` as a component. The local geometry is real, but this decision rule is not sufficient.

## 10. Acceptance

Use `M1_COMPONENT1_ADMISSION.md`:

- primary MME MAE 0.0010 or Pearson 0.0050;
- other metric non-inferior (MAE +0.002, Pearson −0.010);
- unlabeled ablation C2 must reduce the gain;
- random-neighbour C3 must remove it;
- seed0 five-fold, then seeds 1/2.
