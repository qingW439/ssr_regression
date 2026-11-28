# UVR Mechanism Discrimination

## Base-stage matched comparison (seed0, five folds)

| Initialization | MAE | Pearson | CCC | R² |
|---|---:|---:|---:|---:|
| Raw RoBERTa | 0.17580 | 0.47398 | 0.34643 | 0.23280 |
| DAPT (MLM on same pool) | 0.17582 | 0.48832 | 0.35774 | 0.24352 |
| UVR-SimCSE | **0.16922** | **0.53100** | **0.40773** | **0.28449** |

- DAPT and Raw have essentially identical MAE (0.17582 vs 0.17580); DAPT gains only +0.014 Pearson. Extra MLM pretraining on the same unlabeled corpus is **not** the MAE source.
- UVR adds **−0.00660 MAE** and **+0.0427 Pearson** over DAPT. The contrastive UVR representation, not just extra pretraining compute, carries the gain.
- Three-seed UVR vs Raw base: MAE −0.00424/−0.00393/−0.00486; Pearson +0.0258/+0.0210/+0.0489. The direction is stable.
- SD ratio changes only 0.43→0.44–0.46; the gain is mostly ranking/representation, not variance calibration.
- E-DAPT checkpoint is absent in the workspace; E-DAPT comparison is **UNKNOWN**, not assumed.

**Conclusion:** the discriminated mechanism is the UVR representation/contrastive information, not extra pretraining steps alone. This is a mechanism-level statement, not proof that every downstream use of the representation will help.

## New component attempt: Relative Geometry Distillation (RGD)

Hypothesis: preserve the frozen UVR pairwise similarity matrix on unlabeled texts during supervised fine-tuning, preventing the ranking geometry from collapsing.

\[
L = L_{1,labeled} + \lambda \|S_{student}(x_u) - S_{UVR}(x_u)\|_F^2
\]

C1 = geometry on; C2 = identical model/compute with geometry off.

| Arm | MAE | Pearson | CCC | R² |
|---|---:|---:|---:|---:|
| B2 | **0.15104** | **0.5930** | **0.4984** | **0.3469** |
| RGD C1 | 0.24223 | 0.2949 | 0.2693 | −0.8783 |
| RGD C2 | 0.15315 | 0.5602 | 0.4874 | 0.3299 |

RGD C1 collapses. The pairwise matrix loss is on a different scale from L1 and dominates the encoder; C2 shows the base recipe itself is near B2 but not identical due RNG/code path. This is not a data problem.

`RGD = NO-GO`. The next repair would need a scale-normalized/adaptive geometry weight; but this is a second-order fix of a mechanism that already failed at its first test, and the user’s evidence rule forbids endless tuning. The family is stopped.

## Current mechanism gap

The UVR contrastive representation is real, but **naively adding a representation-preservation regularizer during supervised fine-tuning is not viable**: the auxiliary geometry gradient is not calibrated to the regression scale. Any future mechanism must either:

- operate at a stage where the geometry can be used without back-propagating a global auxiliary gradient into the encoder; or
- provide a rigorously normalized/constrained update with a pre-registered compute control.

No such mechanism has yet passed the matched screen.
