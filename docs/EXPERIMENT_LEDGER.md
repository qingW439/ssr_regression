# EXPERIMENT_LEDGER — formal results (EmoBank lb500, 3 seeds × 5 folds = 15 units)

All numbers recomputed from OOF parquet with one metric function (`src/hdtm/evaluate.py`).
OOF = out-of-fold predictions, i.e. predictions on samples that did not take part in that fold's training.
MAE ×4 so the scale is 1..5.

## Formal 15-fold (2026-11-27)

| Model | MAE | CCC | Pearson | SRCC | R² | tail MAE | mid MAE | sd_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0 (scalar supervised) | 0.180315 | 0.416833 | 0.475304 | 0.446249 | 0.209075 | 0.279518 | 0.123519 | 0.595 |
| U1 (scalar semi-sup.) | 0.178130 | 0.414510 | 0.482969 | 0.452955 | 0.221069 | 0.281952 | 0.118626 | 0.570 |
| HPL | 0.177314 | 0.433869 | 0.492856 | 0.464977 | 0.230410 | 0.274206 | 0.121871 | 0.599 |
| RCUS | 0.196126 | 0.494020 | 0.509413 | 0.492391 | 0.083731 | 0.247850 | 0.166844 | 0.888 |
| H-S (histogram supervised) | 0.175208 | 0.470203 | 0.511561 | 0.480010 | 0.234729 | 0.262770 | 0.125129 | 0.671 |
| H-U (scalar teachers) | 0.174910 | 0.429681 | 0.502574 | 0.468301 | 0.246289 | 0.278036 | 0.115824 | 0.566 |
| **HDT-M** | **0.173036** | 0.465931 | **0.518222** | 0.482137 | **0.257238** | 0.264927 | 0.120458 | 0.632 |

Per-seed MAE (HDT-M vs strongest baselines):

| Model | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- |
| HDT-M | **0.172927** | **0.173579** | **0.172603** |
| H-S | 0.175515 | 0.174955 | 0.175155 |
| H-U | 0.174034 | 0.175099 | 0.175598 |
| HPL | 0.175903 | 0.176782 | 0.179256 |

Paired (15 folds, same seed/fold):

| comparison | ΔMAE (se) | ΔCCC | Δtail | W:L |
| --- | --- | --- | --- | --- |
| HDT-M − U0 | −0.007278 (0.002229) | +0.049098 | −0.014591 | 11:4 |
| HDT-M − U1 | −0.005094 (0.001851) | +0.051421 | −0.017025 | 11:4 |
| HDT-M − HPL | −0.004278 (0.001689) | +0.032062 | −0.009279 | 11:4 |
| HDT-M − RCUS | −0.023090 (0.002684) | −0.028089 | +0.017076 | 15:0 |
| HDT-M − H-S | −0.002172 (0.000543) | −0.004272 | +0.002157 | 12:3 |
| HDT-M − H-U | −0.001874 (0.000649) | +0.036250 | −0.013109 | 13:2 |

Caveat: 15 folds share data across seeds and are not independent samples; the standard errors are
descriptive, not a formal significance test.

## Distribution teachers (12 cross-fit teachers, 3 dev units × 4)

| metric | scalar teachers | distribution teachers | wins |
| --- | --- | --- | --- |
| MAE | 0.182044 | **0.175681** | 11/12 |
| CCC | 0.405889 | **0.464975** | 12/12 |
| SRCC | 0.438903 | 0.481606 | — |
| sd_ratio | 0.567980 | 0.633456 | 12/12 |
| bias | +0.007238 | −0.000565 | — |

Teacher → student transfer (15 units): Pearson(teacher holdout MAE, student MAE) = **−0.636** (p=0.011).

## 2×2 teacher × student (3 dev units)

| teacher | scalar student | distribution student |
| --- | --- | --- |
| scalar | U1 0.177670 | H-U 0.174287 |
| distribution | T-HS **0.173546** | HDT-M 0.172451 |

Teacher gain: scalar student 0.004125, distribution student 0.001836 → **interaction −0.002288**
(per-unit mean; the earlier −0.000888 came from a wrong T-HS mean and is retired).

## Development-only results (3 units, not formal)

| model | MAE | CCC | tail |
| --- | --- | --- | --- |
| H-S | 0.174857 | 0.450111 | 0.253733 |
| H-U | 0.174287 | 0.412706 | 0.269947 |
| T-HS | 0.173546 | 0.422919 | 0.265437 |
| HDT-M | 0.172451 | 0.444131 | 0.256952 |
| HDT-D | 0.173016 | 0.445523 | 0.256552 |
| C-S (60-dim head + L1) | 0.174036 | 0.418093 | 0.269815 |
| C-U | 0.176501 | 0.396894 | 0.275481 |

## Efficiency / other controls

- UUL-6 (6-layer encoder, semi-supervised): 0.180128 MAE / 0.411744 CCC (15 folds) — depth/efficiency study.
- SUP-6: 0.187393 / 0.397537.


## Additional models (restored 2026-11-28)

| model | MAE | CCC | units | note |
| --- | --- | --- | --- | --- |
| AAN (3-D adaptation) | 0.188113 | 0.482730 | 15 formal | 15 readout weights restored and verified 2026-11-28 |
| S_v1 (strong supervised scalar) | 0.180800 | — | 3 seed summaries | `M1_BASELINE_V2_3SEED.csv` mean row; distinct protocol |
| SUP-6 (6-layer, supervised) | 0.187393 | 0.397537 | 15 formal | 15 readout weights restored and verified 2026-11-28 |
| UUL-6 (6-layer, semi-supervised) | 0.180128 | 0.411744 | 15 formal | 15 readout weights restored and verified 2026-11-28 |

**UUL-6 naming**: UUL-6 is a **6-layer encoder** semi-supervised model (depth/efficiency study),
not "six unlabelled batches per step". The earlier description was wrong and is retired.

## Different-protocol results (not mixed into the 15-fold table)

| source | metric | value | protocol |
| --- | --- | --- | --- |
| RankUp-3D v2 | MAE | 0.186333 | 3 seeds × 5 folds; recomputed from `off_b3v2_*/sup96/oof_predictions.parquet`, fold IDs verified |
| RankUp Full v2 (RDA) | MAE | 0.185048 | 3 seeds × 5 folds; recomputed from `off_b3rdav2_*/sup96/oof_predictions.parquet`, same aggregation |
| RankUp "raw" | MAE / CCC | 0.181740 / 0.426728 | nested OOF on 500 samples (`outputs/oof/oof_metrics.json`) |

## Teacher → student correlation (correct phrasing)

Pearson(teacher_holdout_MAE, student_MAE) = **−0.636** across the 15 outer units. Algebraically,
higher teacher error is associated with lower student error in this sample. This observational
correlation does not establish that worsening a teacher helps the student; fold difficulty,
selection effects and other confounders remain possible. The earlier same-direction interpretation
was incorrect.
