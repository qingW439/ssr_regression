# M1 Baseline Failure Atlas

Scope: EmoBank lb500, seed0 five-fold OOF, B0 Raw supervised and B2 UVR supervised.  
n = 500 holdout samples (100 per fold). All MAE values are on the 1–5 scale (`normalized MAE * 4`).  
Predictions are `raw_*`; no DEV/TEST used.

## 1. Macro and per-dimension performance

| Arm | Dim | MAE | Pearson | CCC | R² | Bias | SD ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| B0 Raw | V | 0.1933 | 0.6999 | 0.5862 | 0.4629 | −0.0187 | 0.544 |
| B0 Raw | A | 0.1768 | 0.4401 | 0.3274 | 0.1882 | −0.0191 | 0.449 |
| B0 Raw | D | 0.1542 | 0.3244 | 0.2347 | 0.0931 | −0.0077 | 0.429 |
| B0 Raw | Macro | **0.17476** | **0.4881** | **0.3828** | **0.2480** | −0.0152 | 0.474 |
| B2 UVR | V | 0.1864 | 0.7367 | 0.6402 | 0.5174 | −0.0150 | 0.583 |
| B2 UVR | A | 0.1702 | 0.4906 | 0.3692 | 0.2368 | −0.0131 | 0.455 |
| B2 UVR | D | 0.1463 | 0.4032 | 0.3036 | 0.1584 | −0.0085 | 0.455 |
| B2 UVR | Macro | **0.16762** | **0.5435** | **0.4377** | **0.3042** | −0.0122 | 0.498 |

**Observed failure:** D has the lowest Pearson (0.403) and CCC (0.304); A is second (0.491). All dimensions have SD ratio ≈ 0.45–0.58, i.e. predictions are strongly variance-compressed.

## 2. Error by true-label region

Bins are quintiles of the true label (duplicates collapsed). B2 values:

| Dim | Bin | n | mean y | MAE | Pearson | Bias | SD ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| V | low | 102 | 0.363 | **0.3119** | 0.411 | **+0.288** | 0.753 |
| V | mid-low | 208 | 0.486 | 0.0961 | 0.415 | −0.002 | 1.953 |
| V | mid | 93 | 0.533 | 0.1183 | 0.227 | −0.078 | 3.049 |
| V | high | 97 | 0.619 | **0.3130** | 0.589 | **−0.301** | 0.777 |
| A | low | 108 | 0.434 | **0.2577** | 0.105 | **+0.253** | 0.734 |
| A | high | 84 | 0.617 | **0.3169** | 0.236 | **−0.313** | 0.681 |
| D | low | 118 | 0.449 | **0.2120** | 0.158 | **+0.207** | 0.722 |
| D | high | 88 | 0.595 | **0.2579** | 0.240 | **−0.255** | 0.638 |

**Observed failure:** low true labels are systematically over-predicted (positive bias), high true labels under-predicted (negative bias). This is the classic conditional-mean compression of direct L1 regression. It is stronger than the macro bias suggests and is worst in the tails.

## 3. Text length and lexical groups

| Group | n | B2 V MAE | B2 A MAE | B2 D MAE |
|---|---:|---:|---:|---:|
| chars 0–50 | 192 | 0.218 | — | — |
| chars 50–100 | 130 | 0.180 | — | — |
| chars 100–200 | 142 | 0.168 | — | — |
| chars ≥200 | 36 | 0.113 | — | — |
| negation keywords | 63 | 0.174 | 0.178 | 0.134 |
| contrast keywords | 42 | 0.218 | 0.178 | 0.143 |
| intensifier keywords | 28 | 0.238 | 0.201 | 0.133 |

The keyword groups are **heuristic regex groups**, not validated semantic labels. They are only used to generate hypotheses, not as causal evidence. Short texts and contrast/intensifier texts have elevated V error.

## 4. UVR vs Raw sample-level effect

Paired per-sample ΔMAE (UVR − Raw; negative means UVR better):

- mean **−0.00714**; 95% bootstrap CI **[−0.01047, −0.00390]**
- median −0.00429
- UVR improved **280/500**, worsened **220/500**

UVR helps on average but is not uniformly better. Examples where UVR improved:

| id | true V | Raw V | UVR V | Δ |
|---|---:|---:|---:|---:|
| The_Black_Willow_1491_1616 | 0.700 | 0.495 | 0.558 | −0.150 |
| hotel-california_7024_7062 | 0.475 | 0.416 | 0.456 | −0.140 |
| SemEval_629 | 0.200 | 0.464 | 0.384 | −0.138 |
| SemEval_789 | 0.178 | 0.413 | 0.367 | −0.128 |

Examples where UVR worsened:

| id | true V | Raw V | UVR V | Δ |
|---|---:|---:|---:|---:|
| SemEval_613 | 0.450 | 0.406 | 0.352 | +0.107 |
| The_Black_Willow_16366_16371 | 0.475 | 0.506 | 0.576 | +0.101 |
| The_Black_Willow_5628_5704 | 0.343 | 0.421 | 0.462 | +0.085 |
| SemEval_1105 | 0.500 | 0.506 | 0.414 | +0.082 |

Texts:

- Improved: “They gazed at each other, lost in the rapture of love …”; “I did a lot of staring and not moving.”; “Participant in water-drinking contest dies.”
- Worsened: “Rowdy fans ejected from the Open”; “So,”; “Arthur's face assumed a long-suffering aspect of authoritative exasperation.”

The failure cases mix **short/elliptical text**, **event headlines without explicit affect**, and **figurative/idiomatic descriptions**. The regex groups do not identify these reliably.

## 5. Interpretation limits

- Seed0 only; UVR seed1/2 OOF predictions are not stored under the same directory, so the atlas is not a three-seed claim.
- Bins have small n and the labels are discrete (0.5 steps); SD ratios inside narrow bins are unstable.
- Keyword categories are not validated semantic annotations.
- These results establish **where** the model fails (tails, D, short/ambiguous text), not yet **why** unlabeled data should fix it.
