# UVR Gain Attribution

Stage: **base 1600**, three seeds × five folds. Readout-stage numbers are not mixed in.

## Per-seed matched comparison

| Seed | Arm | MAE | Pearson | CCC | R² | Bias | SD ratio |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | Raw | 0.17346 | 0.5052 | 0.3771 | 0.2613 | −0.0031 | 0.438 |
| 0 | UVR | **0.16922** | **0.5310** | **0.4077** | **0.2845** | −0.0113 | 0.460 |
| 1 | Raw | 0.17690 | 0.4590 | 0.3446 | 0.2291 | −0.0010 | 0.431 |
| 1 | UVR | **0.17297** | **0.4800** | **0.3676** | **0.2494** | +0.0022 | 0.444 |
| 2 | Raw | 0.17888 | 0.4672 | 0.3335 | 0.2256 | +0.0014 | 0.408 |
| 2 | UVR | **0.17402** | **0.5161** | **0.3858** | **0.2726** | +0.0038 | 0.439 |

UVR improves MAE by **0.00424, 0.00393, 0.00486** and Pearson by **0.0258, 0.0210, 0.0489** across the three seeds. The direction is consistent.

## What changes

- UVR does **not** mainly fix variance compression: SD ratio moves only 0.438→0.460, 0.431→0.444, 0.408→0.439.
- The improvement is primarily a **ranking/representation gain**: Pearson and CCC rise more than the SD ratio.
- Bias remains small and seed-dependent; no consistent global shift correction is the source of the gain.

## Seed0 per-dimension and sample evidence

Seed0 base/readout atlas:

- V: MAE 0.1933→0.1864, Pearson 0.6999→0.7367;
- A: 0.1768→0.1702, Pearson 0.4401→0.4906;
- D: 0.1542→0.1463, Pearson 0.3244→0.4032.

All three improve, but D remains the weakest. Tail bins still show strong compression:

- low-V bias +0.288, high-V bias −0.301;
- low-D bias +0.207, high-D bias −0.255.

Per-sample UVR−Raw ΔMAE: mean −0.00714, 95% CI [−0.01047, −0.00390], 280 improved / 220 worsened. The gain is real but heterogeneous.

## Sample-type conclusion

No semantic category is verified. The heuristic groups (short text, contrast words, intensifiers) are confounded with length and label range. The only repeatable statements are:

1. UVR helps all three dimensions on average;
2. D remains the lowest-Pearson dimension;
3. the tail bins remain compressed;
4. gains are not uniform per sample.

No text-case-driven mechanism is proposed from individual examples.
