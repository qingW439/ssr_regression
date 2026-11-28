# MODEL_ARCHITECTURE — HDT-M (Histogram Distribution Transfer, M1 candidate)

Status: formal 15-fold complete (2026-11-27). MAE 0.173036, CCC 0.465931.

## 1. Pipeline (actual source, verified on the server)

```
text ──► RoBERTa-base (UVR-SimCSE init)  ──► masked-mean pooling ──► HistHead (768→3×20)
                                                                        │
                       ┌────────────────────────────────────────────────┤
                       │                                                │
             labelled VAD y (400)                          unlabelled text (6657)
                       │                                                │
             gaussian_target(y)                          fused teacher distribution Q
                       │                                     (4 teachers, frozen)
             CE  L_sup_CE                                        │
                       │                              decode_corrected(Q) = pseudo VAD
                       │                                                │
                       │                                     gaussian_target(pseudo)
                       │                                                │
                       │                                     CE  L_ulb_CE  (weight 0.25)
                       └───────────────────► L = L_sup_CE + 0.25·L_ulb_CE ◄──┘
                                                     │
                                          96-step supervised readout (CE)
                                                     │
                                     decode_corrected → V, A, D ∈ [0,1]
```

Inference runs **one student forward**; teachers are offline only.

## 2. Modules and source locations

| Module | File | Function / class | Shape |
| --- | --- | --- | --- |
| Encoder | `src/hdtm/train_hdt.py` | `HistVAD.__init__` (RobertaVADRegressor) | ids[B,128] → hidden[B,128,768] |
| Pooling | `src/hdtm/train_hdt.py` | `HistVAD.forward` | → [B,768] |
| Histogram head | `src/hdtm/hist_loss.py` | `HistHead` | [B,768] → [B,3,20] |
| Gaussian target | `src/hdtm/hist_loss.py` | `gaussian_target` | y[B,3] → q[B,3,20], σ=1/20 |
| Loss | `src/hdtm/hist_loss.py` | `hist_loss` | CE over 20 bins |
| Decoder | `src/hdtm/hist_loss.py` | `decode_corrected` (+`_lut`) | logits → [B,3] |
| Teacher | `src/hdtm/train_teacher_hist.py` | — | 3200 steps, CE, 96-step readout |
| Teacher cache | `src/hdtm/make_fused_fold.py` | — | 4×probs → Q, decode → pseudo VAD |
| Student | `src/hdtm/train_hdt.py` | `--arm hdtm` | 3200 steps + 96 readout |
| Evaluation | `src/hdtm/evaluate.py` | `met` | MAE/CCC/Pearson/SRCC/R²/tail/sd |

## 3. Differences vs baselines

| vs | change |
| --- | --- |
| RankUp-3D / U1 | scalar head `[B,3]` + L1 → histogram head `[B,3,20]` + CE + corrected decoding |
| H-S | same student, but H-S has no unlabelled term |
| H-U | same student, but U1's **scalar** teachers → **distribution-regression** teachers |

## 4. Hyperparameters (verified against the actual runs)

3200 steps, batch 16, lr 1e-5, weight decay 5e-4, layer decay 0.75, 96-step readout (lr 5e-6, wd 0.01), λ_ulb = 0.25, 500-label EmoBank, 3 seeds × 5 folds, 400 train / 100 holdout per fold.

## 5. Boundary correction

`decode_corrected` uses a 401-point inverse LUT built only from the target construction
(`gaussian_target` + boundary reflection). Target-recovery bias over the full range is 2e-6,
the map is monotone, output stays in [0,1], and gradients flow to the logits (verified in
`tests/test_decoder.py`).

## 6. Known limitations

- The corrected decoder is a fixed, target-derived map: it removes construction bias, not model bias.
- Teacher-to-student "representation matching" was tested and **rejected** (2×2 interaction −0.0023).
- CCC / tail remain slightly below the supervised H-S (ΔCCC −0.0043, Δtail +0.0022).
