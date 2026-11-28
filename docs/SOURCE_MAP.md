# SOURCE_MAP — model → source → origin

All model code now lives in `wq-main` (`src/`). Server assets (data, weights, targets) stay outside git
under `$SSR_ASSETS` (default `/root/autodl-tmp/rankup_project`).

| model | entrypoint | loss/impl | origin (pre-consolidation path) | status |
| --- | --- | --- | --- | --- |
| S (strong supervised) | `src/baselines/train_uul.py --arm off` (scalar head, no UUL) | same file | `m1_v2/train_m1.py` | code restored, metrics from `M1_BASELINE_V2_3SEED.csv` |
| U0 | `src/baselines/train_uul.py --arm off` | same | `m1_v2/train_m1.py` | code + 15 readouts + OOF |
| U1 | `src/baselines/train_uul.py --arm raw --lambda-u 0.25` | same | `m1_v2/train_m1.py` | code + 15 readouts + OOF |
| T-HS | `src/baselines/train_uul.py --arm raw --targets targets_th/...` | same | `m1_v2/train_m1.py` | code + 3 readouts + OOF |
| H-S | `src/baselines/train_hist.py --arm hs` | `src/hdtm/hist_loss.py` | `m1_hist_gate/{train_hist,hist_loss}.py` | code + 15 readouts + OOF |
| H-U | `src/baselines/train_hist.py --arm hu` | same | same | code + 15 readouts + OOF |
| C-S / C-U | `src/baselines/train_hist.py --arm cs/cu` | same | same | code + 3+3 readouts + OOF |
| HPL | `src/baselines/train_hpl.py` | same file | `m1_hpl_gate/train_hpl.py` | code + 15 readouts + OOF |
| RCUS | `src/baselines/train_rcus.py` | `src/baselines/rcus_loss.py` | `m1_rcus_gate/{train_rcus2,rcus_loss}.py` | code + 15 readouts + OOF |
| AAN | `src/baselines/train_aan.py` | same file | `m1_aan_gate/train_aan.py` | code + 15 OOF, **weights deleted** |
| SUP-6 | `src/baselines/train_shallow.py --arm sup6` | same | `m1_shallow_gate/train_shallow.py` | code + 15 OOF, **weights deleted** |
| UUL-6 (6-layer semi-supervised, efficiency study) | `src/baselines/train_shallow.py --arm uul6` | same | same | code + 15 OOF, **weights deleted** |
| HDT-D | `src/hdtm/train_hdt.py --arm hdtd` | `src/hdtm/hist_loss.py` | `m1_hdt_gate/train_hdt.py` | code + 3 readouts + OOF |
| HDT-M | `src/hdtm/train_hdt.py --arm hdtm` | same | same | code + 15 readouts + OOF |
| distribution teacher | `src/hdtm/train_teacher_hist.py` | same | `m1_hdt_gate/train_teacher_hist.py` | code + 60 caches |
| teacher fusion | `src/hdtm/make_fused_fold.py` | — | `m1_hdt_gate/make_fused_fold.py` | code + 15 fused targets |
| RankUp-3D v2 / Full v2 | `src/baselines/train_rankup.py` (arms b1/b2/b3/b4) | same | `Semi-Supervised-Regression-rankup-m1/scripts/vad_research/train_rankup_official.py` | code restored; results reported (different protocol) |

Shared dependencies: `semilearn/datasets/nlp_datasets/emobank.py` (GoldData/TextCollator/read_gold),
`scripts/vad_research/train_roberta_vad_supervised.py` (RobertaVADRegressor/set_seed),
`scripts/vad_research/train_acvr.py` (mask_tokens) — all inside the repo.

Removed/legacy entries in `m1_*_gate` are historical copies; `scripts/run_baseline.sh` and
`scripts/hdtm_*.sh` call **only** `src/`. Do not import from the old directories.
