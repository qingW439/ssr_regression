#!/bin/bash
# Unified baseline entrypoint (wq-main). All code comes from src/, assets from $SSR_ASSETS.
# Usage: bash scripts/run_baseline.sh <arm> <seed> <fold>
# arms: U0 | U1 | THS | H-S | H-U | C-S | C-U | HPL | RCUS | AAN | SUP6 | UUL6 | HDT-D | HDT-M
set -euo pipefail
ARM=${1:?arm}; SEED=${2:?seed}; FOLD=${3:?fold}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/root/autodl-tmp/rankup_project/envs/rankup/bin/python}
export SSR_ASSETS=${SSR_ASSETS:-/root/autodl-tmp/rankup_project}
A=$SSR_ASSETS
VERSION=${SSR_RUN_VERSION:-v0.1.2-hdtm-ready}
RUNS=$A/m1_runs_unified/$VERSION; mkdir -p "$RUNS"
MODEL=$ARM
if [ "$ARM" = THS ]; then MODEL=T-HS; fi
OUT=$RUNS/${MODEL}_s${SEED}_f${FOLD}
LOG=$RUNS/${MODEL}_s${SEED}_f${FOLD}.log
echo "=== $(date) $ARM s$SEED f$FOLD" | tee -a "$LOG"
if [ "$ARM" = HDT-M ] || [ "$ARM" = HDT-D ]; then
  "$PY" "$REPO/tools/check_teacher_cache.py" "$SEED" "$FOLD" --fused
fi

case "$ARM" in
  U0)   "$PY" "$REPO/src/baselines/train_uul.py" --arm off --seed "$SEED" --fold "$FOLD" \
          --targets "$A/m1_v2_assets/targets/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" ;;
  U1)   "$PY" "$REPO/src/baselines/train_uul.py" --arm raw --seed "$SEED" --fold "$FOLD" \
          --targets "$A/m1_v2_assets/targets/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" --lambda-u 0.25 ;;
  THS)  "$PY" "$REPO/src/baselines/train_uul.py" --arm raw --seed "$SEED" --fold "$FOLD" \
          --targets "$A/m1_hdt_gate/targets_th/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" --lambda-u 0.25 ;;
  H-S)  "$PY" "$REPO/src/baselines/train_hist.py" --arm hs --seed "$SEED" --fold "$FOLD" --out-dir "$OUT" ;;
  H-U)  "$PY" "$REPO/src/baselines/train_hist.py" --arm hu --seed "$SEED" --fold "$FOLD" \
          --targets "$A/m1_v2_assets/targets/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" ;;
  C-S)  "$PY" "$REPO/src/baselines/train_hist.py" --arm cs --seed "$SEED" --fold "$FOLD" --out-dir "$OUT" ;;
  C-U)  "$PY" "$REPO/src/baselines/train_hist.py" --arm cu --seed "$SEED" --fold "$FOLD" \
          --targets "$A/m1_v2_assets/targets/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" ;;
  HPL)  "$PY" "$REPO/src/baselines/train_hpl.py" --seed "$SEED" --fold "$FOLD" --out-dir "$OUT" ;;
  RCUS) "$PY" "$REPO/src/baselines/train_rcus.py" --seed "$SEED" --fold "$FOLD" --out-dir "$OUT" ;;
  AAN)  "$PY" "$REPO/src/baselines/train_aan.py" --seed "$SEED" --fold "$FOLD" --out-dir "$OUT" ;;
  SUP6) "$PY" "$REPO/src/baselines/train_shallow.py" --arm sup6 --seed "$SEED" --fold "$FOLD" --out-dir "$OUT" ;;
  UUL6) "$PY" "$REPO/src/baselines/train_shallow.py" --arm uul6 --seed "$SEED" --fold "$FOLD" \
          --targets "$A/m1_v2_assets/targets/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" ;;
  HDT-D) "$PY" "$REPO/src/hdtm/train_hdt.py" --arm hdtd --seed "$SEED" --fold "$FOLD" \
          --fused "$A/m1_hdt_gate/fused/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" ;;
  HDT-M) "$PY" "$REPO/src/hdtm/train_hdt.py" --arm hdtm --seed "$SEED" --fold "$FOLD" \
          --fused "$A/m1_hdt_gate/fused/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
rc=$?
if [ $rc -ne 0 ]; then echo "FAILED rc=$rc" | tee -a "$LOG"; exit 2; fi
"$PY" "$REPO/tools/check_status.py" --experiment "$MODEL" --version "$VERSION" --seed "$SEED" --fold "$FOLD" --recompute
echo "=== $(date) $ARM s$SEED f$FOLD DONE" | tee -a "$LOG"
