#!/bin/bash
set -u
ARM=${1:?arm}; SEED=${2:?seed}; FOLD=${3:?fold}
ROOT=/root/autodl-tmp/rankup_project; PY=$ROOT/envs/rankup/bin/python
RUNS=$ROOT/m1_hdt_runs; OUT=$RUNS/${ARM}_s${SEED}_f${FOLD}; LOG=$RUNS/${ARM}_s${SEED}_f${FOLD}.log
mkdir -p "$RUNS"
echo "=== $(date) $ARM s$SEED f$FOLD" | tee -a "$LOG"
if [ ! -f "$OUT/readout.pth" ]; then
  "$PY" "$ROOT/m1_hdt_gate/train_hdt.py" --arm "$ARM" --seed "$SEED" --fold "$FOLD" \
    --fused "$ROOT/m1_hdt_gate/fused/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$OUT/readout.pth" ]; then echo "FAILED rc=$rc" | tee -a "$LOG"; exit 2; fi
fi
echo "=== $(date) $ARM s$SEED f$FOLD DONE" | tee -a "$LOG"; tail -2 "$LOG"
