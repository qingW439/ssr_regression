#!/bin/bash
set -u
SEED=${1:?seed}; FOLD=${2:?fold}
ROOT=/root/autodl-tmp/rankup_project; PY=$ROOT/envs/rankup/bin/python
RUNS=$ROOT/m1_hdt_runs; OUT=$RUNS/ths_s${SEED}_f${FOLD}; LOG=$RUNS/ths_s${SEED}_f${FOLD}.log
mkdir -p "$RUNS"
echo "=== $(date) T-HS s$SEED f$FOLD" | tee -a "$LOG"
if [ ! -f "$OUT/readout.pth" ]; then
  "$PY" "$ROOT/m1_v2/train_m1.py" --arm raw --seed "$SEED" --fold "$FOLD" \
    --targets "$ROOT/m1_hdt_gate/targets_th/B_s${SEED}_f${FOLD}.npz" \
    --out-dir "$OUT" --lambda-u 0.25 >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$OUT/readout.pth" ]; then echo "FAILED rc=$rc" | tee -a "$LOG"; exit 2; fi
fi
echo "=== $(date) T-HS s$SEED f$FOLD DONE" | tee -a "$LOG"; tail -2 "$LOG"
