#!/bin/bash
set -u
REPO=/root/autodl-tmp/rankup_project/ssr_regression
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
export SSR_ASSETS=/root/autodl-tmp/rankup_project
RUNS=$SSR_ASSETS/m1_hdt_runs
LOG=$RUNS/ths_formal.log
{
for s in 0 1 2; do for f in 0 1 2 3 4; do
  out=$RUNS/ths_s${s}_f${f}
  if [ -f "$out/readout.pth" ]; then echo "SKIP s$s f$f"; continue; fi
  echo "--- $(date) THS s$s f$f"
  "$PY" "$REPO/src/baselines/train_uul.py" --arm raw --seed "$s" --fold "$f" \
    --targets "$SSR_ASSETS/m1_hdt_gate/targets_th/B_s${s}_f${f}.npz" --out-dir "$out" --lambda-u 0.25 \
    > "$RUNS/ths_s${s}_f${f}.log" 2>&1 || { echo "FAIL s$s f$f"; continue; }
  echo "DONE s$s f$f $(date)"
done; done
echo "THS FORMAL DONE $(date)"
} >> "$LOG" 2>&1
