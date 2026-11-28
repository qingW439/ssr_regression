#!/bin/bash
set -u
REPO=/root/autodl-tmp/rankup_project/ssr_regression
PY=/root/autodl-tmp/rankup_project/envs/rankup/bin/python
export SSR_ASSETS=/root/autodl-tmp/rankup_project
RUNS=$SSR_ASSETS/m1_hdt_runs
LOG=$RUNS/acl_dev.log
LAM=1.4305982197781755
{
for unit in "0 1" "1 2" "2 3"; do
  set -- $unit
  out=$RUNS/acl_s${1}_f${2}
  if [ -f "$out/readout.pth" ]; then echo "SKIP $unit"; continue; fi
  echo "--- $(date) ACL s$1 f$2"
  "$PY" "$REPO/src/hdtm/train_hdt.py" --arm acl --seed "$1" --fold "$2" --lam "$LAM" \
    --fused "$SSR_ASSETS/m1_hdt_gate/fused/B_s${1}_f${2}.npz" --out-dir "$out" \
    > "$RUNS/acl_s${1}_f${2}.log" 2>&1 || { echo "FAIL $unit"; continue; }
  echo "DONE $unit $(date)"
done
echo "ACL DEV DONE $(date)"
} >> "$LOG" 2>&1
