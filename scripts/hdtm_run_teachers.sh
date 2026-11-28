#!/bin/bash
set -u
SEED=${1:?seed}; FOLD=${2:?fold}
ROOT=/root/autodl-tmp/rankup_project; PY=$ROOT/envs/rankup/bin/python
OUT=$ROOT/m1_hdt_gate/teachers
mkdir -p "$OUT"
for j in 0 1 2 3; do
  f="$OUT/B_s${SEED}_f${FOLD}_j${j}.npz"
  if [ -f "$f" ]; then echo "SKIP j$j"; continue; fi
  echo "=== $(date) teacher j$j ==="
  "$PY" "$ROOT/m1_hdt_gate/train_teacher_hist.py" --seed "$SEED" --fold "$FOLD" --j "$j" --out "$f" \
    > "$OUT/B_s${SEED}_f${FOLD}_j${j}.log" 2>&1 || echo "TEACHER j$j FAILED"
done
echo "=== $(date) teachers done ==="
