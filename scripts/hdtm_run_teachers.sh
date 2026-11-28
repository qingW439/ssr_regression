#!/usr/bin/env bash
set -euo pipefail
SEED=${1:?seed}; FOLD=${2:?fold}
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-/root/autodl-tmp/rankup_project/envs/rankup/bin/python}
export SSR_ASSETS=${SSR_ASSETS:-/root/autodl-tmp/rankup_project}
OUT=$SSR_ASSETS/m1_hdt_gate/teachers; mkdir -p "$OUT"
for j in 0 1 2 3; do
  f="$OUT/B_s${SEED}_f${FOLD}_j${j}.npz"
  if [ -f "$f" ] && "$PY" "$REPO/tools/check_teacher_cache.py" "$SEED" "$FOLD" --teacher "$j" >/dev/null 2>&1; then echo "SKIP j$j"; continue; fi
  "$PY" "$REPO/src/hdtm/train_teacher_hist.py" --seed "$SEED" --fold "$FOLD" --j "$j" --out "$f" \
    > "$OUT/B_s${SEED}_f${FOLD}_j${j}.log" 2>&1 || { echo "TEACHER j$j FAILED"; exit 2; }
done
"$PY" "$REPO/tools/check_teacher_cache.py" "$SEED" "$FOLD"
echo "teachers done s${SEED}f${FOLD}"
