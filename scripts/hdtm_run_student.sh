#!/usr/bin/env bash
set -u
ARM=${1:?arm}; SEED=${2:?seed}; FOLD=${3:?fold}
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-/root/autodl-tmp/rankup_project/envs/rankup/bin/python}
export SSR_ASSETS=${SSR_ASSETS:-/root/autodl-tmp/rankup_project}
OUT=$SSR_ASSETS/m1_runs_unified/${ARM}_s${SEED}_f${FOLD}
mkdir -p "$SSR_ASSETS/m1_runs_unified"
"$PY" "$REPO/src/hdtm/train_hdt.py" --arm "$ARM" --seed "$SEED" --fold "$FOLD" \
  --fused "$SSR_ASSETS/m1_hdt_gate/fused/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" || exit 2
"$PY" "$REPO/tools/check_status.py" --runs "$OUT" --recompute --teacher-guard | tail -3
