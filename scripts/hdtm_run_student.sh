#!/usr/bin/env bash
set -euo pipefail
ARM=${1:?arm}; SEED=${2:?seed}; FOLD=${3:?fold}
[ "$ARM" = hdtm ] || { echo "hdtm_run_student.sh only accepts hdtm" >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-/root/autodl-tmp/rankup_project/envs/rankup/bin/python}
export SSR_ASSETS=${SSR_ASSETS:-/root/autodl-tmp/rankup_project}
VERSION=${SSR_RUN_VERSION:-v0.1.2-hdtm-ready}
OUT=$SSR_ASSETS/m1_runs_unified/$VERSION/HDT-M_s${SEED}_f${FOLD}
mkdir -p "$(dirname "$OUT")"
"$PY" "$REPO/tools/check_teacher_cache.py" "$SEED" "$FOLD" --fused
"$PY" "$REPO/src/hdtm/train_hdt.py" --arm "$ARM" --seed "$SEED" --fold "$FOLD" \
  --fused "$SSR_ASSETS/m1_hdt_gate/fused/B_s${SEED}_f${FOLD}.npz" --out-dir "$OUT" || exit 2
"$PY" "$REPO/tools/check_status.py" --experiment HDT-M --version "$VERSION" --seed "$SEED" --fold "$FOLD" --recompute
