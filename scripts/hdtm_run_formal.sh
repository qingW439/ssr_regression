#!/usr/bin/env bash
# 15-fold HDT-M: teachers -> fused targets -> student -> completion check. Resume-safe.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-/root/autodl-tmp/rankup_project/envs/rankup/bin/python}
export SSR_ASSETS=${SSR_ASSETS:-/root/autodl-tmp/rankup_project}
LOG=$SSR_ASSETS/m1_hdt_runs/formal_unified.log; mkdir -p "$SSR_ASSETS/m1_hdt_runs"
{
for s in 0 1 2; do for f in 0 1 2 3 4; do
  out=$SSR_ASSETS/m1_runs_unified/hdtm_s${s}_f${f}
  if "$PY" "$REPO/tools/check_status.py" --runs "$out" --teacher-guard >/dev/null 2>&1; then
    echo "SKIP s$s f$f"; continue; fi
  bash "$REPO/scripts/hdtm_run_teachers.sh" "$s" "$f" || { echo "FAIL teachers s$s f$f"; continue; }
  [ -f "$SSR_ASSETS/m1_hdt_gate/fused/B_s${s}_f${f}.npz" ] || \
    "$PY" "$REPO/src/hdtm/make_fused_fold.py" "$s" "$f" || { echo "FAIL fuse s$s f$f"; continue; }
  bash "$REPO/scripts/hdtm_run_student.sh" hdtm "$s" "$f" || { echo "FAIL student s$s f$f"; continue; }
  echo "DONE s$s f$f $(date)"
done; done
echo "FORMAL DONE $(date)"
} >> "$LOG" 2>&1
