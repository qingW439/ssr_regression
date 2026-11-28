#!/usr/bin/env bash
# 15-fold HDT-M: teachers -> fused targets -> student -> completion check. Resume-safe.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-/root/autodl-tmp/rankup_project/envs/rankup/bin/python}
export SSR_ASSETS=${SSR_ASSETS:-/root/autodl-tmp/rankup_project}
VERSION=${SSR_RUN_VERSION:-v0.1.2-hdtm-ready}
LOG=$SSR_ASSETS/m1_runs_unified/$VERSION/formal.log; mkdir -p "$(dirname "$LOG")"
{
failed=0
for s in 0 1 2; do for f in 0 1 2 3 4; do
  if "$PY" "$REPO/tools/check_status.py" --experiment HDT-M --version "$VERSION" --seed "$s" --fold "$f" >/dev/null 2>&1; then
    echo "SKIP s$s f$f"; continue; fi
  bash "$REPO/scripts/hdtm_run_teachers.sh" "$s" "$f" || { echo "FAIL teachers s$s f$f"; failed=1; continue; }
  if ! "$PY" "$REPO/tools/check_teacher_cache.py" "$s" "$f" --fused >/dev/null 2>&1; then
    "$PY" "$REPO/src/hdtm/make_fused_fold.py" "$s" "$f" || { echo "FAIL fuse s$s f$f"; failed=1; continue; }
  fi
  "$PY" "$REPO/tools/check_teacher_cache.py" "$s" "$f" --fused || { echo "FAIL fused guard s$s f$f"; failed=1; continue; }
  bash "$REPO/scripts/hdtm_run_student.sh" hdtm "$s" "$f" || { echo "FAIL student s$s f$f"; failed=1; continue; }
  echo "DONE s$s f$f $(date)"
done; done
"$PY" "$REPO/tools/check_status.py" --experiment HDT-M --version "$VERSION" --recompute || failed=1
echo "FORMAL END failed=$failed $(date)"
exit "$failed"
} >> "$LOG" 2>&1
