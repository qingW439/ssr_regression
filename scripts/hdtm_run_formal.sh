#!/bin/bash
# Formal 15-fold HDT-M: 4 distribution teachers per outer unit, fused targets, HDT-M student.
set -u
ROOT=/root/autodl-tmp/rankup_project
PY=$ROOT/envs/rankup/bin/python
RUNS=$ROOT/m1_hdt_runs
LOG=$ROOT/m1_hdt_runs/formal.log
mkdir -p "$RUNS" "$ROOT/m1_hdt_gate/teachers" "$ROOT/m1_hdt_gate/fused"
{
  echo "=== $(date) FORMAL START ==="
  for s in 0 1 2; do
    for f in 0 1 2 3 4; do
      out=$RUNS/hdtm_s${s}_f${f}
      if [ -f "$out/readout.pth" ]; then echo "SKIP hdtm s$s f$f"; continue; fi
      echo "--- $(date) unit s${s}f${f} ---"
      for j in 0 1 2 3; do
        tf=$ROOT/m1_hdt_gate/teachers/B_s${s}_f${f}_j${j}.npz
        if [ -f "$tf" ]; then continue; fi
        "$PY" "$ROOT/m1_hdt_gate/train_teacher_hist.py" --seed "$s" --fold "$f" --j "$j" --out "$tf" \
          > "$ROOT/m1_hdt_gate/teachers/B_s${s}_f${f}_j${j}.log" 2>&1 || { echo "TEACHER FAILED s$s f$f j$j"; }
      done
      fu=$ROOT/m1_hdt_gate/fused/B_s${s}_f${f}.npz
      if [ ! -f "$fu" ]; then
        "$PY" "$ROOT/m1_hdt_gate/make_fused_fold.py" "$s" "$f" >> "$ROOT/m1_hdt_runs/fused.log" 2>&1 || { echo "FUSE FAILED s$s f$f"; continue; }
      fi
      "$PY" "$ROOT/m1_hdt_gate/train_hdt.py" --arm hdtm --seed "$s" --fold "$f" \
        --fused "$fu" --out-dir "$out" > "$RUNS/hdtm_s${s}_f${f}.log" 2>&1 || { echo "STUDENT FAILED s$s f$f"; continue; }
      echo "DONE s$s f$f $(date)"
    done
  done
  echo "=== $(date) FORMAL DONE ==="
} >> "$LOG" 2>&1
