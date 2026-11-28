#!/usr/bin/env python3
"""Validate one or all teacher caches and optionally their fused target."""
import argparse
from check_status import check_teacher

parser = argparse.ArgumentParser()
parser.add_argument('seed', type=int)
parser.add_argument('fold', type=int)
parser.add_argument('--teacher', type=int, choices=range(4))
parser.add_argument('--fused', action='store_true')
args = parser.parse_args()
issues = check_teacher(args.seed, args.fold, fused=args.fused, teacher=args.teacher)
for issue in issues:
    print(issue)
if issues:
    raise SystemExit(1)
print('teacher cache verified')
