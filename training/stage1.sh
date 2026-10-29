#!/usr/bin/env bash
# First stage of the released model: 100,000 steps on simulated datasets and GTEx backgrounds (7.1 h on one RTX A6000).
#   bash training/stage1.sh [output folder] [extra train.py arguments, e.g. --seed 2027 --gpu 1]
cd "$(dirname "$0")/.."
OUT=${1:-outputs/stage1}
[ $# -gt 0 ] && shift
python training/train.py --output_dir "$OUT" "$@"
