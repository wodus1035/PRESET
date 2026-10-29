#!/usr/bin/env bash
# Second stage of the released model: 25,000 steps from the best first-stage checkpoint, with real-cohort tasks in 25% of
# the steps (3.0 h on one RTX A6000). The released weights are <output folder>/preset.pt, the checkpoint at the last step.
#   bash training/stage2.sh [first-stage checkpoint] [output folder] [extra train.py arguments]
cd "$(dirname "$0")/.."
INIT=${1:-outputs/stage1/preset_best.pt}
OUT=${2:-outputs/stage2}
shift $(( $# < 2 ? $# : 2 ))
python training/train.py --output_dir "$OUT" --init_from "$INIT" \
    --steps 25000 --lr 1e-4 --lr_warmup 2000 --warm 1 --warm_regime 5000 --p_cons 0.25 "$@"
