#!/usr/bin/env bash
# Table 1: every run that paper/make_table1.py reads, with the commands and seeds of the paper, then the table.
# Held-out subsamples are seeded by size, replicate and comparison, so these runs draw the subsamples of the paper.
# For the paper's compcodeR and SPsimSeq rows, unpack the simulated datasets of the release into $PRESET_RESULTS first (README.md).
source "$(dirname "$0")/../config.sh"
P=${P:-$JOBS}
# GTEx: the 240 shards of run/jobs/gtex_sweeps.jobs (make_table1.py checks that all 200 four-tissue shards are present)
shard() { local name=$1; shift; local out="$PRESET_RESULTS/$name"; mkdir -p "$out"; [ -s "$out/sweep.csv" ] && return 0
          (cd "$EVAL/analysis" && $PYTHON -u "${@/CKPT/$CKPT}" --output_dir "$out") > "$out/run.log" 2>&1; }
export -f shard
sed "s#CKPT#$CKPT#g" "$EVAL/run/jobs/gtex_sweeps.jobs" | xargs -P "$P" -L 1 bash -c 'shard "$@"' _
bash "$EVAL/run/05_simulators.sh"
bash "$EVAL/run/06_seqc.sh"
run v4eval3_v4a_fair heldout/fair_bench.py --model "$CKPT" --gpu -1 --data ipf sepsis crohn skin2 --official --ns 8 --reps 20 --jobs "$JOBS" --tag _v4eval3_v4a_fair
run v4eval_v4a_fair heldout/fair_bench.py --model "$CKPT" --gpu -1 --data mouse_genotype tcga --official --ns 8 --reps 20 --jobs "$JOBS" --tag _v4eval_v4a_fair
run v4eval3_v4a_pkg heldout/fair_bench.py --model "$CKPT" --gpu -1 --data ipf sepsis crohn skin2 --official --ns 8 --reps 20 --jobs "$JOBS" --preset_only --tag _v4eval3_v4a_pkg
run v4eval_v4a_pkg heldout/fair_bench.py --model "$CKPT" --gpu -1 --data mouse_genotype tcga --official --ns 8 --reps 20 --jobs "$JOBS" --preset_only --tag _v4eval_v4a_pkg
(cd "$EVAL/paper" && $PYTHON make_table1.py) > "$PRESET_OUTPUT/make_table1.py.log" 2>&1 || echo "  FAILED, see $PRESET_OUTPUT/make_table1.py.log"
echo "done: $PRESET_OUTPUT/tables/table1.tex"
