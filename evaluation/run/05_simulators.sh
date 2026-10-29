#!/usr/bin/env bash
# Table 1, simulated rows: compcodeR (Soneson and Delorenzi 2013) and SPsimSeq (TCGA-BRCA source).
source "$(dirname "$0")/../config.sh"
# compcodeR has no fixed seed: to reuse the paper's datasets, unpack the release paper-data-v1 into $PRESET_RESULTS first (README.md).
D=$PRESET_RESULTS/compcodeR_data
[ -d "$D" ] || for spc in 2 5 10; do $TXDE_RSCRIPT "$EVAL/analysis/simulated/compcodeR_generate.R" "$D" $spc 10; done
run compcodeR_v4a_fair2 simulated/compcodeR_bench.py --data_dir "$D" --ckpt "$CKPT" --gpu -1 --jobs "$JOBS"                       # every method
run compcodeR_v4a_pkg simulated/compcodeR_bench.py --data_dir "$D" --ckpt "$CKPT" --gpu -1 --jobs "$JOBS" --preset_decoy --preset_only   # PRESET default rule
# SPsimSeq: 20 datasets each at 3, 4 and 8 per group from TCGA-BRCA (data/tcga_sim), then every method
S=$PRESET_RESULTS/spsimseq_data
[ -d "$S" ] || (cd "$EVAL/analysis" && $TXDE_RSCRIPT simulated/spsimseq_generate.R "$S" 20 "${R_LIBS_SIM:-${R_LIBS:-.}}" \
    "$PRESET_DATA/tcga_sim/BRCA_star_counts.tsv.gz" "$PRESET_DATA/tcga_sim/BRCA_samples.tsv" 3 4 8) > "$PRESET_RESULTS/spsimseq_generate.log" 2>&1
run spsimseq_v4a_pkg simulated/compcodeR_bench.py --data_dir "$S" --ckpt "$CKPT" --gpu -1 --jobs "$JOBS" --preset_decoy
