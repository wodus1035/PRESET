#!/usr/bin/env bash
# Table 1, SEQC rows: ERCC spike-ins and TaqMan fold changes (data/seqc, data/README.md).
source "$(dirname "$0")/../config.sh"
run seqc_v4a_fair seqc/seqc_ercc_fair.py --ckpt "$CKPT" --gpu -1 --jobs 8          # every method
run seqc_v4a_pkg seqc/seqc_ercc_fair.py --ckpt "$CKPT" --gpu -1 --jobs 8           # PRESET rows read by make_table1.py
# TaqMan, every method; writes $PRESET_RESULTS/seqc_taqman_v4a_pkg{,_genes}.csv itself (hence py, not run)
py taqman_v4a_pkg seqc/seqc_taqman_eval.py --model "$CKPT" --tag v4a_pkg --official --gpu -1 --ns 3 --jobs 8
