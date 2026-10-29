# Table 1

Code that produced Table 1 of the paper, the comparison of PRESET with edgeR, DESeq2 and limma-voom on simulated datasets, effects planted into GTEx, spike-ins and TaqMan fold changes, and held-out real cohorts. Every method is run on the same count matrix, subsamples and seeds, and is scored on the same genes.

## Setup

```bash
pip install . && pip install -r evaluation/requirements.txt   # from the repository root
Rscript evaluation/install_R_packages.R                         # R 4.4, Bioconductor 3.20
bash evaluation/data/download.sh                                # public data, see data/README.md
```

## Rebuild the table from the stored results

The result files of the paper are attached to the release `paper-data-v1`.

```bash
curl -L -O https://github.com/wodus1035/PRESET/releases/download/paper-data-v1/preset_table1_results.tar.gz
mkdir -p evaluation/results && tar -xzf preset_table1_results.tar.gz -C evaluation/results
PRESET_RESULTS=$PWD/evaluation/results/table1 python evaluation/paper/make_table1.py   # evaluation/output/tables/table1.tex
```

## Rerun every comparison

compcodeR has no fixed seed, so unpack the simulated datasets of the paper from the same release first. `run/table1.sh` then runs all parts on the CPU and writes the table. Settings such as the number of workers are in `config.sh`.

```bash
curl -L -O https://github.com/wodus1035/PRESET/releases/download/paper-data-v1/preset_simulated_datasets.tar.gz
mkdir -p evaluation/results && tar -xzf preset_simulated_datasets.tar.gz -C evaluation/results
bash evaluation/run/table1.sh
```

| Rows | Script in `analysis/` |
|---|---|
| GTEx, four tissues and outlier counts | `planted/defrac_sweep.py` |
| compcodeR, SPsimSeq | `simulated/compcodeR_bench.py` |
| ERCC, TaqMan | `seqc/seqc_ercc_fair.py`, `seqc/seqc_taqman_eval.py` |
| disease, TCGA and mouse cohorts | `heldout/fair_bench.py` |

Run from an empty results folder, this regenerated Table 1 byte for byte.
