# Training

The code that trained the released weights (`preset/weights/preset_v1.0.pt`). Training has two stages (Supplementary Note S2).

| Stage | Tasks | Steps | Time on one RTX A6000 |
|---|---|---|---|
| 1 | simulated datasets from the SCM prior (`prior.py`), and GTEx backgrounds with planted effects in half of the steps (`real_background.py`) | 100,000 | 7.1 h |
| 2 | stage 1 continued, with real cohorts labeled on held-out folds in a quarter of the steps (`consensus_bank.py`) | 25,000 | 3.0 h |

Install the package from the repository root (`pip install .`) and run the scripts from the root. A step needs about 16 GB of GPU memory. `--wandb_project NAME` logs to Weights & Biases (`pip install wandb`).

## Data

The training data are three banks under `training/data/`.

| File | Used by | Built by |
|---|---|---|
| `gtex_real_bank_v3.npz` | both stages | `build_real_bank.py` |
| `consensus_bank/` (35 cohorts) | stage 2 | `build_consensus_bank.py`, then `add_observed_labels.py` |
| `mouse_atlas_bank_v1.npz` | only the low-dispersion retraining | `build_mouse_bank.py` |

Raw files go into the data folder of the evaluation (`evaluation/data/`). Run the download commands from the repository root after this setup.

```bash
D=${PRESET_DATA:-evaluation/data}
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series
get() { curl -fL --create-dirs -o "$D/$1" "$2"; }
```

### GTEx bank

The 11 GTEx tissues that no evaluation uses, with the donors of the training half of the donor split (`evaluation/data_cache/gtex_donor_split_v3.json`), so training and the GTEx evaluations share no donor.

```bash
G=https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/counts-by-tissue
for t in breast_mammary_tissue esophagus_mucosa kidney_cortex liver ovary pancreas pituitary prostate spleen testis uterus; do
    get gtex/gene_reads_v10_$t.gct.gz $G/gene_reads_v10_$t.gct.gz
done
python training/build_real_bank.py
```

### Real-cohort bank

The cohorts of Supplementary Table S3, none of which is used in an evaluation. For each fold of a cohort, the labels come from edgeR, limma-voom and the Wilcoxon test on the other folds (`build_consensus_bank.py`), or from the observed log2 ratios of the other folds (`add_observed_labels.py`). The GTEx sex comparisons also need the GTEx bank tissues above and the GTEx subject phenotypes (`bash evaluation/data/download.sh gtex`).

```bash
get training/GSE268958_raw_counts.tsv.gz $GEO/GSE268nnn/GSE268958/suppl/GSE268958_raw_counts.tsv.gz
get training/GSE268958_family.soft.gz $GEO/GSE268nnn/GSE268958/soft/GSE268958_family.soft.gz
get training/GSE291471_RNASeq_raw_count.csv.gz $GEO/GSE291nnn/GSE291471/suppl/GSE291471_RNASeq_raw_count.csv.gz
get training/GSE285190_TED_HC_GH_ERCC_count_matrix.csv.gz $GEO/GSE285nnn/GSE285190/suppl/GSE285190_TED_HC_GH_ERCC_count_matrix.csv.gz
get training/GSE314817_mouse_raw_counts_wo_muscle.txt.gz $GEO/GSE314nnn/GSE314817/suppl/GSE314817_mouse_raw_counts_wo_muscle.txt.gz
get training/GSE314817_family.soft.gz $GEO/GSE314nnn/GSE314817/soft/GSE314817_family.soft.gz
get bulkformer/cvd_adata.h5ad https://zenodo.org/records/15744294/files/cvd_adata.h5ad   # BulkFormer cardiovascular collection

# R packages of evaluation/install_R_packages.R, 169 reference fits (7 min with 40 workers)
python training/build_consensus_bank.py --jobs 40
python training/add_observed_labels.py
```

`training/data_cache/` holds the group and series of each sample of the cardiovascular collection, taken from GEO.

## Run

```bash
bash training/stage1.sh                 # outputs/stage1/preset_best.pt, the lowest validation loss
bash training/stage2.sh                 # outputs/stage2/preset.pt, the checkpoint at the last step
```

The released weights are the second-stage checkpoint at step 25,000 without the optimizer state. Pass a checkpoint to the package with `preset.run(..., weights="outputs/stage2/preset.pt")` or `PRESET_WEIGHTS`.

Both scripts pass further arguments on to `train.py`, which lists every option with `--help`. The other models of the paper were trained as follows.

| Model | Commands |
|---|---|
| other seeds, both stages (Note S2: 3101, 3102, 3103) | `stage1.sh outputs/s1_3101 --seed 3101`, then `stage2.sh outputs/s1_3101/preset_best.pt outputs/s2_3101 --seed 3101` |
| other seeds, second stage (Note S2: 2027 to 2030) | `stage2.sh outputs/stage1/preset_best.pt outputs/s2_2027 --seed 2027` |
| labels without DE methods (Sect. 3.3) | `stage2.sh outputs/stage1/preset_best.pt outputs/s2_observed --cons_label observed --cons_exclude Psoriasis GSE165303` |
| no GTEx backgrounds (pre-registration D4) | `stage1.sh outputs/s1_nobank --p_real 0 --steps 25000 --lr_warmup 2000`, then `stage2.sh outputs/s1_nobank/preset_best.pt outputs/s2_nobank --p_real 0` |
| low-dispersion backgrounds (Note S6) | both stages with `--real_bank2 training/data/mouse_atlas_bank_v1.npz --real_bank2_frac 0.3` |

## Reproducibility

The paper's models were trained on a GPU, where some operations are not deterministic, so a rerun gives a model close to the released one but not identical to it. On the CPU the training loop is deterministic. With the same seed, this code reproduces the original training code exactly: 30 steps of each stage gave identical losses, gradient norms, validation losses, weights and optimizer states. The three banks rebuilt by the scripts above are identical to those used for training.
