# PRESET

**Differential Expression Analysis of Bulk RNA-seq with a Prior-Fitted Network Trained Across Experiments**

PRESET is a prior-fitted network for two-group bulk RNA-seq experiments with few samples (trained on 2–16 samples per group, at least 4 per group recommended). It was trained once on simulated experiments, real GTEx backgrounds with planted effects, and real two-group cohorts with labels from held-out samples. Given a new dataset, it returns a posterior for every gene in a single forward pass, without fitting a model to that dataset.

For every gene, PRESET reports

- the probability of a change of meaningful size, $P(|\log_2\mathrm{FC}| > \tau)$ (default $\tau = 0.5$);
- the posterior mean $\log_2$ fold change and the effect size if the gene changes;
- for any list of top-ranked genes, the expected number of false discoveries and the Bayesian FDR.

## Installation

```bash
git clone git@github.com:wodus1035/PRESET.git
cd PRESET
pip install .                 # ".[plot]" adds volcano plots and embedding maps; ".[tutorial]" everything for the notebook
```

Python ≥ 3.10 and PyTorch ≥ 2.1. The trained weights (10 MB) ship with the package. PRESET runs on a CPU or a GPU. On eight CPU threads, the posterior for 25,000 genes and 8 samples takes about 3 s, and the default calls, which add eight passes with decoy genes, about 20 s in total; on a GPU both take a few seconds.

## Quick start

```python
import numpy as np, pandas as pd
import preset

counts = pd.read_csv("counts.tsv", sep="\t", index_col=0).T     # samples x genes, raw integer counts
cond = np.array([0, 0, 0, 0, 1, 1, 1, 1])                       # 0 = control, 1 = case

X, genes, info, expressed = preset.io.prepare(counts)           # give PRESET every gene with a count
res = preset.run(pd.DataFrame(X, columns=genes), cond, report=expressed)
hits = res[preset.calls(res, fdr=0.05)]                         # default rule "auto": target-decoy FDR below 16 per group, Bayesian FDR from 16
# res.expected_false_in_list and res.bayes_fdr: the posterior's own error estimate for any top-k list

preset.plot.volcano(res)                                        # called genes coloured, largest changes named (needs matplotlib)
emb = preset.embed(pd.DataFrame(X, columns=genes), cond)        # genes x 128 network embedding (same matrix as run)
preset.plot.embedding(emb, res)                                 # UMAP map coloured by P(DE) (needs umap-learn; PCA otherwise)
```

Command line:

```bash
preset run counts.tsv samples.tsv --condition group --control healthy --case disease --out results.tsv   # --rule auto (default)
preset run counts.tsv samples.tsv --condition group --control healthy --case disease --out results.tsv --volcano volcano.png --embed embedding.tsv
preset run counts.tsv samples.tsv --condition group --control healthy --case disease --rule bfdr --out results_bfdr.tsv   # Bayesian FDR only (one pass)
preset diagnose counts.tsv samples.tsv --condition group
```

`counts.tsv` may have genes in rows or columns; `samples.tsv` has one row per sample.

## Output

| Column | Meaning |
|---|---|
| `p_de` | $P(\lvert\log_2\mathrm{FC}\rvert > \tau)$ |
| `p_up`, `p_down` | $P(\log_2\mathrm{FC} > \tau)$, $P(\log_2\mathrm{FC} < -\tau)$ |
| `lfc_mean` | posterior mean $\log_2$ fold change |
| `lfc_given_de` | effect size if the gene changes |
| `rank` | rank by `p_de` among reported genes |
| `expected_false_in_list` | expected false discoveries if this gene and every gene above it are called |
| `bayes_fdr` | the same as a fraction of the list |
| `decoy_fdr` | target-decoy FDR of the list down to this gene, used by the default calls below 16 samples per group (`decoy=False` skips it and the calls fall back to `bayes_fdr`) |

`preset.run(..., params=True)` adds the posterior parameters ($\pi_0$, slab location, scale and degrees of freedom); `intervals=True` adds credible intervals (experimental).

`preset.embed` returns the 128-dimensional vector the network computes for each gene of the dataset (the input of the posterior head). Genes close together are genes the network treats alike in this dataset; this is a descriptive view, not a validated measure of co-regulation.

## Tutorial

[`tutorial/PRESET_tutorial.ipynb`](tutorial/PRESET_tutorial.ipynb) walks through an analysis step by step, following the sections of the paper: input data, the per-gene posterior, a DEG list with its expected false discoveries, and four checks against known truth (planted effects, effect sizes, outlier counts and the rest of the cohort). It uses a psoriasis cohort (GSE121212) that PRESET never saw during training (another psoriasis cohort, GSE268958, was among the training cohorts) and runs in a few minutes on a laptop CPU. It also shows the volcano plot, the gene embedding map and a pathway analysis.

## Reproducing the paper

The code that trained the released weights, with the commands of both training stages and the training data, is in [training/](training/README.md). The code that produced Table 1, the comparison with edgeR, DESeq2 and limma-voom, is in [evaluation/](evaluation/README.md), and its stored results are attached to the release `paper-data-v1`.

## Citation

Jang J, Jung I. PRESET: Differential Expression Analysis of Bulk RNA-seq with a Prior-Fitted Network Trained Across Experiments. 2026 (in preparation).
