#!/usr/bin/env Rscript
# compcodeR simulations (Soneson & Delorenzi 2013) as in the package vignette, written as TSVs for compcodeR_bench.py.
# Designs (12,500 genes, 1,250 DE): B = baseline NB, P = half the genes Poisson, S = single outliers, R = random outliers;
# _625_625 = 625 up + 625 down, _1250_0 = all up. Other settings are vignette defaults.
# usage: Rscript analysis/simulated/compcodeR_generate.R <outdir> <samples.per.cond> <n_repl>
suppressMessages(library(compcodeR))
args <- commandArgs(trailingOnly = TRUE)
outdir <- args[1]
spc <- as.integer(args[2])
nrep <- as.integer(args[3])
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
designs <- list(
  B_625_625 = list(n.diffexp = 1250, fraction.upregulated = 0.5),
  B_1250_0  = list(n.diffexp = 1250, fraction.upregulated = 1.0),
  P_625_625 = list(n.diffexp = 1250, fraction.upregulated = 0.5, fraction.non.overdispersed = 0.5),
  S_625_625 = list(n.diffexp = 1250, fraction.upregulated = 0.5, single.outlier.high.prob = 0.1),
  R_625_625 = list(n.diffexp = 1250, fraction.upregulated = 0.5, random.outlier.high.prob = 0.1)
)
for (nm in names(designs)) for (r in seq_len(nrep)) {
  a <- designs[[nm]]

  # Simulate one dataset
  obj <- do.call(generateSyntheticData, c(list(dataset = nm, n.vars = 12500, samples.per.cond = spc, repl.id = r,
                                                seqdepth = 1e7, between.group.diffdisp = FALSE,
                                                filter.threshold.total = 1, filter.threshold.mediancpm = 0), a))
  cnt <- obj@count.matrix
  va <- obj@variable.annotations
  sa <- obj@sample.annotations

  # Write counts, truth and samples
  tag <- sprintf("%s_%dspc_r%d", nm, spc, r)
  write.table(cnt, file.path(outdir, paste0(tag, "_counts.tsv")), sep = "\t", quote = FALSE)
  write.table(data.frame(gene = rownames(va), de = va$differential.expression, true_log2fc = va$truelog2foldchanges,
                         upregulation = va$upregulation, downregulation = va$downregulation,
                         n_random_outliers = if (is.null(va$n.random.outliers)) NA else va$n.random.outliers,
                         n_single_outliers = if (is.null(va$n.single.outliers)) NA else va$n.single.outliers),
              file.path(outdir, paste0(tag, "_truth.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(data.frame(sample = rownames(sa), condition = sa$condition, depth_factor = sa$depth.factor),
              file.path(outdir, paste0(tag, "_samples.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
  cat(tag, dim(cnt), "\n")
}
