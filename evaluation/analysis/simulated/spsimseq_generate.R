#!/usr/bin/env Rscript
# SPsimSeq simulations (Assefa et al., Bioinformatics 2020; semi-parametric, Gaussian copula) in the compcodeR file layout.
# Source: TCGA-BRCA normals vs primary tumors of the same patients (tcga_download.py --max_tumour 0), edgeR filterByExpr,
# 10,000 random genes (seed 2026) to keep the copula's eigendecomposition feasible.
# Defaults except n.genes = 9,900, pDE = 0.05, lfc.thrld = 0.5: the source has 554 DE and 9,446 null candidates, and larger
# values make SPsimSeq reuse source genes, which gives identical rows.
# Truth (SPsimSeq 1.16.0): non-DE genes come from one pooled density (log2FC 0), DE genes from per-group densities, so
# log2FC = log2 ratio of group mean CPM (TMM-scaled, as calculateCPM).
# genCopula is replaced for speed only: one eigendecomposition per correlation matrix, same "eigen" draw as rmvnorm.
# usage: Rscript analysis/simulated/spsimseq_generate.R <outdir> <n_datasets> <libdir> <source_counts.tsv.gz> <source_samples.tsv> <spc...>
args <- commandArgs(trailingOnly = TRUE)
outdir <- args[1]
nsim <- as.integer(args[2])
.libPaths(c(args[3], .libPaths()))
f_cnt <- args[4]
f_sam <- args[5]
spcs <- as.integer(args[-(1:5)])
suppressMessages({
  library(SPsimSeq)
  library(edgeR)
})
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Source data: TCGA-BRCA normals and tumors
raw <- read.delim(gzfile(f_cnt), check.names = FALSE, row.names = 1)
raw$gene_name <- NULL
sam <- read.delim(f_sam)
sam <- sam[sam$sample_type %in% c("Solid Tissue Normal", "Primary Tumor") & sam$sample %in% colnames(raw), ]
grp <- ifelse(sam$sample_type == "Solid Tissue Normal", "normal", "tumour")
cnt <- as.matrix(raw[, sam$sample])
storage.mode(cnt) <- "double"
keep <- filterByExpr(DGEList(cnt), group = grp)
set.seed(2026)
genes <- sample(rownames(cnt)[keep], min(10000, sum(keep)))
cnt <- cnt[genes, ]
cat("source:", nrow(raw), "genes ->", sum(keep), "after filterByExpr ->", nrow(cnt), "used;", ncol(cnt), "samples;",
    paste(names(table(grp)), table(grp), collapse = " "), "\n")

# True log2 fold change of each source gene
cpm <- sweep(cnt, 2, edgeR::calcNormFactors(cnt) * colSums(cnt), "/") * 1e6   # = exp(SPsimSeq:::calculateCPM(cnt, 1e6, 1)) - 1
true_src <- log2(rowMeans(cpm[, grp == "tumour"]) / rowMeans(cpm[, grp == "normal"]))

# Cached copula: one eigendecomposition per correlation matrix
.eig <- new.env()
genCopula_cached <- function(corMats, exprmt.design) {
  n.batch <- rowSums(exprmt.design$exprmt.config)
  copList <- lapply(names(n.batch), function(b) {
    S <- corMats[[b]]
    key <- paste(b, nrow(S), signif(sum(S[seq_len(min(5e5, length(S)))]), 12))
    if (is.null(.eig[[key]])) {
      t0 <- Sys.time()
      ev <- eigen(S, symmetric = TRUE)
      .eig[[key]] <- t(ev$vectors %*% (t(ev$vectors) * sqrt(pmax(ev$values, 0))))
      cat("  eigendecomposition", nrow(S), "genes:", format(round(Sys.time() - t0, 1)),
          "; negative eigenvalues clipped:", sum(ev$values < 0), "\n")
    }
    Z <- matrix(rnorm(n.batch[b] * nrow(S)), nrow = n.batch[b]) %*% .eig[[key]]
    Cpl <- pnorm(Z)
    colnames(Cpl) <- rownames(S)
    t(Cpl)
  })
  Reduce(copList, f = cbind)
}
assignInNamespace("genCopula", genCopula_cached, "SPsimSeq")

for (spc in spcs) {
  set.seed(2026 + spc)
  t0 <- Sys.time()
  sim <- SPsimSeq(n.sim = nsim, s.data = cnt, group = grp, n.genes = 9900, batch.config = 1, group.config = c(0.5, 0.5),
                  tot.samples = 2 * spc, pDE = 0.05, lfc.thrld = 0.5, result.format = "list", verbose = FALSE)
  for (r in seq_len(nsim)) {
    s <- sim[[r]]
    y <- s$counts
    rd <- as.data.frame(s$rowData)
    cd <- as.data.frame(s$colData)

    # Truth: source gene fold change for DE genes, 0 otherwise
    de <- as.integer(rd$DE.ind)
    src <- as.character(rd$source.ID)
    tl <- ifelse(de == 1, true_src[src], 0)
    cond <- ifelse(as.character(cd$Group) == "normal", 1L, 2L)          # condition 1 = normal (control), 2 = tumor

    # Write counts, truth and samples
    tag <- sprintf("SPsim_brca_de05_%dspc_r%d", spc, r)
    rownames(y) <- sprintf("g%05d", seq_len(nrow(y)))
    colnames(y) <- sprintf("s%02d", seq_len(ncol(y)))
    write.table(y, file.path(outdir, paste0(tag, "_counts.tsv")), sep = "\t", quote = FALSE)
    write.table(data.frame(gene = rownames(y), de = de, true_log2fc = tl, source = src),
                file.path(outdir, paste0(tag, "_truth.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
    write.table(data.frame(sample = colnames(y), condition = cond, group = cd$Group, lib_size = colSums(y)),
                file.path(outdir, paste0(tag, "_samples.tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
    cat(tag, dim(y), "DE", sum(de), "|true|>0.5", sum(abs(tl) > 0.5 & de == 1),
        "reused source genes", sum(duplicated(src)), "groups", paste(table(cond), collapse = "/"),
        "median lib", median(colSums(y)), "\n")
  }
  cat("spc", spc, "done in", format(round(Sys.time() - t0, 1)), "\n")
}
