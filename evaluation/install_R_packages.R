#!/usr/bin/env Rscript
# R packages of the evaluation (R 4.4, Bioconductor 3.20). Versions used for the paper are listed in README.md.
#   Rscript install_R_packages.R [library folder]
lib <- commandArgs(trailingOnly = TRUE)[1]
if (!is.na(lib)) {
  dir.create(lib, showWarnings = FALSE, recursive = TRUE)
  .libPaths(c(lib, .libPaths()))
} else lib <- .libPaths()[1]
repos <- "https://cloud.r-project.org"
for (p in c("BiocManager", "remotes")) if (!requireNamespace(p, quietly = TRUE)) install.packages(p, lib = lib, repos = repos)

# Bioconductor and CRAN
BiocManager::install(version = "3.20", ask = FALSE, update = FALSE, lib = lib)
BiocManager::install(c("edgeR", "limma", "DESeq2", "apeglm", "BiocParallel", "EBSeq", "compcodeR", "SPsimSeq"),
                     ask = FALSE, update = FALSE, lib = lib)
install.packages(c("ashr", "statmod", "mvtnorm", "data.table"), lib = lib, repos = repos)

# GitHub, pinned commits
remotes::install_github("JSB-UCLA/Clipper", ref = "33a5513524833c00e5e256b883a91111fa4c82e0",
                        lib = lib, upgrade = "never")
remotes::install_github("chexjiang/NullstrapDE", ref = "26b1e347622ed2eb4391157d285167c1cf2a9712",
                        lib = lib, upgrade = "never")

# Installed versions
for (p in c("edgeR", "limma", "DESeq2", "apeglm", "BiocParallel", "EBSeq", "compcodeR", "SPsimSeq", "ashr", "Clipper", "NullstrapDE"))
  cat(p, as.character(packageVersion(p, lib.loc = lib)), "\n")
