"""Wrappers for the conventional methods (DESeq2, edgeR, limma-voom, Wilcoxon, EBSeq, Nullstrap-edgeR) used by the other
evaluation scripts, and an FDR-power benchmark on prior draws that needs the prior simulator (training/prior.py on PYTHONPATH).

    python analysis/common/benchmark.py --ckpts CKPT --names PRESET
"""

import argparse, json, math, os, random, subprocess, tempfile, warnings
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
import torch

import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from preset.model import ModelConfig, PRESET

try:
    from prior import PriorConfig, generate_batch  # prior simulator (training/prior.py)
except ImportError:
    PriorConfig = generate_batch = None
# PyDESeq2: only for TXDE_DESEQ2=py
try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ImportError:
    DeseqDataSet = DeseqStats = None

RSCRIPT = os.environ.get("TXDE_RSCRIPT", "Rscript")
R_TIMEOUT = 600


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(path, device):
    if path is None:
        from preset.weights import resolve_weights

        path = resolve_weights(None)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["mcfg"]) if "mcfg" in ckpt else ModelConfig()
    model = PRESET(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def design_ok(cond, batch):
    if len(np.unique(batch)) < 2:
        return True
    X = np.column_stack([np.ones(len(cond)), cond, batch])
    return np.linalg.matrix_rank(X) == X.shape[1]


# TXDE_DESEQ2=r (default, all paper results): Bioconductor DESeq2 following its vignette; py: PyDESeq2
DESEQ2_IMPL = os.environ.get("TXDE_DESEQ2", "r").strip().lower()


def run_deseq2_r(counts, cond, batch, tau=0.5, cooks=True, shrink=False):
    """Bioconductor DESeq2 (vignette workflow), indexed g0.. in input order; filtered genes get NaN."""
    counts = np.asarray(counts, np.int64)
    cond = np.asarray(cond, int)
    batch = np.asarray(batch, int)
    if len(np.unique(batch)) > 1 and not design_ok(cond, batch):
        return None
    d = tempfile.mkdtemp(prefix="preset_deseq2r_")
    cf = os.path.join(d, "counts.tsv")
    mf = os.path.join(d, "meta.tsv")
    of = os.path.join(d, "out.tsv")
    sf = os.path.join(d, "run.R")
    np.savetxt(cf, counts.T, fmt="%d", delimiter="\t")
    pd.DataFrame({"condition": cond, "batch": batch}).to_csv(mf, sep="\t", index=False)
    script = f"""
        suppressMessages({{
            library(DESeq2)
        }})
        cnt <- as.matrix(read.table("{cf}",sep="\\t"))
        meta <- read.table("{mf}",header=TRUE,sep="\\t")
        rownames(cnt) <- paste0("g",seq_len(nrow(cnt))-1)
        colnames(cnt) <- paste0("s",seq_len(ncol(cnt)))
        meta$condition <- factor(meta$condition,levels=c(0,1))
        meta$batch <- factor(meta$batch)
        rownames(meta) <- colnames(cnt)
        des <- if (length(unique(meta$batch))>1) ~ batch + condition else ~ condition
        dds <- DESeqDataSetFromMatrix(cnt, meta, des)
        keep <- rowSums(counts(dds) >= 10) >= min(table(meta$condition))
        dds <- dds[keep,]
        dds <- DESeq(dds, quiet=TRUE)
        res <- results(dds, name="condition_1_vs_0", lfcThreshold={tau}, altHypothesis="greaterAbs", alpha=0.05{"" if cooks else ", cooksCutoff=FALSE"})
        out <- data.frame(gene=rownames(res), baseMean=res$baseMean, log2FoldChange=res$log2FoldChange, pvalue=res$pvalue, padj=res$padj)
        if ({"TRUE" if shrink else "FALSE"}) {{
            s <- lfcShrink(dds, coef="condition_1_vs_0", type="apeglm", quiet=TRUE)
            out$lfc_apeglm <- s$log2FoldChange
        }}
        write.table(out,"{of}",sep="\\t",row.names=FALSE,quote=FALSE)
    """
    try:
        open(sf, "w").write(script)
        r = subprocess.run([RSCRIPT, "--vanilla", sf], capture_output=True, text=True, timeout=R_TIMEOUT)
        if not os.path.exists(of):
            print("[DESeq2 (R) failed]", r.stderr[-500:])
            return None
        res = pd.read_csv(of, sep="\t")
        genes = [f"g{i}" for i in range(counts.shape[1])]
        return res.set_index("gene").reindex(genes)
    except subprocess.TimeoutExpired:
        print("[DESeq2 (R) timeout]")
        return None
    finally:
        for f in (cf, mf, of, sf):
            try:
                os.remove(f)
            except OSError:
                pass


def run_deseq2_scored(counts, cond, batch, ev, tau=0.5, cooks=True, n_cpus=1):
    """DESeq2 results for all G genes of counts (n, G), NaN outside `ev`. R mode fits all G genes, py mode counts[:, ev]."""
    G = counts.shape[1]
    out = pd.DataFrame(
        index=[f"g{i}" for i in range(G)], columns=["log2FoldChange", "pvalue", "padj"], dtype=float
    )
    if DESEQ2_IMPL == "r":
        r = run_deseq2_r(counts, cond, batch, tau=tau, cooks=cooks)
        if r is None:
            return None
        out[["log2FoldChange", "pvalue", "padj"]] = r[["log2FoldChange", "pvalue", "padj"]].to_numpy(float)
    else:
        r = run_deseq2(np.asarray(counts)[:, ev], cond, batch, tau=tau, n_cpus=n_cpus, cooks=cooks)
        if r is None:
            return None
        out.loc[out.index[ev], ["log2FoldChange", "pvalue", "padj"]] = r[
            ["log2FoldChange", "pvalue", "padj"]
        ].to_numpy(float)
    out.loc[out.index[~np.asarray(ev, bool)]] = np.nan
    return out


def universe_mask(counts_gs):
    """Genes passed to every method: all with a count (TXDE_PREFILTER=none, default) or >= 10 counts in >= 25% of samples ("shared")."""
    counts_gs = np.asarray(counts_gs)
    if os.environ.get("TXDE_PREFILTER", "none").strip().lower() == "none":
        return counts_gs.sum(1) > 0
    return (counts_gs >= 10).sum(1) >= max(1, int(np.ceil(0.25 * counts_gs.shape[1])))


print(
    f"[benchmark] DESeq2 implementation: {'Bioconductor DESeq2 (R)' if DESEQ2_IMPL=='r' else 'PyDESeq2'}; "
    f"pre-filter: {os.environ.get('TXDE_PREFILTER','none')}",
    flush=True,
)


def run_deseq2(counts, cond, batch, tau=0.5, n_cpus=1, cooks=False):
    """PyDESeq2, or Bioconductor DESeq2 if TXDE_DESEQ2=r. cooks=True enables outlier handling (the R default)."""
    if DESEQ2_IMPL == "r":
        return run_deseq2_r(counts, cond, batch, tau=tau, cooks=cooks)
    counts = np.asarray(counts, np.int64)
    cond = np.asarray(cond, int)
    batch = np.asarray(batch, int)
    n, G = counts.shape
    samples = [f"s{i}" for i in range(n)]
    genes = [f"g{i}" for i in range(G)]
    c = pd.DataFrame(counts, index=samples, columns=genes)
    meta = {"condition": pd.Categorical(cond.astype(str), categories=["0", "1"])}
    design = "~condition"

    if len(np.unique(batch)) > 1:
        if not design_ok(cond, batch):
            return None
        meta["batch"] = pd.Categorical(batch.astype(str), categories=["0", "1"])
        design = "~batch + condition"

    meta = pd.DataFrame(meta, index=samples)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pydeseq2 <= 0.4: n_cpus kwargs; >= 0.5: an `inference` object
            try:
                dds = DeseqDataSet(
                    counts=c, metadata=meta, design=design, refit_cooks=cooks, n_cpus=n_cpus, quiet=True
                )
                dds.deseq2()
                ds = DeseqStats(
                    dds,
                    contrast=["condition", "1", "0"],
                    lfc_null=tau,
                    alt_hypothesis="greaterAbs",
                    cooks_filter=cooks,
                    independent_filter=True,
                    n_cpus=n_cpus,
                    quiet=True,
                )
            except TypeError:
                from pydeseq2.default_inference import DefaultInference

                inf = DefaultInference(n_cpus=n_cpus)
                dds = DeseqDataSet(
                    counts=c, metadata=meta, design=design, refit_cooks=cooks, inference=inf, quiet=True
                )
                dds.deseq2()
                ds = DeseqStats(
                    dds,
                    contrast=["condition", "1", "0"],
                    lfc_null=tau,
                    alt_hypothesis="greaterAbs",
                    cooks_filter=cooks,
                    independent_filter=True,
                    inference=inf,
                    quiet=True,
                )
            ds.summary()
        return ds.results_df.reindex(genes)
    except Exception as e:
        print(f"[PyDESeq2 failed] {type(e).__name__}: {e}")
        return None


def run_limma_voom(counts, cond, batch, tau=0.5, filter_expr=False):
    """limma-voom with treat(lfc=tau). filter_expr=True: filterByExpr -> calcNormFactors -> voom, as in the user's guide."""
    counts = np.asarray(counts, np.int64)
    cond = np.asarray(cond, int)
    batch = np.asarray(batch, int)
    if len(np.unique(batch)) > 1 and not design_ok(cond, batch):
        return None
    d = tempfile.mkdtemp(prefix="preset_limma_")
    cf = os.path.join(d, "counts.tsv")
    mf = os.path.join(d, "meta.tsv")
    of = os.path.join(d, "out.tsv")
    sf = os.path.join(d, "run.R")
    np.savetxt(cf, counts.T, fmt="%d", delimiter="\t")
    pd.DataFrame({"condition": cond, "batch": batch}).to_csv(mf, sep="\t", index=False)
    script = f"""
        suppressMessages({{
            library(limma)
            library(edgeR)
        }})
        cnt <- as.matrix(read.table("{cf}",sep="\\t"))
        meta <- read.table("{mf}",header=TRUE,sep="\\t")
        rownames(cnt) <- paste0("g",seq_len(nrow(cnt))-1)
        colnames(cnt) <- paste0("s",seq_len(ncol(cnt)))
        meta$condition <- factor(meta$condition,levels=c(0,1))
        meta$batch <- factor(meta$batch)
        des <- if (length(unique(meta$batch))>1) model.matrix(~batch+condition,data=meta) else model.matrix(~condition,data=meta)
        y <- DGEList(counts=cnt)
        if ({"TRUE" if filter_expr else "FALSE"}) {{
            keep <- filterByExpr(y, des)
            y <- y[keep,,keep.lib.sizes=FALSE]
        }}
        y <- calcNormFactors(y)
        v <- voom(y,des)
        fit <- lmFit(v,des)
        coef_idx <- grep("^condition",colnames(des))
        tt <- if ({tau}>0) topTreat(treat(fit,lfc={tau}),coef=coef_idx,n=Inf,sort.by="none") else topTable(eBayes(fit),coef=coef_idx,n=Inf,sort.by="none")
        out <- data.frame(gene=rownames(tt),logFC=tt$logFC,PValue=tt$P.Value,FDR=tt$adj.P.Val)
        write.table(out,"{of}",sep="\\t",row.names=FALSE,quote=FALSE)
    """
    try:
        open(sf, "w").write(script)
        r = subprocess.run([RSCRIPT, "--vanilla", sf], capture_output=True, text=True, timeout=R_TIMEOUT)
        if not os.path.exists(of):
            print("[limma failed]", r.stderr[-500:])
            return None
        res = pd.read_csv(of, sep="\t")
        genes = [f"g{i}" for i in range(counts.shape[1])]
        return res.set_index("gene").reindex(genes)
    finally:
        for f in (cf, mf, of, sf):
            try:
                os.remove(f)
            except OSError:
                pass


def run_wilcoxon(counts, cond, batch=None, tau=0.5):
    """Wilcoxon rank-sum on log2(normalized count + 1), BH-adjusted; logFC = log2 ratio of normalized means; tau is ignored."""
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    counts = np.asarray(counts, np.float64)
    cond = np.asarray(cond, int)  # counts (n,G)
    logg = np.log(counts + 0.5)
    ref = logg.mean(0, keepdims=True)
    ok = (counts > 0).all(0)
    sf = (
        np.exp(np.median((logg - ref)[:, ok], axis=1))
        if ok.sum() >= 50
        else counts.sum(1) / counts.sum(1).mean()
    )
    norm = counts / sf[:, None]
    x = np.log2(norm + 1.0)
    a = x[cond == 0]
    b = x[cond == 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = stats.mannwhitneyu(b, a, axis=0, alternative="two-sided").pvalue
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    fdr = multipletests(p, method="fdr_bh")[1]
    lfc = np.log2(norm[cond == 1].mean(0) + 0.5) - np.log2(norm[cond == 0].mean(0) + 0.5)
    return pd.DataFrame(
        {"logFC": lfc, "PValue": p, "FDR": fdr}, index=[f"g{i}" for i in range(counts.shape[1])]
    )


def run_ebseq(counts, cond, batch=None, tau=0.5, maxround=7):
    """EBSeq PPDE and logFC (case over control); None on failure. tau is ignored (EBSeq has no effect-size threshold)."""
    counts = np.asarray(counts, np.int64)
    cond = np.asarray(cond, int)
    d = tempfile.mkdtemp(prefix="preset_ebseq_")
    cf = os.path.join(d, "counts.tsv")
    mf = os.path.join(d, "meta.tsv")
    of = os.path.join(d, "out.tsv")
    sf = os.path.join(d, "run.R")
    np.savetxt(cf, counts.T, fmt="%d", delimiter="\t")
    pd.DataFrame({"condition": cond}).to_csv(mf, sep="\t", index=False)
    script = f"""
        suppressMessages(library(EBSeq))
        cnt <- as.matrix(read.table("{cf}",sep="\\t"))
        meta <- read.table("{mf}",header=TRUE,sep="\\t")
        rownames(cnt) <- paste0("g",seq_len(nrow(cnt))-1)
        colnames(cnt) <- paste0("s",seq_len(ncol(cnt)))
        cond <- factor(meta$condition,levels=c(0,1))          # Condition1 = control(0), Condition2 = case(1)
        keep <- rowSums(cnt) > 0 & apply(cnt,1,function(r) length(unique(r))>1)
        res <- data.frame(gene=rownames(cnt), PPDE=0, logFC=0, stringsAsFactors=FALSE)
        ok <- tryCatch({{
            sz <- MedianNorm(cnt[keep,,drop=FALSE])
            eb <- EBTest(Data=cnt[keep,,drop=FALSE], Conditions=cond, sizeFactors=sz, maxround={maxround})
            pp <- GetPPMat(eb)
            fc <- PostFC(eb)
            i <- match(rownames(pp), res$gene)
            res$PPDE[i] <- pp[,"PPDE"]
            pf <- as.numeric(fc$PostFC)
            names(pf) <- names(eb$MeanList)   # PostFC comes back unnamed, in MeanList order
            j <- match(names(pf), res$gene)
            res$logFC[j] <- -log2(pf)                         # PostFC is Condition1/Condition2 = control/case
            TRUE
        }}, error=function(e) {{
            message(conditionMessage(e))
            FALSE
        }})
        if (!ok) quit(status=3)
        write.table(res, "{of}", sep="\\t", quote=FALSE, row.names=FALSE)
    """
    open(sf, "w").write(script)
    try:
        r = subprocess.run([RSCRIPT, sf], capture_output=True, text=True, timeout=R_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not os.path.exists(of):
        return None
    return pd.read_table(of)


def run_edger(counts, cond, batch, tau=0.5, filter_expr=False, robust=False):
    """edgeR glmQLFit + glmTreat(lfc=tau). filter_expr=True and robust=True (both off by default) give the user's guide workflow."""
    counts = np.asarray(counts, np.int64)
    cond = np.asarray(cond, int)
    batch = np.asarray(batch, int)
    if len(np.unique(batch)) > 1 and not design_ok(cond, batch):
        return None

    d = tempfile.mkdtemp(prefix="preset_edger_")
    cf = os.path.join(d, "counts.tsv")
    mf = os.path.join(d, "meta.tsv")
    of = os.path.join(d, "out.tsv")
    sf = os.path.join(d, "run.R")

    np.savetxt(cf, counts.T, fmt="%d", delimiter="\t")
    pd.DataFrame({"condition": cond, "batch": batch}).to_csv(mf, sep="\t", index=False)

    script = f"""
        suppressMessages({{
            library(edgeR)
            library(statmod)
        }})
        cnt <- as.matrix(read.table("{cf}",sep="\\t"))
        meta <- read.table("{mf}",header=TRUE,sep="\\t")
        rownames(cnt) <- paste0("g",seq_len(nrow(cnt))-1)  # 0-based to match Python gene labels g0..g(G-1)
        colnames(cnt) <- paste0("s",seq_len(ncol(cnt)))

        meta$condition <- factor(meta$condition,levels=c(0,1))
        meta$batch <- factor(meta$batch)

        y <- DGEList(counts=cnt)
        y <- calcNormFactors(y)

        if (length(unique(meta$batch))>1) {{
            des <- model.matrix(~batch + condition,data=meta)
        }} else {{
            des <- model.matrix(~condition,data=meta)
        }}

        if ({"TRUE" if filter_expr else "FALSE"}) {{
            keep <- filterByExpr(y, des)
            y <- y[keep,,keep.lib.sizes=FALSE]
            y <- calcNormFactors(y)
        }}
        {"y <- estimateDisp(y,des)" if os.environ.get("TXDE_EDGER_LEGACY","0")=="1" else ""}
        fit <- glmQLFit(y,des,legacy={"TRUE" if os.environ.get("TXDE_EDGER_LEGACY","0")=="1" else "FALSE"},robust={"TRUE" if robust else "FALSE"})   # TXDE_EDGER_LEGACY=1: edgeR 3.x QL fit with estimateDisp
        coef_idx <- grep("^condition",colnames(des))
        tt <- glmTreat(fit,coef=coef_idx,lfc={tau})

        tab <- topTags(tt,n=Inf,sort.by="none")$table
        out <- data.frame(
            gene=rownames(tab),
            logFC=tab$logFC,
            PValue=tab$PValue,
            FDR=tab$FDR
        )
        write.table(out,"{of}",sep="\\t",row.names=FALSE,quote=FALSE)
        """

    try:
        open(sf, "w").write(script)
        r = subprocess.run([RSCRIPT, "--vanilla", sf], capture_output=True, text=True, timeout=R_TIMEOUT)
        if not os.path.exists(of):
            print("[edgeR failed]", r.stderr[-500:])
            return None
        res = pd.read_csv(of, sep="\t")
        genes = [f"g{i}" for i in range(counts.shape[1])]
        return res.set_index("gene").reindex(genes)
    finally:
        for f in (cf, mf, of, sf):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(d)
        except OSError:
            pass


class Metrics:
    def __init__(self, thresholds):
        self.t = thresholds
        self.tp = {x: 0 for x in thresholds}
        self.fp = {x: 0 for x in thresholds}
        self.nt = {x: 0 for x in thresholds}
        self.ae = 0.0
        self.se = 0.0
        self.n = 0
        self.failed = 0
        self.total = 0
        self.skipped = 0  # dropped for all methods (batch confounded with condition)
        self.nll = 0.0
        self.nb = 0

    def update_preset(self, model, logits, target, tau):
        pred = model.head.mean(logits)
        p = model.head.prob_abs_gt(logits, tau)
        truth = target.abs() > tau
        err = pred - target
        self.ae += err.abs().sum().item()
        self.se += (err**2).sum().item()
        self.n += target.numel()
        self.nll += model.head.nll(logits, target).item()
        self.nb += 1
        for q in self.t:
            sel = p > q
            self.tp[q] += (sel & truth).sum().item()
            self.fp[q] += (sel & ~truth).sum().item()
            self.nt[q] += truth.sum().item()

    def update_pvalue(self, p, truth_lfc, tau, lfc=None):
        self.total += 1
        if p is None:
            self.failed += 1
            return
        truth = np.abs(truth_lfc) > tau

        if lfc is not None:
            valid = np.isfinite(lfc)
            if valid.any():
                err = lfc[valid] - truth_lfc[valid]
                self.ae += np.abs(err).sum()
                self.se += (err**2).sum()
                self.n += valid.sum()

        p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
        for a in self.t:
            sel = p < a
            self.tp[a] += np.sum(sel & truth)
            self.fp[a] += np.sum(sel & ~truth)
            self.nt[a] += truth.sum()

    def summary(self):
        out = {
            "mae": self.ae / max(1, self.n),
            "rmse": math.sqrt(self.se / max(1, self.n)),
            "failed": self.failed,
            "total": self.total,
            "skipped": self.skipped,
        }
        if self.nb:
            out["nll"] = self.nll / self.nb
        for x in self.t:
            out[f"FDR@{x}"] = self.fp[x] / max(1, self.tp[x] + self.fp[x])
            out[f"power@{x}"] = self.tp[x] / max(1, self.nt[x])
        return out


@torch.no_grad()
def run_nullstrap_edger(counts, cond, batch, fdr=0.05, test="lrt"):
    """edgeR + Nullstrap-DE (Jiang, Wang & Li 2025) as in the package README, with package defaults. Returns a boolean call
    per input gene (no q-values; filtered genes False), None on failure. counts: samples x genes."""
    counts = np.asarray(counts, np.int64)
    cond = np.asarray(cond, int)
    batch = np.asarray(batch, int)
    if len(np.unique(batch)) > 1 and not design_ok(cond, batch):
        return None
    d = tempfile.mkdtemp(prefix="preset_nullstrap_")
    cf = os.path.join(d, "counts.tsv")
    mf = os.path.join(d, "meta.tsv")
    of = os.path.join(d, "out.tsv")
    sf = os.path.join(d, "run.R")
    np.savetxt(cf, counts.T, fmt="%d", delimiter="\t")
    pd.DataFrame({"condition": cond, "batch": batch}).to_csv(mf, sep="\t", index=False)
    script = f"""
        suppressMessages({{
            library(edgeR)
            library(NullstrapDE)
        }})
        cnt <- as.matrix(read.table("{cf}", sep="\\t"))
        meta <- read.table("{mf}", header=TRUE, sep="\\t")
        rownames(cnt) <- paste0("g", seq_len(nrow(cnt)) - 1)
        colnames(cnt) <- paste0("s", seq_len(ncol(cnt)))
        meta$condition <- factor(meta$condition, levels=c(0, 1))
        meta$batch <- factor(meta$batch)
        design <- if (length(unique(meta$batch)) > 1) model.matrix(~batch + condition, data=meta) else model.matrix(~condition, data=meta)
        dge <- DGEList(counts=cnt, group=meta$condition)
        keep <- filterByExpr(dge, design=design)
        dge <- dge[keep, , keep.lib.sizes=FALSE]
        dge <- calcNormFactors(dge)
        dge <- estimateDisp(dge, design)
        fit <- glmQLFit(dge, design)
        res <- Nullstrap_edgeR(dge=dge, design=design, fdrcutoff={fdr}, test="{test}")
        write.table(data.frame(gene=rownames(res)), "{of}", sep="\\t", row.names=FALSE, quote=FALSE)
    """
    open(sf, "w").write(script)
    try:
        r = subprocess.run([RSCRIPT, "--vanilla", sf], capture_output=True, text=True, timeout=R_TIMEOUT)
    except Exception:
        return None
    if r.returncode != 0 or not os.path.exists(of):
        warnings.warn("Nullstrap_edgeR failed: " + r.stderr[-800:])
        return None
    sig = pd.read_csv(of, sep="\t")
    call = np.zeros(counts.shape[1], bool)
    idx = [int(g[1:]) for g in sig.gene.astype(str) if g.startswith("g")]
    call[idx] = True
    return call


def benchmark(models, names, cfg, device, ns, batches_per_n, B, tau, qs, alphas, seed, deseq_cpus):
    dep = {name: Metrics(qs) for name in names}
    dep_n = {name: {n: Metrics(qs) for n in ns} for name in names}
    deseq = Metrics(alphas)
    edge = Metrics(alphas)
    deseq_n = {n: Metrics(alphas) for n in ns}
    edge_n = {n: Metrics(alphas) for n in ns}
    counter = 0

    for n in ns:
        c = PriorConfig(**vars(cfg))
        c.n_per_group = (n // 2, n // 2)
        print(f"\n=== n={n} ===")

        for bi in range(batches_per_n):
            set_seed(seed + counter)
            counter += 1
            d = generate_batch(c, B=B, device=device)

            # drop datasets with batch confounded with condition, for every method
            keep = torch.tensor(
                [design_ok(d["cond"][b].cpu().numpy(), d["batch"][b].cpu().numpy()) for b in range(B)],
                device=device,
            )
            n_skip = int(B - keep.sum().item())
            for m in (deseq, edge, deseq_n[n], edge_n[n], *dep.values(), *(dn[n] for dn in dep_n.values())):
                m.skipped += n_skip
            if n_skip == B:
                continue

            for name, model in zip(names, models):
                logits, _ = model(d["counts"], d["cond"], d["batch"])
                dep[name].update_preset(model, logits[keep], d["lfc_true"][keep], tau)
                dep_n[name][n].update_preset(model, logits[keep], d["lfc_true"][keep], tau)

            for b in range(B):
                if not keep[b]:
                    continue
                counts = d["counts"][b].cpu().numpy()
                cond = d["cond"][b].cpu().numpy()
                batch = d["batch"][b].cpu().numpy()
                truth = d["lfc_true"][b].cpu().numpy()

                r = run_deseq2(counts, cond, batch, tau, deseq_cpus)
                if r is None:
                    deseq.update_pvalue(None, truth, tau)
                    deseq_n[n].update_pvalue(None, truth, tau)
                else:
                    p = r["padj"].to_numpy(float)
                    lfc = r["log2FoldChange"].to_numpy(float)
                    deseq.update_pvalue(p, truth, tau, lfc)
                    deseq_n[n].update_pvalue(p, truth, tau, lfc)

                r = run_edger(counts, cond, batch, tau)
                if r is None:
                    edge.update_pvalue(None, truth, tau)
                    edge_n[n].update_pvalue(None, truth, tau)
                else:
                    p = r["FDR"].to_numpy(float)
                    lfc = r["logFC"].to_numpy(float)
                    edge.update_pvalue(p, truth, tau, lfc)
                    edge_n[n].update_pvalue(p, truth, tau, lfc)

            if (bi + 1) % max(1, batches_per_n // 4) == 0:
                print(f"  {bi+1}/{batches_per_n}", flush=True)

    return dep, dep_n, deseq, deseq_n, edge, edge_n


def curve_df(dep, deseq, edge, qs, alphas):
    rows = []
    for name, m in dep.items():
        s = m.summary()
        for q in qs:
            rows.append({"model": name, "threshold": q, "FDR": s[f"FDR@{q}"], "power": s[f"power@{q}"]})
    for name, m in [("PyDESeq2", deseq), ("edgeR", edge)]:
        s = m.summary()
        for a in alphas:
            rows.append({"model": name, "threshold": a, "FDR": s[f"FDR@{a}"], "power": s[f"power@{a}"]})
    return pd.DataFrame(rows)


def curve_n_df(dep_n, deseq_n, edge_n, qs, alphas):
    rows = []
    for name, per_n in dep_n.items():
        for n, m in per_n.items():
            s = m.summary()
            for q in qs:
                rows.append(
                    {"model": name, "n": n, "threshold": q, "FDR": s[f"FDR@{q}"], "power": s[f"power@{q}"]}
                )
    for name, src in [("PyDESeq2", deseq_n), ("edgeR", edge_n)]:
        for n, m in src.items():
            s = m.summary()
            for a in alphas:
                rows.append(
                    {"model": name, "n": n, "threshold": a, "FDR": s[f"FDR@{a}"], "power": s[f"power@{a}"]}
                )
    return pd.DataFrame(rows)


def plot_curve(df, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in df.model.unique():
        s = df[df.model == model].sort_values("FDR")
        ax.plot(s.FDR, s.power, marker="o", label=model)
        for _, r in s.iterrows():
            ax.annotate(
                f'{r["threshold"]:.2f}',
                (r.FDR, r.power),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel("Realized FDR")
    ax.set_ylabel("Power")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--output_dir", default=os.path.join(RESULTS, "benchmark"))
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--batches_per_n", type=int, default=16)
    ap.add_argument("--ns", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--a_min", type=float, default=1.5)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--deseq_cpus", type=int, default=4)
    ap.add_argument(
        "--qs",
        nargs="+",
        type=float,
        default=[
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.85,
            0.9,
            0.925,
            0.95,
            0.975,
            0.99,
        ],
    )
    ap.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001],
    )
    args = ap.parse_args()

    if len(args.ckpts) != len(args.names):
        raise ValueError("--ckpts and --names must match")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    models = []
    for name, path in zip(args.names, args.ckpts):
        model, ckpt = load_model(path, device)
        models.append(model)
        print(f"{name}: {path} step={ckpt.get('step','?')}")

    cfg = PriorConfig(G=args.G, K=args.K, a_min=args.a_min)
    qs = tuple(args.qs)
    alphas = tuple(args.alphas)

    dep, dep_n, deseq, deseq_n, edge, edge_n = benchmark(
        models,
        args.names,
        cfg,
        device,
        args.ns,
        args.batches_per_n,
        args.B,
        args.tau,
        qs,
        alphas,
        args.seed,
        args.deseq_cpus,
    )

    curve = curve_df(dep, deseq, edge, qs, alphas)
    curve_n = curve_n_df(dep_n, deseq_n, edge_n, qs, alphas)

    curve.to_csv(os.path.join(args.output_dir, "fdr_power_overall.csv"), index=False)
    curve_n.to_csv(os.path.join(args.output_dir, "fdr_power_by_n.csv"), index=False)
    plot_curve(curve, os.path.join(args.output_dir, "fdr_power_curve.png"), "FDR-Power Curve")

    for n in args.ns:
        plot_curve(
            curve_n[curve_n.n == n],
            os.path.join(args.output_dir, f"fdr_power_curve_n{n}.png"),
            f"FDR-Power Curve (n={n})",
        )

    print("\n=== Overall ===")
    for name, m in dep.items():
        s = m.summary()
        print(f"{name}: NLL={s['nll']:.4f} MAE={s['mae']:.4f} RMSE={s['rmse']:.4f}")
        for q in qs:
            print(f"  q={q:.3f} FDR={s[f'FDR@{q}']:.4f} power={s[f'power@{q}']:.4f}")

    for name, m in [("PyDESeq2", deseq), ("edgeR", edge)]:
        s = m.summary()
        print(
            f"{name}: MAE={s['mae']:.4f} RMSE={s['rmse']:.4f} failed={s['failed']}/{s['total']} skipped(unidentifiable)={s['skipped']}"
        )
        for a in alphas:
            print(f"  adj<{a:.3f} FDR={s[f'FDR@{a}']:.4f} power={s[f'power@{a}']:.4f}")

    with open(os.path.join(args.output_dir, "benchmark_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
