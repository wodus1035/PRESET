"""Rules shared by the evaluation scripts for comparing methods.

1. Same estimand: each method tests PRESET's threshold |LFC| > tau (edgeR glmTreat, DESeq2 lfcThreshold, limma treat);
   Wilcoxon has none and runs at tau=0.
2. Reference: the consensus of edgeR, limma-voom and Wilcoxon at tau=0 on samples not used for testing, never a single
   method that is also scored.
3. Same genes: one gene set for all methods; an NA never removes a gene.
4. NA: FDR NA is not called and logFC NA is 0; in rankings NA goes last.
5. Ties: broken by the method's own |logFC|, then mean count. EBSeq (PPDE = 1 for hundreds of genes) and PRESET (p_de
   near 0 and 1) depend on this.
"""

import numpy as np


def expressed_mask(counts_gs, min_count=10, min_frac=0.25):
    """counts_gs: genes x samples. Genes passed to every method: all with a count (TXDE_PREFILTER=none, default) or
    >= min_count in >= min_frac of samples ("shared")."""
    import os

    if os.environ.get("TXDE_PREFILTER", "none").strip().lower() == "none":
        return np.asarray(counts_gs).sum(1) > 0
    return (counts_gs >= min_count).sum(1) >= max(1, int(np.ceil(min_frac * counts_gs.shape[1])))


def rank_score(score, lfc, mean_count, higher_is_better=True):
    """Gene order, best first (rules 4 and 5): by score, then the method's own |lfc|, then mean_count."""
    s = np.asarray(score, float)
    l = np.abs(np.nan_to_num(np.asarray(lfc, float), nan=0.0))
    mc = np.asarray(mean_count, float)
    if not higher_is_better:
        s = -s
    bad = ~np.isfinite(s)
    s = np.where(bad, -np.inf, s)
    # lexsort uses the last key as primary
    return np.lexsort((-mc, -l, -s))


def calls_from_fdr(fdr, alpha=0.05):
    """Rule 4 for calls: NA is not a call."""
    return np.nan_to_num(np.asarray(fdr, float), nan=1.0) < alpha


MIN_POSITIVES = 30  # fewer labeled positives: no pAUPRC for that cell


def consensus_reference(stats, min_methods=2, pos_fdr=0.05, pos_lfc=1.0, neg_fdr=0.5, neg_lfc=0.25):
    """stats: {method: (fdr, logFC)} from the reference samples at tau=0 -> (pos, neg). Positive: FDR < pos_fdr and
    |log2FC| > pos_lfc in >= min_methods methods; negative: FDR > neg_fdr and |log2FC| < neg_lfc in all. Two of three because
    Wilcoxon p-values are discrete (all three at FDR < 0.01 left 4 positives in GTEx lung)."""
    pos_votes = None
    neg = None
    for _, (fdr, lfc) in stats.items():
        f = np.nan_to_num(np.asarray(fdr, float), nan=1.0)
        L = np.abs(np.nan_to_num(np.asarray(lfc, float), nan=0.0))
        p = ((f < pos_fdr) & (L > pos_lfc)).astype(int)
        n = (f > neg_fdr) & (L < neg_lfc)
        pos_votes = p if pos_votes is None else (pos_votes + p)
        neg = n if neg is None else (neg & n)
    return (pos_votes >= min_methods), neg


def sweep_power(order, pos, neg, alpha=0.05):
    """Max recall of `pos` over list lengths whose realized FDR among labeled genes is <= alpha."""
    p = np.asarray(pos)[order]
    n = np.asarray(neg)[order]
    tp = np.cumsum(p)
    fp = np.cumsum(n)
    lab = tp + fp
    fdr = np.divide(fp, np.maximum(lab, 1), where=lab > 0)
    fdr = np.where(lab > 0, fdr, 0.0)
    ok = fdr <= alpha
    return float(tp[ok].max() / max(1, p.sum())) if ok.any() else 0.0


def top_k_precision(order, pos, neg, k):
    """Precision among the labelled genes of the top k."""
    top = order[:k]
    lab = (np.asarray(pos) | np.asarray(neg))[top]
    return float(np.asarray(pos)[top].sum() / max(1, lab.sum()))


def partial_auprc(order, pos, neg, rmax=0.2, min_pos=None):
    """Area under the PR curve up to recall rmax on labeled genes, divided by rmax; NaN with < min_pos positives."""
    mp = MIN_POSITIVES if min_pos is None else min_pos
    p = np.asarray(pos)[order]
    n = np.asarray(neg)[order]
    keep = p | n
    p = p[keep]
    if p.sum() < mp:
        return np.nan
    tp = np.cumsum(p)
    k = np.arange(1, len(p) + 1)
    rec = tp / p.sum()
    prec = tp / k
    m = rec <= rmax
    if m.sum() < 2:
        return np.nan
    return float(np.trapz(prec[m], rec[m]) / rmax)


def mcc(call, pos, neg):
    lab = np.asarray(pos) | np.asarray(neg)
    c = np.asarray(call)[lab]
    p = np.asarray(pos)[lab]
    tp = float((c & p).sum())
    fp = float((c & ~p).sum())
    fn = float((~c & p).sum())
    tn = float((~c & ~p).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else np.nan


# per-gene scores are stored so a new scoring rule needs only a re-scoring pass
GENE_SCORE_COLUMNS = (
    "dataset",
    "n",
    "rep",
    "method",
    "gene",
    "score",
    "higher_is_better",
    "lfc",
    "mean_count",
)


def gene_score_rows(dataset, n, rep, method, genes, score, lfc, mean_count, higher_is_better):
    """One row per gene; `score` on the method's own scale, its direction in `higher_is_better`."""
    import numpy as _np

    g = _np.asarray(genes)
    sc = _np.asarray(score, float)
    lf = _np.asarray(lfc, float)
    mc = _np.asarray(mean_count, float)
    return [
        {
            "dataset": dataset,
            "n": int(n),
            "rep": int(rep),
            "method": method,
            "gene": str(g[i]),
            "score": float(sc[i]),
            "higher_is_better": bool(higher_is_better),
            "lfc": float(lf[i]),
            "mean_count": float(mc[i]),
        }
        for i in range(len(g))
    ]


# Wilcoxon and EBSeq are supplement only: tied statistics at small n (Wilcoxon p-values across 93-100% of the top 500)
MAIN_COMPARATORS = ("edgeR TREAT", "DESeq2 lfcThr", "limma treat", "apeglm")
SUPPLEMENT_ONLY = ("Wilcoxon", "EBSeq")
