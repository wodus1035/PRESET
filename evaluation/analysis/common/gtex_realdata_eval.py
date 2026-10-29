"""GTEx tissue-pair evaluation against a large-sample edgeR reference, plus helpers used by the other scripts (donor
split, consensus_reference, ctx_matrix, preset_default).

Reference: n_ref donors per tissue, disjoint from the test pool. Test: --reps random n/2 vs n/2 subsets per n, scored by LFC
agreement with the reference and FDR and power over threshold grids; --null_tissue adds within-tissue random splits.

    python analysis/common/gtex_realdata_eval.py --ckpts CKPT --names PRESET --donor_split eval --gpu -1
"""

import argparse, gzip, os, sys, warnings
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
ANALYSIS = os.path.join(ROOT, "analysis")
for p_ in [ROOT] + [os.path.join(ANALYSIS, d) for d in sorted(os.listdir(ANALYSIS))]:
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from benchmark import load_model, run_deseq2, run_edger, run_limma_voom, run_wilcoxon

GTEX = os.path.join(os.environ.get("PRESET_DATA", os.path.join(ROOT, "data")), "gtex")
PAIRS = [
    ("whole_blood", "brain_cortex"),
    ("colon_transverse", "stomach"),
    ("heart_left_ventricle", "muscle_skeletal"),
]
# dense grids: one starting at 0.5 hides ranking power when P(DE) is compressed
QS = (
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
)
ALPHAS = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025, 0.01, 0.005, 0.001)


DONOR_SPLITS = (None, "all", "eval", "bank")


def donor_mask(samples, donor_split=None, split_path=None):
    """Mask over GTEx sample ids: None/"all" = every sample, "eval" = donors outside the training bank (used for all GTEx
    evaluations), "bank" = training donors."""
    if donor_split not in DONOR_SPLITS:
        raise ValueError(f"donor_split must be one of {DONOR_SPLITS}, got {donor_split!r}")
    if donor_split in (None, "all"):
        return np.ones(len(samples), bool)
    from real_background import donor_of, load_donor_split, DONOR_SPLIT_V3

    keep = set(load_donor_split(split_path or DONOR_SPLIT_V3)[donor_split])
    return np.array([d in keep for d in donor_of(samples)])


def load_tissue(name, donor_split=None, split_path=None, return_samples=False):
    """Ensembl ids and counts (genes x samples) [, sample ids] of the donors in donor_split."""
    df = pd.read_csv(
        os.path.join(GTEX, f"gene_reads_v10_{name}.gct.gz"), sep="\t", skiprows=2, compression="gzip"
    )
    genes = df["Name"].to_numpy()
    samples = np.array([c for c in df.columns if c not in ("Name", "Description")])
    m = donor_mask(samples, donor_split, split_path)
    X = df[samples[m]].to_numpy(np.int64)
    samples = samples[m]
    return (genes, X, samples) if return_samples else (genes, X)


def extra_baselines(cnt_gs, cond, tau):
    """limma-voom (treat) and Wilcoxon on a genes x samples matrix -> {name: (logFC, FDR)}."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nm, f in (("limma-voom", run_limma_voom), ("Wilcoxon", run_wilcoxon)):
            r = f(cnt_gs.T, cond, np.zeros_like(cond), tau=tau)
            if r is not None:
                out[nm] = (r["logFC"].to_numpy(float), np.nan_to_num(r["FDR"].to_numpy(float), nan=1.0))
    return out


def consensus_reference(cnt_gs, cond, deseq_cpus=8):
    """Labels from the agreement of edgeR, DESeq2, limma-voom and Wilcoxon: +1 = all FDR < 0.01 and edgeR |logFC| > 1,
    -1 = all FDR > 0.5 and |logFC| < 0.25."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        re = run_edger(cnt_gs.T, cond, np.zeros_like(cond), tau=0.0)
        rl = re["logFC"].to_numpy(float)
        rf = np.nan_to_num(re["FDR"].to_numpy(float), nan=1.0)
        fdrs = {"edgeR": rf}
        rd = run_deseq2(cnt_gs.T, cond, np.zeros_like(cond), tau=0.0, n_cpus=deseq_cpus)
        if rd is not None:
            fdrs["DESeq2"] = np.nan_to_num(rd["padj"].to_numpy(float), nan=1.0)
        rlv = run_limma_voom(cnt_gs.T, cond, np.zeros_like(cond), tau=0.0)
        if rlv is not None:
            fdrs["limma-voom"] = np.nan_to_num(rlv["FDR"].to_numpy(float), nan=1.0)
        rw = run_wilcoxon(cnt_gs.T, cond, np.zeros_like(cond), tau=0.0)
        fdrs["Wilcoxon"] = np.nan_to_num(rw["FDR"].to_numpy(float), nan=1.0)
    all_sig = np.all([f < 0.01 for f in fdrs.values()], axis=0)
    all_null = np.all([f > 0.5 for f in fdrs.values()], axis=0)
    labels = np.zeros(len(rl), int)
    labels[all_sig & (np.abs(rl) > 1)] = 1
    labels[all_null & (np.abs(rl) < 0.25)] = -1
    print(
        f"   consensus reference over {list(fdrs)}: positives {int((labels==1).sum())} (edgeR-only would give {int(((rf<0.01)&(np.abs(rl)>1)).sum())}), negatives {int((labels==-1).sum())}",
        flush=True,
    )
    return labels, rl, rf, fdrs


def ctx_matrix(X_full, cols, scored_idx, rule="auto", low_frac=0.25):
    """PRESET gene context for the subsample cols. rule: 'expressed' (scored rows), 'all' (any count), 'cap' (scored plus the
    top low-count rows, up to low_frac), 'auto' (expressed if n < 8, else cap, as preset.io.prepare). Returns (matrix, positions
    of scored_idx)."""
    sub = X_full[:, cols]
    n = len(cols)
    scored = np.zeros(len(X_full), bool)
    scored[scored_idx] = True
    any_count = sub.sum(1) > 0
    r = rule if rule != "auto" else ("expressed" if n < 8 else "cap")
    if r == "expressed":
        keep = scored
    elif r == "all":
        keep = any_count | scored
    else:
        tot = sub.sum(1)
        low = np.where(any_count & ~scored)[0]
        k = min(len(low), int(low_frac * scored.sum()))
        keep = scored.copy()
        keep[low[np.argsort(-tot[low])[:k]]] = True
    rows = np.where(keep)[0]
    return sub[rows], np.searchsorted(rows, scored_idx)


def preset_default(counts_ge, cond, model, **kw):
    """preset.analyze on the matrix the other methods get (its own gene filter and context). p_de and ranks are NaN for
    genes PRESET does not report."""
    import preset

    ra = preset.analyze(pd.DataFrame(np.asarray(counts_ge).T), cond, model=model, **kw)
    ra.loc[
        ~ra.reported.to_numpy(bool),
        [c for c in ("p_de", "p_positive", "rank", "bayes_fdr", "expected_false_in_list") if c in ra],
    ] = np.nan
    return ra


def fdr_power(score_low_is_de, labels, thresholds, less_than=True):
    """labels: +1 positive, -1 negative, 0 ignore. Returns rows of (thr, FDR, power, frac_called)."""
    pos = labels == 1
    neg = labels == -1
    rows = []
    for t in thresholds:
        sel = score_low_is_de < t if less_than else score_low_is_de > t
        tp = (sel & pos).sum()
        fp = (sel & neg).sum()
        rows.append(
            {
                "threshold": t,
                "FDR": fp / max(1, tp + fp),
                "power": tp / max(1, pos.sum()),
                "frac_called": sel.mean(),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--n_ref", type=int, default=60)
    ap.add_argument("--ns", nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument(
        "--max_genes", type=int, default=6000, help="random subsample of expressed genes per pair (runtime)"
    )
    ap.add_argument("--deseq_cpus", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null_tissue", default="lung")
    ap.add_argument(
        "--donor_split",
        choices=["all", "eval", "bank"],
        default="all",
        help="'eval': donors outside the training bank; 'all': every donor",
    )
    ap.add_argument("--output_dir", default=os.path.join(RESULTS, "gtex_real"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"donor_split = {args.donor_split}", flush=True)
    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    models = {nm: load_model(p, dev)[0] for nm, p in zip(args.names, args.ckpts)}

    # counts genes x n
    def run_preset(model, counts, cond):
        c = torch.tensor(counts.T, dtype=torch.float32, device=dev)[None]
        cd = torch.tensor(cond, dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            logits, _ = model(c, cd, torch.zeros_like(cd))
            return (
                model.head.mean(logits)[0].cpu().numpy(),
                model.head.prob_abs_gt(logits, args.tau)[0].cpu().numpy(),
            )

    curve_rows, lfc_rows, null_rows = [], [], []
    for A, B in PAIRS:
        gA, XA = load_tissue(A, args.donor_split)
        gB, XB = load_tissue(B, args.donor_split)
        assert (gA == gB).all()
        pa = rng.permutation(XA.shape[1])
        pb = rng.permutation(XB.shape[1])
        refA, poolA = XA[:, pa[: args.n_ref]], XA[:, pa[args.n_ref :]]
        refB, poolB = XB[:, pb[: args.n_ref]], XB[:, pb[args.n_ref :]]
        ref = np.hstack([refA, refB])
        expressed = (ref >= args.min_count).sum(1) >= ref.shape[1] // 4
        idx = np.where(expressed)[0]
        if len(idx) > args.max_genes:
            idx = np.sort(rng.choice(idx, args.max_genes, replace=False))
        print(
            f"\n### {A} vs {B}: ref {refA.shape[1]}+{refB.shape[1]}, pool {poolA.shape[1]}+{poolB.shape[1]}, genes {len(idx)}",
            flush=True,
        )
        cond_ref = np.array([0] * refA.shape[1] + [1] * refB.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_edger(ref[idx].T, cond_ref, np.zeros_like(cond_ref), tau=0.0)
        ref_lfc = r["logFC"].to_numpy(float)
        ref_fdr = r["FDR"].to_numpy(float)
        labels = np.zeros(len(idx), int)
        labels[(ref_fdr < 0.01) & (np.abs(ref_lfc) > 1)] = 1
        labels[np.abs(ref_lfc) < 0.25] = -1
        conf = (ref_fdr < 0.05) & (np.abs(ref_lfc) > 0.5)
        print(
            f"  reference: positives {int((labels==1).sum())}, negatives {int((labels==-1).sum())}, frac |LFC|>1 = {(np.abs(ref_lfc)>1).mean():.2f}, frac |LFC|>0.5 = {(np.abs(ref_lfc)>0.5).mean():.2f}",
            flush=True,
        )

        for n in args.ns:
            for rep in range(args.reps):
                sa = rng.choice(poolA.shape[1], n // 2, replace=False)
                sb = rng.choice(poolB.shape[1], n // 2, replace=False)
                cnt = np.hstack([poolA[idx][:, sa], poolB[idx][:, sb]])
                cond = np.array([0] * (n // 2) + [1] * (n // 2))
                ests = {}
                for nm, m in models.items():
                    lfc, pde = run_preset(m, cnt, cond)
                    ests[nm] = lfc
                    for row in fdr_power(pde, labels, QS, less_than=False):
                        curve_rows.append({"pair": f"{A}_vs_{B}", "n": n, "rep": rep, "model": nm, **row})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    re = run_edger(cnt.T, cond, np.zeros_like(cond), tau=args.tau)
                    rd = run_deseq2(cnt.T, cond, np.zeros_like(cond), tau=args.tau, n_cpus=args.deseq_cpus)
                if re is not None:
                    ests["edgeR"] = re["logFC"].to_numpy(float)
                    for row in fdr_power(np.nan_to_num(re["FDR"].to_numpy(float), nan=1.0), labels, ALPHAS):
                        curve_rows.append(
                            {"pair": f"{A}_vs_{B}", "n": n, "rep": rep, "model": "edgeR", **row}
                        )
                if rd is not None:
                    ests["PyDESeq2"] = rd["log2FoldChange"].to_numpy(float)
                    for row in fdr_power(np.nan_to_num(rd["padj"].to_numpy(float), nan=1.0), labels, ALPHAS):
                        curve_rows.append(
                            {"pair": f"{A}_vs_{B}", "n": n, "rep": rep, "model": "PyDESeq2", **row}
                        )
                for nm, e in ests.items():
                    ok = conf & np.isfinite(e)
                    sl, ic = np.polyfit(ref_lfc[ok], e[ok], 1)
                    lfc_rows.append(
                        {
                            "pair": f"{A}_vs_{B}",
                            "n": n,
                            "rep": rep,
                            "model": nm,
                            "slope_vs_ref": sl,
                            "intercept": ic,
                            "corr_vs_ref": np.corrcoef(ref_lfc[ok], e[ok])[0, 1],
                            "rmse_vs_ref": np.sqrt(np.mean((e[ok] - ref_lfc[ok]) ** 2)),
                            "n_genes": int(ok.sum()),
                        }
                    )
            print(f"  n={n} done", flush=True)

    # null check: random split within one tissue
    g, X = load_tissue(args.null_tissue, args.donor_split)
    expressed = (X >= args.min_count).sum(1) >= X.shape[1] // 4
    idx = np.where(expressed)[0]
    if len(idx) > args.max_genes:
        idx = np.sort(rng.choice(idx, args.max_genes, replace=False))
    for n in args.ns:
        for rep in range(args.reps):
            s = rng.choice(X.shape[1], n, replace=False)
            cnt = X[idx][:, s]
            cond = np.array([0] * (n // 2) + [1] * (n // 2))
            for nm, m in models.items():
                lfc, pde = run_preset(m, cnt, cond)
                null_rows.append(
                    {
                        "tissue": args.null_tissue,
                        "n": n,
                        "rep": rep,
                        "model": nm,
                        "frac_called_q0.9": (pde > 0.9).mean(),
                        "mean_abs_lfc": np.abs(lfc).mean(),
                    }
                )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                re = run_edger(cnt.T, cond, np.zeros_like(cond), tau=args.tau)
            if re is not None:
                null_rows.append(
                    {
                        "tissue": args.null_tissue,
                        "n": n,
                        "rep": rep,
                        "model": "edgeR",
                        "frac_called_q0.9": (np.nan_to_num(re["FDR"].to_numpy(float), nan=1) < 0.05).mean(),
                        "mean_abs_lfc": np.nanmean(np.abs(re["logFC"].to_numpy(float))),
                    }
                )
    print("null check done", flush=True)

    curve = pd.DataFrame(curve_rows)
    lfc = pd.DataFrame(lfc_rows)
    null = pd.DataFrame(null_rows)
    curve.to_csv(os.path.join(args.output_dir, "fdr_power_curves.csv"), index=False)
    lfc.to_csv(os.path.join(args.output_dir, "lfc_vs_reference.csv"), index=False)
    null.to_csv(os.path.join(args.output_dir, "null_split.csv"), index=False)

    print("\n=== LFC vs reference (mean over reps): slope / corr ===")
    print(
        lfc.groupby(["pair", "n", "model"])[["slope_vs_ref", "corr_vs_ref", "rmse_vs_ref"]]
        .mean()
        .round(3)
        .to_string()
    )
    print(
        "\n=== power @ realized FDR <= 0.05 (per rep: best threshold with FDR<=0.05; then mean over reps) ==="
    )

    def best(g):
        ok = g[g.FDR <= 0.05]
        return ok.power.max() if len(ok) else np.nan

    pw = (
        curve.groupby(["pair", "n", "model", "rep"])
        .apply(best)
        .groupby(["pair", "n", "model"])
        .mean()
        .unstack("model")
    )
    print(pw.round(3).to_string())
    print("\n=== calibration: realized FDR at PFN q>0.9 / classical adj<0.05 (mean over reps) ===")
    cal = curve[
        ((curve.model.isin(args.names)) & ((curve.threshold - 0.9).abs() < 1e-9))
        | ((~curve.model.isin(args.names)) & ((curve.threshold - 0.05).abs() < 1e-9))
    ]
    print(
        cal.groupby(["pair", "n", "model"])[["FDR", "power", "frac_called"]]
        .mean()
        .unstack("model")
        .round(3)
        .to_string()
    )
    print("\n=== null split (same tissue): fraction called ===")
    print(null.groupby(["n", "model"])[["frac_called_q0.9", "mean_abs_lfc"]].mean().round(4).to_string())
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
