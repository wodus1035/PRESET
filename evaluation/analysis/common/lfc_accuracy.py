"""LFC accuracy on datasets drawn from the training prior: PRESET posterior mean vs edgeR, DESeq2 MLE and apeglm against the
true lfc. Also holds the DESeq2 MLE + apeglm helpers of the real-data scripts. main() needs the prior simulator
(training/prior.py on PYTHONPATH) except with --plot_only. Writes lfc_predictions.parquet, lfc_accuracy_by_n.csv and two plots.

    python analysis/common/lfc_accuracy.py --ckpts CKPT --names PRESET --gpu 0 --batches_per_n 4 --B 4
"""

import argparse, os, sys, warnings
import numpy as np, pandas as pd, torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
ANALYSIS = os.path.join(ROOT, "analysis")
for p_ in [ROOT] + [os.path.join(ANALYSIS, d) for d in sorted(os.listdir(ANALYSIS))]:
    if p_ not in sys.path:
        sys.path.insert(0, p_)
try:
    from prior import PriorConfig, generate_batch  # prior simulator (training/prior.py)
except ImportError:
    PriorConfig = generate_batch = None
from benchmark import load_model, run_edger, design_ok, set_seed
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# PRESET models first, then the other methods
COLORS = {
    "__preset__": ["#2a78d6", "#1baf7a"],
    "edgeR": "#eb6834",
    "PyDESeq2": "#eda100",
    "PyDESeq2-apeglm": "#e87ba4",
}


def run_deseq2_mle_and_shrunk(counts, cond, batch, tau, n_cpus, cooks=False):
    """As benchmark.run_deseq2, plus the apeglm-shrunk LFC. cooks=True: DESeq2 defaults (refitting acts only with >= 7
    replicates per group). Returns (mle, shrunk) or (None, None)."""
    import benchmark as _bm

    # Bioconductor DESeq2 + apeglm on the genes passed in
    if _bm.DESEQ2_IMPL == "r":
        r = _bm.run_deseq2_r(counts, cond, batch, tau=tau, cooks=cooks, shrink=True)
        return (None, None) if r is None else (r.log2FoldChange.to_numpy(float), r.lfc_apeglm.to_numpy(float))
    counts = np.asarray(counts, np.int64)
    cond = np.asarray(cond, int)
    batch = np.asarray(batch, int)
    n, G = counts.shape
    samples = [f"s{i}" for i in range(n)]
    genes = [f"g{i}" for i in range(G)]
    meta = {"condition": pd.Categorical(cond.astype(str), categories=["0", "1"])}
    design = "~condition"
    if len(np.unique(batch)) > 1:
        if not design_ok(cond, batch):
            return None, None
        # any number of levels (pairs/blocks)
        meta["batch"] = pd.Categorical(batch.astype(str), categories=[str(v) for v in np.unique(batch)])
        design = "~batch + condition"
    try:
        cdf = pd.DataFrame(counts, index=samples, columns=genes)
        mdf = pd.DataFrame(meta, index=samples)
        # pydeseq2 <= 0.4 takes n_cpus; >= 0.5 takes an `inference` object
        try:
            dds = DeseqDataSet(
                counts=cdf, metadata=mdf, design=design, refit_cooks=cooks, n_cpus=n_cpus, quiet=True
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
                counts=cdf, metadata=mdf, design=design, refit_cooks=cooks, inference=inf, quiet=True
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
        mle = ds.results_df.reindex(genes).log2FoldChange.to_numpy(float)
        coef = [c for c in dds.varm["LFC"].columns if c.startswith("condition")][0]
        ds.lfc_shrink(coeff=coef)
        shrunk = ds.results_df.reindex(genes).log2FoldChange.to_numpy(float)
        return mle, shrunk
    except Exception as e:
        print(f"[PyDESeq2 failed] {type(e).__name__}: {e}")
        return None, None


def mle_shrunk_scored(counts, cond, batch, ev, tau, n_cpus=1, cooks=True):
    """(mle, apeglm) of length G, NaN outside ev; R mode fits all genes, py mode counts[:, ev]."""
    import benchmark as _bm

    G = counts.shape[1]
    ev = np.asarray(ev, bool)
    mle = np.full(G, np.nan)
    shrunk = np.full(G, np.nan)
    if not ev.any():
        return mle, shrunk
    if _bm.DESEQ2_IMPL == "r":
        r = _bm.run_deseq2_r(counts, cond, batch, tau=tau, cooks=cooks, shrink=True)
        if r is None:
            return None, None
        mle = np.where(ev, r.log2FoldChange.to_numpy(float), np.nan)
        shrunk = np.where(ev, r.lfc_apeglm.to_numpy(float), np.nan)
        return mle, shrunk
    m_, s_ = run_deseq2_mle_and_shrunk(np.asarray(counts)[:, ev], cond, batch, tau, n_cpus, cooks=cooks)
    if m_ is None:
        return None, None
    mle[ev] = m_
    shrunk[ev] = s_
    return mle, shrunk


def _baselines(item, tau, deseq_cpus, official=False):
    """edgeR and DESeq2 for one dataset (joblib worker); deterministic, so --jobs does not change rows."""
    import warnings as _w

    _w.simplefilter("ignore")
    df, counts, cond, batch = item
    # official=True: user's guide edgeR workflow; all methods scored on filterByExpr genes
    r = run_edger(counts, cond, batch, tau, filter_expr=official, robust=official)
    el = r["logFC"].to_numpy(float) if r is not None else np.full(counts.shape[1], np.nan)
    ev = np.isfinite(el) if official else np.ones(counts.shape[1], bool)
    df["edgeR"] = el
    mle = np.full(len(ev), np.nan)
    shrunk = np.full(len(ev), np.nan)
    if ev.any():
        m_, s_ = mle_shrunk_scored(counts, cond, batch, ev, tau, deseq_cpus, cooks=official)
        if m_ is not None:
            mle, shrunk = m_, s_
    df["PyDESeq2"] = mle
    df["PyDESeq2-apeglm"] = shrunk
    if official:
        for c in df.columns:
            if c not in ("n", "dataset", "gene", "lfc_true", "edgeR", "PyDESeq2", "PyDESeq2-apeglm"):
                df.loc[~ev, c] = np.nan
    df["scored"] = ev
    return df


@torch.no_grad()
def collect(models, names, cfg, device, ns, batches_per_n, B, tau, seed, deseq_cpus, jobs=1, official=False):
    # serial: data and PRESET forward pass; parallel: CPU baselines per dataset
    items = []
    counter = 0
    ds_id = 0
    skipped = 0
    for n in ns:
        c = PriorConfig(**vars(cfg))
        c.n_per_group = (n // 2, n // 2)
        for bi in range(batches_per_n):
            set_seed(seed + counter)
            counter += 1
            d = generate_batch(c, B=B, device=device)
            preds = {
                nm: m.head.mean(m(d["counts"], d["cond"], d["batch"])[0]).cpu().numpy()
                for nm, m in zip(names, models)
            }
            for b in range(B):
                cond = d["cond"][b].cpu().numpy()
                batch = d["batch"][b].cpu().numpy()
                if not design_ok(cond, batch):
                    skipped += 1
                    continue
                counts = d["counts"][b].cpu().numpy()
                truth = d["lfc_true"][b].cpu().numpy()
                G = len(truth)
                df = pd.DataFrame({"n": n, "dataset": ds_id, "gene": np.arange(G), "lfc_true": truth})
                for nm in names:
                    df[nm] = preds[nm][b]
                items.append((df, counts, cond, batch))
                ds_id += 1
        print(f"=== n={n} generated ({ds_id} datasets so far)", flush=True)
    print(
        f"datasets kept={ds_id} skipped(unidentifiable)={skipped}; baselines over {jobs} workers", flush=True
    )
    if jobs > 1:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=jobs, verbose=5)(
            delayed(_baselines)(it, tau, deseq_cpus, official) for it in items
        )
    else:
        rows = [_baselines(it, tau, deseq_cpus, official) for it in items]
    return pd.concat(rows, ignore_index=True)


def accuracy_table(df, methods):
    out = []
    for n, g in df.groupby("n"):
        for m in methods:
            ok = np.isfinite(g[m])
            t = g.lfc_true[ok].to_numpy()
            p = g[m][ok].to_numpy()
            e = p - t
            out.append(
                {
                    "n": n,
                    "method": m,
                    "rmse": np.sqrt(np.mean(e**2)),
                    "mae": np.mean(np.abs(e)),
                    "pearson_r": pearsonr(t, p)[0],
                    "spearman_rho": spearmanr(t, p)[0],
                    "coverage": ok.mean(),
                    "n_genes": int(ok.sum()),
                }
            )
    return pd.DataFrame(out)


def plot_scatter(df, methods, colors, path, lim=4.0):
    fig, axes = plt.subplots(1, len(methods), figsize=(3.6 * len(methods), 3.9), sharex=True, sharey=True)
    for ax, m in zip(axes, methods):
        ok = np.isfinite(df[m])
        t = df.lfc_true[ok].to_numpy()
        p = df[m][ok].to_numpy()
        ax.hexbin(
            t, p, gridsize=60, extent=(-lim, lim, -lim, lim), bins="log", cmap="Blues", mincnt=1, linewidths=0
        )
        ax.plot([-lim, lim], [-lim, lim], color="#8a8a86", lw=1, ls="--", zorder=3)
        e = p - t
        ax.text(
            0.03,
            0.97,
            f"r = {pearsonr(t, p)[0]:.3f}\nRMSE = {np.sqrt(np.mean(e**2)):.3f}\nMAE = {np.mean(np.abs(e)):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )
        ax.set_title(m, color=colors[m], fontsize=11, fontweight="bold")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("true log2FC", fontsize=9)
    axes[0].set_ylabel("estimated log2FC", fontsize=9)
    ns = sorted(df.n.unique())
    fig.suptitle(
        f"LFC estimation, ID prior, pooled n={ns[0]}–{ns[-1]} ({df.dataset.nunique()} datasets × {df.gene.nunique()} genes; axes clipped at ±{lim:g})",
        fontsize=10,
        y=1.0,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_by_n(tab, methods, colors, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, metric, label in zip(
        axes, ["rmse", "pearson_r"], ["RMSE (log2FC)", "Pearson r (true vs estimated)"]
    ):
        # marker and line style too, not color alone
        for i, m in enumerate(methods):
            s = tab[tab.method == m].sort_values("n")
            ax.plot(
                s.n,
                s[metric],
                color=colors[m],
                lw=2,
                marker="os"[i % 2],
                ms=5.5,
                ls=["-", "--"][i % 2],
                markerfacecolor="white" if i % 2 else colors[m],
                markeredgewidth=1.6,
                label=m,
            )
        ax.set_xlabel("samples per dataset (n)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xticks(sorted(tab.n.unique()))
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.08)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h, l, frameon=False, fontsize=8.5, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 0.95)
    )
    fig.suptitle("LFC estimation accuracy by n (ID prior, identifiable datasets only)", fontsize=10, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--output_dir", default=os.path.join(RESULTS, "lfc_accuracy"))
    ap.add_argument("--G", type=int, default=2000)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--a_min", type=float, default=1.5)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--ns", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16])
    ap.add_argument("--batches_per_n", type=int, default=16)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--deseq_cpus", type=int, default=8)
    ap.add_argument(
        "--edger_official",
        action="store_true",
        help="edgeR filterByExpr + robust; all methods scored on the filterByExpr genes",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="joblib workers for the CPU baselines (one dataset per task); use with --deseq_cpus 1",
    )
    ap.add_argument(
        "--plot_only", action="store_true", help="re-plot from an existing lfc_predictions.parquet"
    )
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pq = os.path.join(args.output_dir, "lfc_predictions.parquet")

    if args.plot_only:
        df = pd.read_parquet(pq)
    else:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        models = []
        for nm, p in zip(args.names, args.ckpts):
            m, ck = load_model(p, device)
            models.append(m)
            print(f"{nm}: {p} step={ck.get('step','?')}")
        cfg = PriorConfig(G=args.G, K=args.K, a_min=args.a_min)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = collect(
                models,
                args.names,
                cfg,
                device,
                args.ns,
                args.batches_per_n,
                args.B,
                args.tau,
                args.seed,
                args.deseq_cpus,
                args.jobs,
                args.edger_official,
            )
        df.to_parquet(pq, index=False)

    methods = list(args.names) + ["edgeR", "PyDESeq2", "PyDESeq2-apeglm"]
    colors = {nm: c for nm, c in zip(args.names, COLORS["__preset__"])}
    colors.update({k: v for k, v in COLORS.items() if k != "__preset__"})
    tab = accuracy_table(df, methods)
    tab.to_csv(os.path.join(args.output_dir, "lfc_accuracy_by_n.csv"), index=False)
    print("\n", tab.round(3).to_string(index=False))
    plot_scatter(df, methods, colors, os.path.join(args.output_dir, "lfc_scatter.png"))
    plot_by_n(tab, methods, colors, os.path.join(args.output_dir, "lfc_rmse_corr_by_n.png"))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
