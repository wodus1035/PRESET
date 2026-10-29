"""LFC accuracy against MAQC TaqMan RT-PCR (GSE5350) on the SEQC samples, a truth that no DE method helped define.
  AB   A (UHRR) vs B (HBRR): truth = TaqMan log2(B/A), median |log2FC| 1.33
  DC   D vs C, the two mixtures (C = 3:1 A:B, D = 1:3 A:B): truth from the mixing design, median |log2FC| 0.63
PRESET reports P(|log2FC| > 0.5), so edgeR glmTreat(0.5) and DESeq2 lfcThreshold = 0.5 run next to the plain tests, and apeglm
next to the posterior mean. MCC is the headline metric of the Quartet/MAQC benchmark (Nat Commun 2024, Fig. 8d) against this
TaqMan reference. Truth labels: positive |true| > 1, negative |true| < 0.25.
The five libraries per sample are technical replicates of one RNA pool, so power is optimistic for every method.
    python analysis/seqc/seqc_taqman_eval.py [--model <weights>] --tag v4a_pkg --official --gpu -1 --ns 3 --jobs 8
Writes results/seqc_taqman_<tag>.csv and _genes.csv.
"""

import os, sys, argparse, itertools, numpy as np, pandas as pd, torch, warnings

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
ANALYSIS = os.path.join(ROOT, "analysis")
for p_ in [ROOT] + [os.path.join(ANALYSIS, d) for d in sorted(os.listdir(ANALYSIS))]:
    if p_ not in sys.path:
        sys.path.insert(0, p_)
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
import preset
from benchmark import run_edger, run_deseq2, run_limma_voom, run_ebseq, run_deseq2_scored, universe_mask
from lfc_accuracy import run_deseq2_mle_and_shrunk, mle_shrunk_scored
from gtex_realdata_eval import ctx_matrix

D = os.path.join(os.environ.get("PRESET_DATA", os.path.join(ROOT, "data")), "seqc")
CONTRASTS = {"AB": ("A", "B", "lfc_BA"), "DC": ("C", "D", "lfc_DC_mix")}  # (control, case, truth column)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--tag", default="v3")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--ns", type=int, nargs="+", default=[3, 5])
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument(
    "--official",
    action="store_true",
    help="official workflows: edgeR filterByExpr + robust, limma filterByExpr, DESeq2 Cook's on",
)
ap.add_argument("--max_reps", type=int, default=8)
ap.add_argument("--cpus", type=int, default=1)
ap.add_argument("--jobs", type=int, default=8, help="parallel workers (R methods are single-threaded)")
a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 else "cpu")
_M = {}


def model():
    """One model per worker process."""
    if "m" not in _M:
        torch.set_num_threads(1)
        _M["m"], _, meta = preset.load_model(a.model, dev)
        print(f"  worker {os.getpid()}: {meta['head']} step {meta['step']} on {dev}", flush=True)
    return _M["m"]


_, _, meta = preset.load_model(a.model, "cpu")
print(f"model {meta['weights']} head={meta['head']} step={meta['step']}; {a.jobs} workers", flush=True)

C = pd.read_csv(os.path.join(D, "seqc_gene_counts_ABCD.tsv"), sep="\t", index_col=0)
T = pd.read_csv(os.path.join(D, "seqc_taqman_truth.tsv"), sep="\t", dtype={"EntrezID": str}).set_index(
    "EntrezID"
)
X = C.to_numpy(np.int64)
genes = C.index.to_numpy().astype(str)
libs = C.columns.to_numpy()
print(f"counts {X.shape[0]} genes x {X.shape[1]} libraries; TaqMan truth rows {len(T)}", flush=True)
rows, grows = [], []


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0]) if len(x) > 5 and x.std() > 0 and y.std() > 0 else np.nan


def sweep_power(score, pos, neg):
    """Max power over thresholds with realized FDR <= 0.05 among labelled genes."""
    o = np.argsort(-score)
    p = pos[o]
    n = neg[o]
    tp = np.cumsum(p)
    fp = np.cumsum(n)
    fdr = fp / np.maximum(tp + fp, 1)
    ok = fdr <= 0.05
    return float((tp[ok] / max(1, pos.sum())).max()) if ok.any() else 0.0


def methods(sub, cond, E):
    """name -> (ranking score, LFC estimate, call at the method's own cut-off or None)."""
    from gtex_realdata_eval import preset_default  # package default on the tools' matrix

    r = preset_default(sub[E], cond, model(), tau=a.tau, decoy=False)
    from preset.posterior import calls as dcalls

    out = {
        "PRESET": (r.p_de.to_numpy(), r.lfc_mean.to_numpy(), (r.p_de > 0.9).to_numpy()),
        "PRESET_bfdr": (r.p_de.to_numpy(), r.lfc_mean.to_numpy(), dcalls(r, fdr=0.05, rule="bfdr")),
    }
    b = np.zeros_like(cond)
    for tau, tag in ((0.0, "edgeR_p"), (0.5, "edgeR_TREAT05")):
        e = run_edger(sub[E].T, cond, b, tau=tau, filter_expr=a.official, robust=a.official)
        if e is not None:
            out[tag] = (
                -np.nan_to_num(e.PValue.to_numpy(float), nan=1.0),
                e.logFC.to_numpy(float),
                np.nan_to_num(e.FDR.to_numpy(float), nan=1.0) < 0.05,
            )
    for tau, tag in ((0.0, "DESeq2_p"), (0.5, "DESeq2_lfc05")):
        d_ = run_deseq2_scored(sub[E].T, cond, b, np.ones(len(E), bool), tau=tau, cooks=a.official)
        if d_ is not None:
            out[tag] = (
                -np.nan_to_num(d_.pvalue.to_numpy(float), nan=1.0),
                d_.log2FoldChange.to_numpy(float),
                np.nan_to_num(d_.padj.to_numpy(float), nan=1.0) < 0.05,
            )
    for tau, tag in ((0.0, "limma_p"), (0.5, "limma_treat05")):
        l_ = run_limma_voom(sub[E].T, cond, b, tau=tau, filter_expr=a.official)
        if l_ is not None:
            out[tag] = (
                -np.nan_to_num(l_.PValue.to_numpy(float), nan=1.0),
                l_.logFC.to_numpy(float),
                np.nan_to_num(l_.FDR.to_numpy(float), nan=1.0) < 0.05,
            )
    eb = run_ebseq(sub[E].T, cond)
    if eb is not None:
        out["EBSeq"] = (eb.PPDE.to_numpy(float), eb.logFC.to_numpy(float), (eb.PPDE.to_numpy(float) > 0.95))
    _, apg = mle_shrunk_scored(sub[E].T, cond, b, np.ones(len(E), bool), 0.5, a.cpus, cooks=a.official)
    if apg is not None:
        apg = np.nan_to_num(apg, nan=0.0)
        out["apeglm"] = (np.abs(apg), apg, None)
    return out


def mcc(call, pos, neg):
    """Matthews correlation coefficient on the labelled genes."""
    lab = pos | neg
    c = call[lab]
    p = pos[lab]
    tp = float((c & p).sum())
    fp = float((c & ~p).sum())
    fn = float((~c & p).sum())
    tn = float((~c & ~p).sum())
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else np.nan


for key, (ctl, case, tcol) in CONTRASTS.items():
    ci = [i for i, l in enumerate(libs) if l.startswith(ctl + "_")]
    ki = [i for i, l in enumerate(libs) if l.startswith(case + "_")]
    for n in a.ns:
        combos = list(itertools.product(itertools.combinations(ci, n), itertools.combinations(ki, n)))
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in rng.permutation(len(combos))[: a.max_reps]]

        def one(rep, cs, ks):
            idx = np.array(cs + ks)
            cond = np.array([0] * n + [1] * n)
            sub = X[:, idx]
            E = np.where(universe_mask(sub))[0]  # TXDE_PREFILTER=none -> every gene with a count
            res = methods(sub, cond, E)
            g = genes[E]
            has = np.isin(g, T.index.to_numpy())
            t = T[tcol].reindex(g[has]).to_numpy(float)
            keep = np.isfinite(t)
            if tcol == "lfc_BA":
                keep &= T["present_AB"].reindex(g[has]).to_numpy(bool)
            # with --official every method is scored on edgeR's filterByExpr genes
            if a.official and "edgeR_TREAT05" in res:
                keep &= np.isfinite(res["edgeR_TREAT05"][1][has])
            t = t[keep]
            pos = np.abs(t) > 1.0
            neg = np.abs(t) < 0.25
            big = np.abs(t) > a.tau
            out, gr = [], []
            for nm, (sc, est, call) in res.items():
                sc = sc[has][keep]
                est = est[has][keep]
                rec = {
                    "contrast": key,
                    "n": 2 * n,
                    "rep": rep,
                    "method": nm,
                    "n_truth_genes": int(keep.sum()),
                    "n_pos": int(pos.sum()),
                    "n_neg": int(neg.sum()),
                    "slope_rev": slope(est, t),
                    "slope_fwd": slope(t, est),
                    "ratio": float(np.abs(est).mean() / np.abs(t).mean()),
                    "rmse": float(np.sqrt(np.mean((est - t) ** 2))),
                    "spearman": float(pd.Series(est).corr(pd.Series(t), method="spearman")),
                    "sign_acc": float((np.sign(est[big]) == np.sign(t[big])).mean()),
                    "sweep": sweep_power(sc, pos, neg),
                }
                if call is not None:
                    c = call[has][keep]
                    rec["op_power"] = float(pos[c].sum() / max(1, pos.sum()))
                    rec["op_fdr"] = float(neg[c].sum() / max(1, c.sum()))
                    rec["op_calls"] = float(c.sum())
                    rec["mcc"] = mcc(c, pos, neg)
                out.append(rec)
                for gg, tt, ee in zip(g[has][keep], t, est):
                    gr.append(
                        {
                            "run": a.tag,
                            "contrast": key,
                            "n": 2 * n,
                            "rep": rep,
                            "method": nm,
                            "gene": gg,
                            "true": tt,
                            "est": ee,
                        }
                    )
            return out, gr

        parts = Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
            delayed(one)(rep, cs, ks) for rep, (cs, ks) in enumerate(combos)
        )
        acc = [r for o, _ in parts for r in o]
        grows += [r for _, gg in parts for r in gg]
        d = pd.DataFrame(acc)
        for nm, gdf in d.groupby("method"):
            for c in gdf.columns:
                if c in ("contrast", "rep", "method"):
                    continue
                v = gdf[c].dropna()
                if len(v):
                    rows.append(
                        {
                            "run": a.tag,
                            "contrast": key,
                            "n": 2 * n,
                            "method": nm,
                            "metric": c,
                            "value": float(v.mean()),
                            "sd": float(v.std()),
                        }
                    )
        w = d.set_index("method")
        print(
            f"  {key} n={2*n}: rev slope "
            + "  ".join(f"{nm} {d[d.method==nm].slope_rev.mean():.3f}" for nm in d.method.unique()),
            flush=True,
        )
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"seqc_taqman_{a.tag}.csv"), index=False)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(RESULTS, f"seqc_taqman_{a.tag}.csv"), index=False)
pd.DataFrame(grows).to_csv(os.path.join(RESULTS, f"seqc_taqman_{a.tag}_genes.csv"), index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
print(
    "\n=== MAQC TaqMan truth (independent of the DE methods). AB = large effects, DC = small effects of the mixture design"
)
for met in (
    "mcc",
    "slope_rev",
    "ratio",
    "rmse",
    "spearman",
    "sign_acc",
    "sweep",
    "op_power",
    "op_fdr",
    "op_calls",
):
    t_ = d[d.metric == met].pivot_table(index="method", columns=["contrast", "n"], values="value")
    if len(t_):
        print(f"\n-- {met}")
        print(t_.round(3).to_string())
print(
    "\nNote: the five libraries per sample are technical replicates of one RNA pool, so there is no biological overdispersion; power is optimistic for every method, and the point is the accuracy of the fold changes."
)
print(f"\nwritten: results/seqc_taqman_{a.tag}.csv")
