"""DE-fraction sweep and outlier test on a real null background (after Li et al. 2022 and Hejblum et al. 2024).

Male GTEx donors of one tissue are split at random n/2 vs n/2, so with nothing planted every call is false. --mode defrac
plants LFC ~ N(0, 1) in a random --fracs share of genes; --mode outlier multiplies a share of (gene, sample) cells by
U(--outlier_mult), with and without planted effects. Methods: PRESET, edgeR, limma-voom, Wilcoxon and optionally DESeq2.

    python analysis/planted/defrac_sweep.py --mode outlier --ns 8 --deseq2 --official --donor_split eval --plant thinning --preset_decoy --ckpts CKPT --names PRESET --gpu -1 --output_dir DIR
Writes sweep.csv.
"""

import argparse, os, sys, warnings
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
ANALYSIS = os.path.join(ROOT, "analysis")
for p_ in [ROOT] + [os.path.join(ANALYSIS, d) for d in sorted(os.listdir(ANALYSIS))]:
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from benchmark import (
    load_model,
    run_edger,
    run_limma_voom,
    run_wilcoxon,
    run_deseq2,
    run_deseq2_scored,
    universe_mask,
)
import preset
from gtex_sex_eval import load_tissue_sex
from gtex_realdata_eval import ctx_matrix
import fair_compare as fc

CKPTS = {"PRESET": None}  # None = the released weights (preset.weights.resolve_weights)
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


def sweep_power(score, pos, neg, thresholds, less):
    best = 0.0
    for t in thresholds:
        sel = score < t if less else score > t
        fdr = (sel & neg).sum() / max(1, (sel & (pos | neg)).sum())
        pw = (sel & pos).sum() / max(1, pos.sum())
        if fdr <= 0.05:
            best = max(best, pw)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["defrac", "outlier"], default="defrac")
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--ns", nargs="+", type=int, default=[8, 24])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument(
        "--max_genes",
        type=int,
        default=0,
        help="0 = every expressed gene; >0 = random gene subset (debugging only)",
    )
    ap.add_argument("--fracs", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument(
        "--context",
        choices=["expressed", "all", "cap", "auto"],
        default="auto",
        help="PRESET input genes with --preset_input universe (see ctx_matrix)",
    )
    ap.add_argument("--ckpts", nargs="+", default=None)
    ap.add_argument("--names", nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--thin",
        type=float,
        default=1.0,
        help="binomial thinning keep probability before planting (depth sweep)",
    )
    ap.add_argument("--output_dir", default=None)
    ap.add_argument(
        "--outlier_mult",
        type=float,
        nargs=2,
        default=[10.0, 50.0],
        help="outlier cell multiplier range (Li et al. 2022 used 10-50)",
    )
    ap.add_argument(
        "--official",
        action="store_true",
        help="user's-guide workflows; all methods scored on edgeR filterByExpr genes",
    )
    ap.add_argument(
        "--deseq2", action="store_true", help="also run DESeq2 (lfcThreshold, greaterAbs, Cook's cutoff on)"
    )
    ap.add_argument(
        "--plant",
        choices=["poisson", "thinning"],
        default="poisson",
        help="'thinning': exact binomial thinning, as in training; 'poisson': Poisson resampling of case counts (raises low-count P(DE))",
    )
    ap.add_argument(
        "--preset_input",
        choices=["package", "universe"],
        default="package",
        help="'package': preset.analyze with its own filter on the tools' matrix; 'universe': every gene ranked",
    )
    ap.add_argument(
        "--preset_decoy",
        action="store_true",
        help="also report 'PRESET decoy': the same posterior called with the package default rule (target-decoy FDR 5%%, 8 passes + 1)",
    )
    ap.add_argument(
        "--decoy_depth_arms",
        action="store_true",
        help="with --preset_decoy also report 'PRESET decoy match' (decoy_depth='match'); only edgeR among the others",
    )
    ap.add_argument(
        "--tissue",
        default="lung",
        help="GTEx tissue for the background (not one of the 11 training-bank tissues)",
    )
    ap.add_argument(
        "--donor_split",
        choices=["all", "bank", "eval"],
        default="all",
        help="'eval': donors outside the training bank; 'bank': training-bank donors; 'all': every donor",
    )
    args = ap.parse_args()
    if args.ckpts:
        CKPTS.clear()
        CKPTS.update(
            {nm: (c if os.path.isabs(c) else os.path.join(ROOT, c)) for nm, c in zip(args.names, args.ckpts)}
        )
    out_dir = args.output_dir or os.path.join(RESULTS, f"{args.mode}_sweep")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    models = {nm: load_model(p, dev)[0] for nm, p in CKPTS.items()}
    sym, XL, sex = load_tissue_sex(
        args.tissue, donor_split=None if args.donor_split == "all" else args.donor_split
    )
    male = np.where(sex == 1)[0]
    _expr = np.where(universe_mask(XL))[0]
    # TXDE_PREFILTER=none -> every gene with a count
    idx = (
        np.sort(rng.choice(_expr, min(args.max_genes, len(_expr)), replace=False))
        if args.max_genes > 0
        else _expr
    )
    X = XL[idx]
    G = len(idx)
    ctx = np.where(XL.sum(1) > 0)[0]
    pos_ctx = np.searchsorted(ctx, idx)
    XLc = XL[ctx]
    print(
        f"{args.tissue} [{args.donor_split}] male donors {len(male)}, genes {G}, thin {args.thin}, median library after thinning {int(np.median(X.sum(0)) * args.thin)}",
        flush=True,
    )

    def run_preset(m, cnt, cond):
        # cnt: genes x samples
        r = preset.run(cnt.T, cond, tau=args.tau, model=m, decoy=False)
        return r.lfc_mean.to_numpy(float), r.p_de.to_numpy(float)

    settings = (
        [("defrac", f, 0.0) for f in args.fracs]
        if args.mode == "defrac"
        else [
            ("outlier", 0.0, 0.005),
            ("outlier", 0.0, 0.02),
            ("outlier", 0.1, 0.005),
            ("outlier", 0.1, 0.02),
        ]
    )
    rows = []
    for n in args.ns:
        for kind, frac, ofrac in settings:
            for rep in range(args.reps):
                s = rng.choice(male, n, replace=False)
                cnt = X[:, s].astype(np.int64)
                h = n // 2
                cond = np.array([0] * h + [1] * h)
                t = np.zeros(G)
                if args.thin < 1.0:
                    cnt = rng.binomial(cnt, args.thin)
                    thin_full = XL.copy()
                    thin_full[:, s] = rng.binomial(XL[:, s], args.thin)
                if frac > 0:
                    de = rng.random(G) < frac
                    t[de] = rng.normal(0, 1.0, de.sum())
                    m_ = t != 0
                    if args.plant == "thinning":
                        from real_background import _inject

                        # up genes: control half thinned by 2^-lfc; down genes: case half by 2^lfc
                        cnt = _inject(cnt, t, h, rng, "thinning")
                    else:
                        cnt[m_, h:] = rng.poisson(cnt[m_, h:] * (2.0 ** t[m_])[:, None])
                if ofrac > 0:
                    cells = rng.random(cnt.shape) < ofrac
                    cnt[cells] = (
                        cnt[cells] * rng.uniform(args.outlier_mult[0], args.outlier_mult[1], cells.sum())
                    ).astype(np.int64)
                # RAW[method] = (evidence, higher_is_better) for the sweep
                pos = np.abs(t) > 1
                neg = np.abs(t) < 0.25
                RAW = {}
                res = {}
                for nm, m in models.items():
                    # same matrix as the tools; PRESET applies its own filter
                    if args.preset_input == "package":
                        import preset as _preset

                        ra = _preset.analyze(
                            pd.DataFrame(cnt.T),
                            cond,
                            model=m,
                            tau=args.tau,
                            decoy=args.preset_decoy,
                            decoy_seed=int(args.seed) * 1000 + rep,
                        )
                        lfc = ra.lfc_mean.to_numpy(float)
                        q = np.where(ra.reported.to_numpy(bool), ra.p_de.to_numpy(float), np.nan)
                        res[nm] = (lfc, q, False)
                        RAW[nm] = (q, True)
                        if args.preset_decoy:
                            res[nm + " decoy"] = (
                                lfc,
                                np.nan_to_num(ra.decoy_fdr.to_numpy(float), nan=1.0),
                                True,
                            )
                            RAW[nm + " decoy"] = (q, True)
                        if args.preset_decoy and args.decoy_depth_arms:
                            rb = _preset.analyze(
                                pd.DataFrame(cnt.T),
                                cond,
                                model=m,
                                tau=args.tau,
                                decoy=True,
                                decoy_seed=int(args.seed) * 1000 + rep,
                                decoy_depth="match",
                            )
                            res[nm + " decoy match"] = (
                                rb.lfc_mean.to_numpy(float),
                                np.nan_to_num(rb.decoy_fdr.to_numpy(float), nan=1.0),
                                True,
                            )
                            RAW[nm + " decoy match"] = (q, True)
                        continue
                    XLt = XL if args.thin >= 1.0 else thin_full
                    cm, cp = ctx_matrix(XLt, s, idx, args.context)
                    cm = cm.astype(np.int64)
                    cm[cp] = cnt
                    lfc, q = run_preset(m, cm, cond)
                    lfc, q = lfc[cp], q[cp]
                    res[nm] = (lfc, q, False)
                    RAW[nm] = (q, True)
                    if args.preset_decoy:
                        import preset as _preset

                        rm_ = np.zeros(cm.shape[0], bool)
                        rm_[cp] = True
                        rd = _preset.run(
                            pd.DataFrame(cm.T),
                            cond,
                            model=m,
                            tau=args.tau,
                            report=rm_,
                            decoy_seed=int(args.seed) * 1000 + rep,
                        )
                        res[nm + " decoy"] = (
                            lfc,
                            np.nan_to_num(rd.decoy_fdr.to_numpy(float)[cp], nan=1.0),
                            True,
                        )
                        RAW[nm + " decoy"] = (q, True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kw = (
                        {"edgeR": dict(filter_expr=True, robust=True), "limma-voom": dict(filter_expr=True)}
                        if args.official
                        else {}
                    )
                    for nm, f in (
                        (("edgeR", run_edger),)
                        if args.decoy_depth_arms
                        else (
                            ("edgeR", run_edger),
                            ("limma-voom", run_limma_voom),
                            ("Wilcoxon", run_wilcoxon),
                        )
                    ):
                        r = f(cnt.T, cond, np.zeros_like(cond), tau=args.tau, **kw.get(nm, {}))
                        if r is not None:
                            res[nm] = (
                                r["logFC"].to_numpy(float),
                                np.nan_to_num(r["FDR"].to_numpy(float), nan=1.0),
                                True,
                            )
                            RAW[nm] = (r["PValue"].to_numpy(float), False)
                    # Cook's handling on (DESeq2's default)
                    if args.deseq2 and not args.decoy_depth_arms:
                        # scored set: edgeR filterByExpr genes
                        sub = (
                            np.isfinite(res["edgeR"][0])
                            if (args.official and "edgeR" in res)
                            else np.ones(G, bool)
                        )
                        # R mode: all genes, masked to the scored set
                        r = run_deseq2_scored(cnt.T, cond, np.zeros_like(cond), sub, tau=args.tau, cooks=True)
                        if r is not None:
                            l_ = r["log2FoldChange"].to_numpy(float)
                            q_ = np.where(sub, np.nan_to_num(r["padj"].to_numpy(float), nan=1.0), 1.0)
                            res["DESeq2"] = (l_, q_, True)
                            RAW["DESeq2"] = (r["pvalue"].to_numpy(float), False)
                # calls only, scored on the common set
                # common scored set
                ev = np.isfinite(res["edgeR"][0]) if (args.official and "edgeR" in res) else np.ones(G, bool)
                pos = pos & ev
                neg = neg & ev
                # every method tests |log2FC| > tau, so a call with |t| < tau is false
                pos_t = (np.abs(t) >= args.tau) & ev
                neg_t = (np.abs(t) < args.tau) & ev
                mc = cnt.mean(1)
                for nm, (lfc, sc, less) in res.items():
                    # calls on all genes (PRESET: its reported set)
                    sc0 = np.nan_to_num(sc, nan=1.0 if less else 0.0)
                    call_all = sc0 < 0.05 if less else sc0 > 0.9
                    sc = np.where(ev, sc, 1.0 if less else 0.0)
                    lfc = np.where(ev, np.nan_to_num(lfc), 0.0)
                    call = sc < 0.05 if less else sc > 0.9
                    row = {
                        "n": n,
                        "kind": kind,
                        "de_frac": frac,
                        "outlier_frac": ofrac,
                        "outlier_mult_hi": args.outlier_mult[1],
                        "thin": args.thin,
                        "rep": rep,
                        "method": nm,
                        "n_scored": int(ev.sum()),
                        "frac_called": float(call[ev].mean()),
                        "n_called": int(call.sum()),
                        "op_power": (call & pos).sum() / max(1, pos.sum()) if pos.any() else np.nan,
                        # op_fdr: false = |t| < tau over all calls; op_fdr_labelled: false = |t| < 0.25 over labeled calls
                        "op_fdr": (
                            (call & ev & (np.abs(t) < args.tau)).sum() / max(1, (call & ev).sum())
                            if (call & ev).any()
                            else 0.0
                        ),
                        "op_fdr_labelled": (
                            (call & neg).sum() / max(1, (call & (pos | neg)).sum()) if call.any() else 0.0
                        ),
                        # op_fdr_zero: false = t == 0
                        "op_fdr_zero": (
                            (call & ev & (t == 0)).sum() / max(1, (call & ev).sum())
                            if (call & ev).any()
                            else 0.0
                        ),
                        "plant": args.plant,
                        "preset_input": args.preset_input,
                        "n_called_all": int(call_all.sum()),
                        "op_fdr_all": (
                            float((call_all & (np.abs(t) < args.tau)).sum() / call_all.sum())
                            if call_all.any()
                            else 0.0
                        ),
                        "op_fdr_zero_all": (
                            float((call_all & (t == 0)).sum() / call_all.sum()) if call_all.any() else 0.0
                        ),
                        "op_power_all": float(
                            (call_all & (np.abs(t) >= args.tau)).sum() / max(1, (np.abs(t) >= args.tau).sum())
                        ),
                        # continuous ranking; the threshold grid (sweep_power_grid) gave DESeq2 zero power in 31% of repeats
                        "sweep_power": (
                            fc.sweep_power(
                                fc.rank_score(np.where(ev, RAW[nm][0], np.nan), lfc, mc, RAW[nm][1]), pos, neg
                            )
                            if (pos.any() and nm in RAW)
                            else np.nan
                        ),
                        "sweep_power_tau": (
                            fc.sweep_power(
                                fc.rank_score(np.where(ev, RAW[nm][0], np.nan), lfc, mc, RAW[nm][1]),
                                pos_t,
                                neg_t,
                            )
                            if (pos_t.any() and nm in RAW)
                            else np.nan
                        ),
                        "sweep_power_grid": (
                            sweep_power(sc, pos, neg, ALPHAS if less else QS, less) if pos.any() else np.nan
                        ),
                        "slope_injected": (
                            np.polyfit(t[pos | neg], np.nan_to_num(lfc[pos | neg]), 1)[0]
                            if pos.sum() >= 5
                            else np.nan
                        ),
                    }
                    rows.append(row)
            print(f"n={n} {kind} f={frac} o={ofrac} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "sweep.csv"), index=False)
    key = "de_frac" if args.mode == "defrac" else "outlier_frac"
    for metric in ("frac_called", "sweep_power", "op_power", "op_fdr", "slope_injected"):
        print(f"\n=== {metric} (mean over reps) ===")
        print(
            df.pivot_table(index=["n", "de_frac", "outlier_frac"], columns="method", values=metric)
            .round(3)
            .to_string()
        )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
