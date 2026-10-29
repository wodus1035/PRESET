"""GTEx male vs female donors within one tissue, where chrY genes and XIST must be DE and most genes are null.

Reference: n_ref donors per sex, edgeR QL at tau=0 (or --ref consensus). Test: --reps subsets of n/2 per sex from the other
donors with PRESET (--ckpts), edgeR glmTreat and DESeq2 (--extra adds limma-voom and Wilcoxon), scored by FDR and power
against the reference, recall of KNOWN_SEX_GENES and LFC slope.

    python analysis/common/gtex_sex_eval.py --ckpts CKPT --names PRESET --donor_split eval --gpu -1
"""

import argparse, os, sys, warnings
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
ANALYSIS = os.path.join(ROOT, "analysis")
for p_ in [ROOT] + [os.path.join(ANALYSIS, d) for d in sorted(os.listdir(ANALYSIS))]:
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from benchmark import load_model, run_deseq2, run_edger
from gtex_realdata_eval import (
    GTEX,
    QS,
    ALPHAS,
    fdr_power,
    extra_baselines,
    consensus_reference,
    ctx_matrix,
    donor_mask,
)

KNOWN_SEX_GENES = [
    "XIST",
    "TSIX",
    "RPS4Y1",
    "DDX3Y",
    "KDM5D",
    "UTY",
    "EIF1AY",
    "USP9Y",
    "ZFY",
    "TXLNGY",
    "NLGN4Y",
    "PRKY",
    "TMSB4Y",
    "LINC00278",
    "TTTY15",
    "TTTY14",
]
TISSUES = ["lung", "thyroid", "whole_blood", "muscle_skeletal"]


def load_tissue_sex(name, donor_split=None, split_path=None, return_samples=False):
    """Gene symbols, counts (genes x samples), sex (1 male, 2 female) [, sample ids]. donor_split as in
    gtex_realdata_eval.donor_mask."""
    df = pd.read_csv(
        os.path.join(GTEX, f"gene_reads_v10_{name}.gct.gz"), sep="\t", skiprows=2, compression="gzip"
    )
    sym = df["Description"].to_numpy()
    samples = np.array([c for c in df.columns if c not in ("Name", "Description")])
    samples = samples[donor_mask(samples, donor_split, split_path)]
    X = df[samples].to_numpy(np.int64)
    ph = pd.read_csv(os.path.join(GTEX, "GTEx_v10_SubjectPhenotypes.txt"), sep="\t").set_index("SUBJID")[
        "SEX"
    ]
    subj = pd.Series(samples).str.extract(r"^(GTEX-[^-]+)")[0]
    sex = subj.map(ph).to_numpy()
    return (sym, X, sex, samples) if return_samples else (sym, X, sex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--n_ref", type=int, default=60)
    ap.add_argument("--ns", nargs="+", type=int, default=[4, 6, 8, 12])
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min_count", type=int, default=10)
    ap.add_argument("--max_genes", type=int, default=6000)
    ap.add_argument("--deseq_cpus", type=int, default=8)
    ap.add_argument(
        "--context",
        choices=["expressed", "all", "cap", "auto"],
        default="auto",
        help="PRESET input genes (see ctx_matrix)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tissues", nargs="+", default=TISSUES)
    ap.add_argument("--ref", choices=["edger", "consensus"], default="edger")
    ap.add_argument("--extra", action="store_true", help="add limma-voom and Wilcoxon baselines")
    ap.add_argument(
        "--donor_split",
        choices=["all", "eval", "bank"],
        default="all",
        help="'eval': donors outside the training bank; 'all': every donor",
    )
    ap.add_argument("--output_dir", default=os.path.join(RESULTS, "gtex_sex"))
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"donor_split = {args.donor_split}", flush=True)
    dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    models = {nm: load_model(p, dev)[0] for nm, p in zip(args.names, args.ckpts)}

    def run_preset(model, counts, cond):
        c = torch.tensor(counts.T, dtype=torch.float32, device=dev)[None]
        cd = torch.tensor(cond, dtype=torch.float32, device=dev)[None]
        with torch.no_grad():
            logits, _ = model(c, cd, torch.zeros_like(cd))
            return (
                model.head.mean(logits)[0].cpu().numpy(),
                model.head.prob_abs_gt(logits, args.tau)[0].cpu().numpy(),
            )

    curve_rows, hard_rows, lfc_rows = [], [], []
    for T in args.tissues:
        sym, X, sex = load_tissue_sex(T, args.donor_split)
        male = np.where(sex == 1)[0]
        female = np.where(sex == 2)[0]
        rng.shuffle(male)
        rng.shuffle(female)
        refM, poolM = male[: args.n_ref], male[args.n_ref :]
        refF, poolF = female[: args.n_ref], female[args.n_ref :]
        ref = X[:, np.concatenate([refM, refF])]
        expressed = (ref >= args.min_count).sum(1) >= ref.shape[1] // 4
        known = np.isin(sym, KNOWN_SEX_GENES) & (X.sum(1) > 0)
        idx = np.where(expressed | known)[0]
        if len(idx) > args.max_genes:
            keep_known = idx[known[idx]]
            rest = idx[~known[idx]]
            idx = np.sort(
                np.concatenate(
                    [keep_known, rng.choice(rest, args.max_genes - len(keep_known), replace=False)]
                )
            )
        ctx = np.where(X.sum(1) > 0)[0]
        pos_ctx = np.searchsorted(ctx, idx)
        cond_ref = np.array([0] * len(refM) + [1] * len(refF))  # 0 = male, 1 = female (LFC = female - male)
        if args.ref == "consensus":
            labels, ref_lfc, ref_fdr, _ = consensus_reference(ref[idx], cond_ref, args.deseq_cpus)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = run_edger(ref[idx].T, cond_ref, np.zeros_like(cond_ref), tau=0.0)
            ref_lfc = r["logFC"].to_numpy(float)
            ref_fdr = r["FDR"].to_numpy(float)
            labels = np.zeros(len(idx), int)
            labels[(ref_fdr < 0.01) & (np.abs(ref_lfc) > 1)] = 1
            labels[(np.abs(ref_lfc) < 0.25) & (ref_fdr > 0.5)] = -1
        hard = known[idx]
        pos = labels == 1
        print(
            f"\n### {T}: ref {len(refM)}M+{len(refF)}F, pool {len(poolM)}M+{len(poolF)}F, genes {len(idx)}; reference positives {pos.sum()} (known sex genes among them {int((hard & pos).sum())}/{int(hard.sum())}), negatives {(labels==-1).sum()}",
            flush=True,
        )
        print(
            "   reference positives:",
            ", ".join(sym[idx][pos][np.argsort(-np.abs(ref_lfc[pos]))][:25]),
            flush=True,
        )

        for n in args.ns:
            for rep in range(args.reps):
                sm = rng.choice(poolM, n // 2, replace=False)
                sf = rng.choice(poolF, n // 2, replace=False)
                ss = np.concatenate([sm, sf])
                cnt = X[idx][:, ss]
                cond = np.array([0] * (n // 2) + [1] * (n // 2))
                ests = {}
                calls = {}
                for nm, m in models.items():
                    cm, cp = ctx_matrix(X, ss, idx, args.context)
                    lfc, pde = run_preset(m, cm, cond)
                    lfc, pde = lfc[cp], pde[cp]
                    ests[nm] = lfc
                    calls[nm] = pde > 0.9
                    for row in fdr_power(pde, labels, QS, less_than=False):
                        curve_rows.append({"tissue": T, "n": n, "rep": rep, "model": nm, **row})
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    re = run_edger(cnt.T, cond, np.zeros_like(cond), tau=args.tau)
                    rd = run_deseq2(cnt.T, cond, np.zeros_like(cond), tau=args.tau, n_cpus=args.deseq_cpus)
                if re is not None:
                    f = np.nan_to_num(re["FDR"].to_numpy(float), nan=1.0)
                    ests["edgeR"] = re["logFC"].to_numpy(float)
                    calls["edgeR"] = f < 0.05
                    for row in fdr_power(f, labels, ALPHAS):
                        curve_rows.append({"tissue": T, "n": n, "rep": rep, "model": "edgeR", **row})
                if rd is not None:
                    f = np.nan_to_num(rd["padj"].to_numpy(float), nan=1.0)
                    ests["PyDESeq2"] = rd["log2FoldChange"].to_numpy(float)
                    calls["PyDESeq2"] = f < 0.05
                    for row in fdr_power(f, labels, ALPHAS):
                        curve_rows.append({"tissue": T, "n": n, "rep": rep, "model": "PyDESeq2", **row})
                if args.extra:
                    for nm, (e, f) in extra_baselines(cnt, cond, args.tau).items():
                        ests[nm] = e
                        calls[nm] = f < 0.05
                        for row in fdr_power(f, labels, ALPHAS):
                            curve_rows.append({"tissue": T, "n": n, "rep": rep, "model": nm, **row})
                for nm, cl in calls.items():
                    hard_rows.append(
                        {
                            "tissue": T,
                            "n": n,
                            "rep": rep,
                            "model": nm,
                            "hard_recall": (cl & hard).sum() / max(1, hard.sum()),
                            "n_called": int(cl.sum()),
                            "called_outside_ref_pos": int((cl & ~pos).sum()),
                            "called_neg": int((cl & (labels == -1)).sum()),
                        }
                    )
                for nm, e in ests.items():
                    ok = pos & np.isfinite(e)
                    if ok.sum() >= 5:
                        lfc_rows.append(
                            {
                                "tissue": T,
                                "n": n,
                                "rep": rep,
                                "model": nm,
                                "slope_vs_ref_pos": np.polyfit(ref_lfc[ok], e[ok], 1)[0],
                                "corr_vs_ref_pos": np.corrcoef(ref_lfc[ok], e[ok])[0, 1],
                            }
                        )
            print(f"  n={n} done", flush=True)

    curve = pd.DataFrame(curve_rows)
    hardd = pd.DataFrame(hard_rows)
    lfc = pd.DataFrame(lfc_rows)
    curve.to_csv(os.path.join(args.output_dir, "fdr_power_curves.csv"), index=False)
    hardd.to_csv(os.path.join(args.output_dir, "hard_positives.csv"), index=False)
    lfc.to_csv(os.path.join(args.output_dir, "lfc_vs_reference.csv"), index=False)
    order = list(args.names) + ["edgeR", "PyDESeq2"]
    print(
        "\n=== operating point (PFN q>0.9 / classical adj<0.05): recall of known sex genes, #called, #called that are reference-negative (mean over reps) ==="
    )
    print(
        hardd.groupby(["tissue", "n", "model"])[["hard_recall", "n_called", "called_neg"]]
        .mean()
        .unstack("model")
        .reindex(columns=order, level=1)
        .round(2)
        .to_string()
    )
    print("\n=== power @ realized FDR<=0.05 vs reference labels (mean over reps) ===")

    def best(g):
        ok = g[g.FDR <= 0.05]
        return ok.power.max() if len(ok) else np.nan

    print(
        curve.groupby(["tissue", "n", "model", "rep"])
        .apply(best)
        .groupby(["tissue", "n", "model"])
        .mean()
        .unstack("model")
        .reindex(columns=order)
        .round(3)
        .to_string()
    )
    print("\n=== realized FDR at operating point (mean over reps) ===")
    cal = curve[
        ((curve.model.isin(args.names)) & ((curve.threshold - 0.9).abs() < 1e-9))
        | ((~curve.model.isin(args.names)) & ((curve.threshold - 0.05).abs() < 1e-9))
    ]
    print(
        cal.groupby(["tissue", "n", "model"])["FDR"]
        .mean()
        .unstack("model")
        .reindex(columns=order)
        .round(3)
        .to_string()
    )
    print("\n=== LFC slope vs reference on reference positives (mean over reps) ===")
    print(
        lfc.groupby(["tissue", "n", "model"])["slope_vs_ref_pos"]
        .mean()
        .unstack("model")
        .reindex(columns=order)
        .round(2)
        .to_string()
    )
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
