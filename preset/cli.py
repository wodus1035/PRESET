"""Command line:
    preset run counts.tsv meta.tsv --condition group [--control A --case B] [--batch plate] [--rule auto|decoy|bfdr|boundary] --out res.tsv
    preset diagnose counts.tsv meta.tsv --condition group
Counts: raw integers, genes x samples or samples x genes (detected). Metadata: one row per sample, indexed by sample id.
"""

import argparse, json, sys
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(prog="preset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("counts")
        p.add_argument("meta")
        p.add_argument("--condition", required=True)
        p.add_argument("--control")
        p.add_argument("--case")
        p.add_argument("--batch")
        p.add_argument(
            "--covariate",
            help="EXPERIMENTAL, not evaluated: metadata column with one continuous covariate (e.g. age), standardized within the dataset",
        )
        p.add_argument(
            "--donor",
            help="EXPERIMENTAL, not evaluated: metadata column with the donor/subject id for paired designs",
        )
        p.add_argument("--tau", type=float, default=0.5)
        p.add_argument(
            "--context",
            choices=["auto", "expressed", "cap", "all"],
            default="auto",
            help="low-count genes the model sees (auto: none below 8 samples, else the highest-count ones up to 25%% of the expressed genes)",
        )
        p.add_argument("--weights")
        p.add_argument("--device", help="cpu, cuda, cuda:1 ...")
        if name == "run":
            p.add_argument("--fdr", type=float, default=0.05)
            p.add_argument(
                "--rule",
                choices=["auto", "decoy", "bfdr", "boundary"],
                default="auto",
                help="auto (default): decoy below 16 samples in the smaller group, bfdr from 16; decoy: target-decoy FDR from 8 extra passes (about 8x the run time); bfdr: Bayesian FDR (one pass); boundary: also decoys with an effect of +-tau, conservative at any tau",
            )
            p.add_argument(
                "--decoy-passes",
                dest="decoy_passes",
                type=int,
                default=8,
                help="decoy passes; the shortest list callable at FDR q is 1/(0.25 q passes) genes",
            )
            p.add_argument("--level", type=float, default=0.90)
            p.add_argument("--out", required=True)
            p.add_argument("--no_diagnose", action="store_true")
            p.add_argument(
                "--experimental-intervals",
                dest="intervals",
                action="store_true",
                help="add lfc_sd/lfc_median/lfc_lo/lfc_hi (credible intervals calibrated on average over genes, not per gene)",
            )
        if name == "run":
            p.add_argument("--volcano", help="save a volcano plot (.png/.pdf; needs matplotlib)")
            p.add_argument(
                "--embed", help="save the network embedding of the reported genes (.tsv, genes x d)"
            )
    a = ap.parse_args(argv)
    from . import io as dio
    from . import run as drun, calls as dcalls, diagnose as ddiag, load_model

    counts, obs = dio.read_counts(a.counts)
    meta = obs if obs is not None else dio.read_metadata(a.meta)
    samples = [s for s in counts.index if s in meta.index]
    if len(samples) < len(counts.index):
        print(
            f"[preset] {len(counts.index) - len(samples)} samples without metadata dropped", file=sys.stderr
        )
    counts = counts.loc[samples]
    cond, batch, keep, labels = dio.design_from_metadata(
        meta, samples, a.condition, a.control, a.case, a.batch
    )
    counts, cond, batch = counts[keep], cond[keep], batch[keep]
    samples = [s for s, k in zip(samples, keep) if k]
    covar = None
    if a.covariate or a.donor:
        covar = dio.build_covariates(meta, samples, continuous=a.covariate, donor=a.donor)
        print(
            "[preset] WARNING: --covariate/--donor are EXPERIMENTAL; their effect on calibration and power has not been evaluated. The model takes ONE continuous covariate + one two-level --batch + a donor code; it is not a ~ age + sex + batch design matrix",
            file=sys.stderr,
        )
        if a.covariate:
            raw = meta.loc[samples, a.covariate].to_numpy(float)
            print(
                f"[preset] covariate '{a.covariate}' standardized within this dataset: raw {raw.min():.4g}..{raw.max():.4g} (mean {raw.mean():.4g}, sd {raw.std(ddof=1):.4g}) -> z {covar[:, 0].min():.3f}..{covar[:, 0].max():.3f}"
                + ("  [sd = 0: constant covariate, column left at 0]" if raw.std(ddof=1) == 0 else ""),
                file=sys.stderr,
            )
        if a.donor:
            d = meta.loc[samples, a.donor].astype(str).to_numpy()
            lv = sorted(set(d))
            both = sum(1 for v in lv if len(set(cond[d == v])) == 2)
            print(
                f"[preset] donor '{a.donor}': {len(lv)} levels, {both} appearing in both conditions; is_paired flag set to 1, donor code = 4 arbitrary numbers per donor",
                file=sys.stderr,
            )
            if both == 0:
                print(
                    "[preset] WARNING: no donor appears in both conditions, so this is not a paired design; the donor code adds nothing the model can use",
                    file=sys.stderr,
                )
    X, genes, rep, expressed = dio.prepare(counts, context=a.context)
    print(
        f"[preset] {labels['control']} (n={int((cond==0).sum())}) vs {labels['case']} (n={int((cond==1).sum())}); genes given to the model {rep['n_genes_kept']}/{rep['n_genes_in']} (expressed {rep['n_genes_expressed']}, context rule {rep['context_rule']}); median library {rep['median_library']}",
        file=sys.stderr,
    )
    model, dev, wmeta = load_model(a.weights, a.device)
    print(f"[preset] weights {wmeta['weights']} on {dev}", file=sys.stderr)
    import pandas as pd

    Xdf = pd.DataFrame(X, index=samples if len(samples) == len(X) else None, columns=genes)
    if max(int((cond == 0).sum()), int((cond == 1).sum())) > 16:
        print(
            "[preset] note: PRESET was trained on 2-16 samples per group; with larger groups classical count GLMs (edgeR, DESeq2) are the natural choice",
            file=sys.stderr,
        )
    if int((cond == 0).sum()) < 4 or int((cond == 1).sum()) < 4:
        print(
            "[preset] WARNING: fewer than 4 samples in a group: outside the validated range (error control was shown for >= 4 per group); use the ranking and expected_false_in_list with care",
            file=sys.stderr,
        )
    if a.cmd == "diagnose" or not a.no_diagnose:
        d = ddiag(Xdf, cond, batch, covar=covar, tau=a.tau, model=model)
        for f in d["flags"]:
            print(f"[preset] WARNING: {f}", file=sys.stderr)
        if a.cmd == "diagnose":
            print(json.dumps(d, indent=1))
            return
    res = drun(
        Xdf,
        cond,
        batch,
        covar=covar,
        tau=a.tau,
        level=a.level,
        model=model,
        intervals=a.intervals,
        report=expressed,
        decoy=a.rule == "decoy"
        or (a.rule == "auto" and min(int((cond == 0).sum()), int((cond == 1).sum())) < 16),
        decoy_passes=a.decoy_passes,
        boundary=a.rule == "boundary",
    )
    res["called"] = dcalls(res, fdr=a.fdr, rule=a.rule)
    res["expressed"] = expressed
    if a.intervals:
        print(
            "[preset] note: credible intervals are calibrated on average over genes; for genes with a true effect they are narrower than nominal",
            file=sys.stderr,
        )
    print(
        f"[preset] note: lfc_mean is the posterior mean and moves towards 0 when evidence is weak (e.g. {int((res.mean_count < 50).sum())} genes with mean count < 50); lfc_given_de is the effect size if the gene changes",
        file=sys.stderr,
    )
    res.sort_values("rank").to_csv(a.out, sep="\t" if a.out.endswith(".tsv") else ",", index=False)
    if a.volcano:
        import matplotlib

        matplotlib.use("Agg")
        from .plot import volcano

        ax = volcano(res, called=res.called.to_numpy(), fdr=a.fdr, rule=a.rule)
        ax.figure.savefig(a.volcano, dpi=200, bbox_inches="tight")
        print(f"[preset] volcano plot -> {a.volcano}", file=sys.stderr)
    if a.embed:
        from . import embed as dembed

        emb = dembed(Xdf, cond, batch, covar=covar, model=model)
        emb[np.asarray(expressed, bool)].to_csv(a.embed, sep="\t")
        print(
            f"[preset] embedding ({int(np.asarray(expressed, bool).sum())} genes x {emb.shape[1]}) -> {a.embed}",
            file=sys.stderr,
        )
    print(
        f"[preset] {int(res.called.sum())} genes called at {'boundary-decoy' if a.rule == 'boundary' else ('target-decoy' if 'decoy_fdr' in res and (a.rule == 'decoy' or (a.rule == 'auto' and min(res.attrs['n_control'], res.attrs['n_case']) < 16)) else 'Bayesian')} FDR <= {a.fdr} (tau={a.tau}); {int((res.p_de > 0.9).sum())} with P(|log2FC|>{a.tau}) > 0.9 -> {a.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
