"""Build the real-cohort bank of the second stage, with labels from conventional methods on the other folds.

Each cohort is split into K folds, stratified by group and by batch where there is one (5 folds, or 3 when the smaller
group has fewer than 40 samples). For fold f the reference is every other fold. edgeR (quasi-likelihood, robust),
limma-voom and the Wilcoxon test are run on it at tau = 0, and each gene gets a label
    changed   at least two of the three at FDR < 0.05 and |log2FC| > 1, with the edgeR and limma log2FC within 0.3 of
              each other and all three signs equal -> mean of the edgeR and limma log2FC
    unchanged all three at FDR > 0.5 and |log2FC| < 0.25 -> exactly 0
    otherwise NaN (masked out of the loss)
Training tasks draw their samples from fold f only, so input and label never share a sample.
Runs the R methods of the evaluation code (evaluation/README.md for the R setup):
    python training/build_consensus_bank.py --jobs 24
then add_observed_labels.py adds the labels without DE methods.
"""

import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import cohorts

from benchmark import (
    run_edger,
    run_limma_voom,
    run_wilcoxon,
)  # noqa: E402  (evaluation/analysis/common, via cohorts)
from fair_compare import consensus_reference  # noqa: E402

MAG_TOL = 0.3  # largest edgeR - limma difference of a labeled log2FC


def expressed_genes(X):
    """Genes with at least 10 counts in a quarter of the samples."""
    keep = (X >= 10).sum(1) >= max(1, int(np.ceil(0.25 * X.shape[1])))
    return np.where(keep)[0]


def make_folds(cond, batch, K, rng):
    fold = np.full(len(cond), -1, int)
    if batch is None:
        strata = cond
    else:
        strata = cond * 1000 + pd.factorize(batch)[0]
    for s_ in np.unique(strata):
        idx = rng.permutation(np.where(strata == s_)[0])
        fold[idx] = np.arange(len(idx)) % K
    return fold


def reference(X_E, rcond, rb):
    """Labels of the expressed genes from the reference samples. Returns (labels, changed, labeled changed, unchanged)."""
    stats = {}
    lfcs = {}
    methods = (
        ("edgeR", run_edger, dict(filter_expr=True, robust=True)),
        ("limma", run_limma_voom, dict(filter_expr=True)),
        ("wilcoxon", run_wilcoxon, {}),
    )
    for name, fn, kw in methods:
        r = fn(X_E.T, rcond, rb, tau=0.0, **kw)
        if r is None:
            return None
        stats[name] = (r["FDR"].to_numpy(float), r["logFC"].to_numpy(float))
        lfcs[name] = np.nan_to_num(r["logFC"].to_numpy(float), nan=0.0)

    pos, neg = consensus_reference(stats)

    # changed genes are labeled only when the effect sizes agree
    agree = np.abs(lfcs["edgeR"] - lfcs["limma"]) <= MAG_TOL
    agree &= np.sign(lfcs["edgeR"]) == np.sign(lfcs["wilcoxon"])
    agree &= np.sign(lfcs["edgeR"]) == np.sign(lfcs["limma"])
    ok = pos & agree

    lab = np.full(X_E.shape[0], np.nan, np.float32)
    lab[neg] = 0.0
    lab[ok] = 0.5 * (lfcs["edgeR"][ok] + lfcs["limma"][ok])
    return lab, int(pos.sum()), int(ok.sum()), int(neg.sum())


def one_fold(name, X, cond, batch, fold, E, f):
    ref = fold != f
    if batch is None:
        rb = np.zeros(int(ref.sum()), int)
    else:
        rb = pd.factorize(np.asarray(batch)[ref])[0]
    return name, f, reference(X[E][:, ref], cond[ref], rb)


def load_cohorts(sources, min_per_group, min_sex):
    out = []
    for src in sources:
        if src == "cvd":
            out += cohorts.cvd(min_per_group=min_per_group)
        elif src == "gtex_sex":
            out += cohorts.gtex_sex(min_sex=min_sex)
        else:
            out += getattr(cohorts, src)()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources", nargs="+", default=["cvd", "psoriasis", "hape", "ted", "mouse_sex", "gtex_sex"]
    )
    ap.add_argument("--min_per_group", type=int, default=40, help="smallest group of a CVD series")
    ap.add_argument("--min_sex", type=int, default=40, help="donors of each sex in a GTEx tissue")
    ap.add_argument("--folds_large", type=int, default=5)
    ap.add_argument("--folds_small", type=int, default=3)
    ap.add_argument(
        "--small_below",
        type=int,
        default=40,
        help="cohorts whose smaller group is below this use folds_small",
    )
    ap.add_argument("--seed", type=int, default=4001)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--output_dir", default=os.path.join(cohorts.HERE, "data", "consensus_bank"))
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Cohorts, folds and expressed genes
    store = {}
    jobs = []
    for name, X, ctrl, case, batch, genes in load_cohorts(a.sources, a.min_per_group, a.min_sex):
        idx = np.concatenate([ctrl, case])
        cond = np.array([0] * len(ctrl) + [1] * len(case))
        X = np.asarray(X)[:, idx].astype(np.int64)
        b = None if batch is None else np.asarray(batch)[idx]

        K = a.folds_large if min(len(ctrl), len(case)) >= a.small_below else a.folds_small
        rng = np.random.default_rng(zlib.crc32(name.encode()) + a.seed)
        fold = make_folds(cond, b, K, rng)
        E = expressed_genes(X)
        store[name] = dict(X=X, cond=cond, batch=b, fold=fold, E=E, genes=np.asarray(genes).astype(str), K=K)
        for f in range(K):
            jobs.append((name, f))
        print(f"{name}: {len(ctrl)} ctrl / {len(case)} case, {K} folds, {len(E)} expressed genes", flush=True)

    # Reference fits, one per fold
    print(f"{len(jobs)} reference fits over {a.jobs} workers", flush=True)
    res = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(one_fold)(
            nm,
            store[nm]["X"],
            store[nm]["cond"],
            store[nm]["batch"],
            store[nm]["fold"],
            store[nm]["E"],
            f,
        )
        for nm, f in jobs
    )

    # One npz per cohort
    rows = []
    for name, d in store.items():
        lab = np.full((d["K"], d["X"].shape[0]), np.nan, np.float32)
        ok = True
        for nm, f, r in res:
            if nm != name:
                continue
            if r is None:
                ok = False
                print(f"{name} fold {f}: a reference method failed", flush=True)
                continue
            lab[f, d["E"]] = r[0]
            n_ref = int((d["fold"] != f).sum())
            rows.append(
                dict(
                    cohort=name,
                    fold=f,
                    n_ref=n_ref,
                    pos=r[1],
                    pos_labelled=r[2],
                    neg=r[3],
                    expressed=len(d["E"]),
                )
            )
        if not ok:
            continue

        if d["batch"] is None:
            batch = np.full(len(d["cond"]), -1, np.int16)
        else:
            batch = pd.factorize(d["batch"])[0].astype(np.int16)
        fn = name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "") + ".npz"
        np.savez_compressed(
            os.path.join(a.output_dir, fn),
            X=d["X"].astype(np.int32),
            cond=d["cond"].astype(np.int8),
            batch=batch,
            fold=d["fold"].astype(np.int8),
            E=d["E"].astype(np.int32),
            lab=lab,
            genes=d["genes"],
            meta=json.dumps(dict(name=name, K=d["K"], mag_tol=MAG_TOL)),
        )

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.output_dir, "bank_summary.csv"), index=False)
    print(t.to_string(index=False))
    print("written", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
