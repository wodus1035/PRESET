"""Real-data benchmark of PRESET against edgeR, DESeq2, limma-voom and apeglm on subsamples of real cohorts.

All methods test |LFC| > --tau (edgeR glmTreat, DESeq2 lfcThreshold, limma treat; apeglm ranks by |shrunken LFC|). Truth is
TaqMan qPCR and the A/B mixing ratios for SEQC, otherwise the consensus of edgeR, limma-voom and Wilcoxon on a reference half
that shares no sample with the test pool. Scoring rules are in fair_compare.py.

    python analysis/heldout/fair_bench.py --data ipf sepsis crohn skin2 --official --ns 8 --reps 20 --jobs 16 --output_dir DIR
Writes fair_bench<tag>_{metrics,scores,reflabels}.csv.
"""

import os, sys, argparse, itertools, numpy as np, pandas as pd, torch, warnings

warnings.filterwarnings("ignore")
from joblib import Parallel, delayed
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
ANALYSIS = os.path.join(ROOT, "analysis")
for p_ in [ROOT] + [os.path.join(ANALYSIS, d) for d in sorted(os.listdir(ANALYSIS))]:
    if p_ not in sys.path:
        sys.path.insert(0, p_)
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
import preset
import fair_compare as fc
from benchmark import run_edger, run_deseq2, run_limma_voom, run_wilcoxon, run_deseq2_scored
from lfc_accuracy import run_deseq2_mle_and_shrunk, mle_shrunk_scored
from gtex_realdata_eval import ctx_matrix
from gtex_sex_eval import load_tissue_sex, KNOWN_SEX_GENES

SEQC = os.path.join(os.environ.get("PRESET_DATA", os.path.join(ROOT, "data")), "seqc")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
ap.add_argument("--gpu", type=int, default=-1)
ap.add_argument(
    "--data",
    nargs="+",
    default=["seqc", "gtex_sex"],
    choices=["seqc", "gtex_sex", "mouse_genotype", "tcga", "ipf", "sepsis", "crohn", "skin2"],
    help="all but seqc and gtex_sex come from heldout_sets.py",
)
ap.add_argument(
    "--ref_min", type=int, default=8, help="minimum reference samples per group; smaller splits are skipped"
)
ap.add_argument(
    "--preset_input",
    choices=["package", "context", "filtered"],
    default="package",
    help="package: preset.analyze; context: genes with a count (+ low-count cap); filtered: edgeR filterByExpr genes only",
)
ap.add_argument(
    "--preset_only", action="store_true", help="skip DESeq2, limma and apeglm (edgeR defines the scored set)"
)
ap.add_argument(
    "--preset_variants", action="store_true", help="also score PRESET by sign confidence and by P(|LFC|>0.1)"
)
ap.add_argument(
    "--ref_pos_lfc",
    type=float,
    default=1.0,
    help="reference positive: FDR<0.05 in >=2 of 3 methods and |LFC| > this (0 = FDR only)",
)
ap.add_argument(
    "--ref_neg_lfc",
    type=float,
    default=0.25,
    help="reference negative: FDR>0.5 in all methods and |LFC| < this (inf = FDR only)",
)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--fdr", type=float, default=0.05)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--ns", nargs="+", type=int, default=[6, 8, 12, 24])
ap.add_argument(
    "--ks",
    nargs="+",
    type=int,
    default=[50, 200, 500, 1000, 2000],
    help="top-k depths (top-50 saturates on these data)",
)
ap.add_argument(
    "--supplement", action="store_true", help="also run Wilcoxon and EBSeq (tied statistics; supplement only)"
)
ap.add_argument("--jobs", type=int, default=16, help="parallel workers")
ap.add_argument(
    "--max_genes",
    type=int,
    default=0,
    help="0 = every expressed gene; >0 = random subsample (debugging only)",
)
ap.add_argument(
    "--official",
    action="store_true",
    help="user's-guide workflows (edgeR filterByExpr + robust, limma filterByExpr, DESeq2 Cook's); score on edgeR filterByExpr genes",
)
ap.add_argument("--tag", default="")
ap.add_argument("--output_dir", default=None)
a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 else "cpu")
_M = {}


def model():
    if "m" not in _M:
        torch.set_num_threads(1)
        _M["m"] = preset.load_model(a.model, dev)[0]
    return _M["m"]


_, _, meta = preset.load_model(a.model, "cpu")
R = a.output_dir or RESULTS
os.makedirs(R, exist_ok=True)
print(f"model head={meta['head']} step={meta['step']}  tau={a.tau}", flush=True)
TASKS = []
REFS = []
# SEQC and GTEx informed the training prior; other sets: heldout_sets.STATUS
DATA_STATUS = {"SEQC": "design-exposed", "GTEx": "design-exposed"}


def run_methods(sub, cond, E, genes):
    """sub: genes x samples (full matrix), E: scored gene indices. Returns {method: (score, higher_is_better, lfc)}."""

    # default: preset.analyze with its own gene filter on sub[E]
    def _pkg(tau_):
        import preset

        ra = preset.analyze(pd.DataFrame(sub[E].T), cond, model=model(), tau=tau_, decoy=False)
        ra.loc[~ra.reported.to_numpy(bool), ["p_de"] + [c_ for c_ in ("p_positive",) if c_ in ra]] = np.nan
        return ra

    if a.preset_input == "package":
        r = _pkg(a.tau)
    # sensitivity: only genes kept by edgeR filterByExpr
    elif a.preset_input == "filtered":
        S0 = sub[E]
        e0 = run_edger(S0.T, cond, np.zeros_like(cond), tau=a.tau, filter_expr=True, robust=True)
        ev0 = np.isfinite(e0.logFC.to_numpy(float)) if e0 is not None else np.ones(len(E), bool)
        rf = preset.run(S0[ev0].T, cond, model=model(), tau=a.tau, decoy=False)
        r = pd.DataFrame(np.nan, index=range(len(E)), columns=rf.columns)
        r.loc[np.where(ev0)[0], :] = rf.to_numpy()
    else:
        cm, cp = ctx_matrix(sub, np.arange(sub.shape[1]), E, "auto")
        rep = np.zeros(len(cm), bool)
        rep[cp] = True
        r = (
            preset.run(cm.T, cond, model=model(), tau=a.tau, report=rep, decoy=False)
            .iloc[cp]
            .reset_index(drop=True)
        )
    out = {"PRESET": (r.p_de.to_numpy(), True, r.lfc_mean.to_numpy())}
    # other rankings from the same model, for small effects
    if a.preset_variants:
        r0 = (
            r
            if a.tau == 0
            else (
                _pkg(0.0)
                if a.preset_input == "package"
                else preset.run(cm.T, cond, model=model(), tau=0.0, report=rep, decoy=False)
                .iloc[cp]
                .reset_index(drop=True)
            )
        )
        # P(theta > 0), P(theta != 0) = 1 - pi0
        pp = r0.p_positive.to_numpy()
        pd0 = r0.p_de.to_numpy()
        # sign confidence = 1 - lfsr, without the spike's share
        out["PRESET sign"] = (np.maximum(pp, pd0 - pp), True, r.lfc_mean.to_numpy())
        r1 = (
            _pkg(0.1)
            if a.preset_input == "package"
            else preset.run(cm.T, cond, model=model(), tau=0.1, report=rep, decoy=False)
            .iloc[cp]
            .reset_index(drop=True)
        )
        out["PRESET tau0.1"] = (r1.p_de.to_numpy(), True, r1.lfc_mean.to_numpy())
    S = sub[E]
    b = np.zeros_like(cond)
    off = a.official
    e = run_edger(S.T, cond, b, tau=a.tau, filter_expr=off, robust=off)
    if e is not None:
        out["edgeR TREAT"] = (e.PValue.to_numpy(float), False, e.logFC.to_numpy(float))
    # common scored set
    ev = np.isfinite(e.logFC.to_numpy(float)) if (off and e is not None) else np.ones(len(E), bool)

    # expand a result on S[ev] to len(E)
    def back(x):
        y = np.full(len(E), np.nan)
        y[ev] = x
        return y

    if not a.preset_only:
        d2 = run_deseq2_scored(S.T, cond, b, ev, tau=a.tau, cooks=off)  # R mode fits all genes, scores ev
        if d2 is not None:
            out["DESeq2 lfcThr"] = (d2.pvalue.to_numpy(float), False, d2.log2FoldChange.to_numpy(float))
    if not a.preset_only:
        lv = run_limma_voom(S.T, cond, b, tau=a.tau, filter_expr=off)
        if lv is not None:
            out["limma treat"] = (
                np.where(ev, lv.PValue.to_numpy(float), np.nan),
                False,
                np.where(ev, lv.logFC.to_numpy(float), np.nan),
            )
    if not a.preset_only:
        # 1 core: parallelism comes from --jobs
        _, apg = mle_shrunk_scored(S.T, cond, b, ev, a.tau, 1, cooks=off)
        if apg is not None:
            out["apeglm"] = (np.where(ev, np.abs(np.nan_to_num(apg)), np.nan), True, np.nan_to_num(apg))
    if off:
        for k_ in [k_ for k_ in out if k_.startswith("PRESET")]:
            sc, hi, l_ = out[k_]
            out[k_] = (np.where(ev, sc, np.nan), hi, l_)
    if a.supplement:
        w = run_wilcoxon(S.T, cond, b, tau=0.0)
        if w is not None:
            out["Wilcoxon"] = (w.PValue.to_numpy(float), False, w.logFC.to_numpy(float))
        from benchmark import run_ebseq

        eb = run_ebseq(S.T, cond)
        if eb is not None:
            out["EBSeq"] = (eb.PPDE.to_numpy(float), True, eb.logFC.to_numpy(float))
    return out


def score_cell(dataset, n, rep, sub, cond, E, genes, pos, neg, known=None):
    """known: genes whose status is known without a DE tool (chrY genes, XIST), scored when the consensus has few positives."""
    res = run_methods(sub, cond, E, genes)
    mc = sub[E].mean(1)
    MET = []
    SCORES = []
    for nm, (sc, hi, lfc) in res.items():
        SCORES.extend(fc.gene_score_rows(dataset, n, rep, nm, genes, sc, lfc, mc, hi))
        order = fc.rank_score(sc, lfc, mc, higher_is_better=hi)
        row = {
            "dataset": dataset,
            "n": n,
            "rep": rep,
            "method": nm,
            "tau": a.tau,
            "status": DATA_STATUS.get(dataset.split()[0], "unknown"),
            "n_pos": int(pos.sum()),
            "n_neg": int(neg.sum()),
            "n_genes": len(E),
            "sweep_power": fc.sweep_power(order, pos, neg, a.fdr),
            "pAUPRC_0.2": fc.partial_auprc(order, pos, neg, 0.2),
            "pAUPRC_0.5": fc.partial_auprc(order, pos, neg, 0.5),
            "pAUPRC_1.0": fc.partial_auprc(order, pos, neg, 1.0),
            "prec@npos": fc.top_k_precision(order, pos, neg, max(1, int(pos.sum()))),
            "insufficient_positives": bool(pos.sum() < fc.MIN_POSITIVES),
            "supplement_only": nm in fc.SUPPLEMENT_ONLY,
        }
        if known is not None and known.any():
            # fixed before the first run; per-rep values kept since one of 16 genes moves recall by 6 points
            kk = int(known.sum())
            rk = np.empty(len(order))
            rk[order] = np.arange(1, len(order) + 1)
            for k in (50, 100):
                row[f"known_recall@{k}"] = float(known[order[: min(k, len(order))]].sum() / kk)
            row["known_median_rank"] = float(np.median(rk[known]))
            row["n_known"] = kk
        for k in a.ks:
            row[f"prec@{k}"] = fc.top_k_precision(order, pos, neg, min(k, len(order)))
        # expected false discoveries in the top k, and the counts by the reference
        if nm == "PRESET":
            p_ = np.nan_to_num(np.asarray(sc, float), nan=0.0)
            for k in (50, 100, 200):
                top = order[: min(k, len(order))]
                row[f"exp_false_k{k}"] = float((1 - p_[top]).sum())
                row[f"real_false_lower_k{k}"] = int(neg[top].sum())
                row[f"real_false_upper_k{k}"] = int((~pos[top]).sum())
        MET.append(row)
    return MET, SCORES


# --- SEQC: tool-independent truth
if "seqc" in a.data:
    C = pd.read_csv(os.path.join(SEQC, "seqc_gene_counts_ABCD.tsv"), sep="\t", index_col=0)
    T = pd.read_csv(os.path.join(SEQC, "seqc_taqman_truth.tsv"), sep="\t", dtype={"EntrezID": str}).set_index(
        "EntrezID"
    )
    X = C.to_numpy(np.int64)
    gn = C.index.to_numpy().astype(str)
    libs = C.columns.to_numpy()
    for key, (ctl, case, tcol) in {
        "SEQC D/C": ("C", "D", "lfc_DC_mix"),
        "SEQC A/B": ("A", "B", "lfc_BA"),
    }.items():
        ci = [i for i, l in enumerate(libs) if l.startswith(ctl + "_")]
        ki = [i for i, l in enumerate(libs) if l.startswith(case + "_")]
        for npg in (3, 4):
            # every balanced choice of npg of the 5 libraries is a replicate
            combos = [
                (c_, k_) for c_ in itertools.combinations(ci, npg) for k_ in itertools.combinations(ki, npg)
            ]
            rr = np.random.default_rng(0)
            combos = [combos[i] for i in rr.permutation(len(combos))[: a.reps]]
            for rep, (c_, k_) in enumerate(combos):
                idx = np.array(c_ + k_)
                cond = np.array([0] * npg + [1] * npg)
                sub = X[:, idx]
                E = np.where(fc.expressed_mask(sub))[0]
                g = gn[E]
                tv = T[tcol].reindex(g).to_numpy(float)
                lab = np.isfinite(tv)
                if tcol == "lfc_BA":
                    lab &= T["present_AB"].reindex(g).fillna(False).to_numpy(bool)
                pos = lab & (np.abs(np.nan_to_num(tv)) > 1.0)
                neg = lab & (np.abs(np.nan_to_num(tv)) < 0.25)
                TASKS.append((key, 2 * npg, rep, sub, cond, E, g, pos, neg, None))

# --- GTEx sex: consensus reference
if "gtex_sex" in a.data:
    for tis in ("lung", "thyroid", "whole_blood", "muscle_skeletal"):
        sym, XL, sex = load_tissue_sex(tis, donor_split="eval")
        male = np.where(sex == 1)[0]
        female = np.where(sex == 2)[0]
        rr = np.random.default_rng(0)
        nref = 60
        rm = rr.choice(male, min(nref, len(male)), replace=False)
        rf = rr.choice(female, min(nref, len(female)), replace=False)
        ridx = np.concatenate([rf, rm])
        rcond = np.array([0] * len(rf) + [1] * len(rm))
        rsub = XL[:, ridx]
        E = np.where(fc.expressed_mask(rsub))[0]
        if a.max_genes and len(E) > a.max_genes:
            E = np.sort(rr.choice(E, a.max_genes, replace=False))
        # reference on the same E, so TMM and dispersion fits see the whole library
        stats = {}
        for nm, f in (("edgeR", run_edger), ("limma", run_limma_voom), ("wilcoxon", run_wilcoxon)):
            rres = f(rsub[E].T, rcond, np.zeros_like(rcond), tau=0.0)
            if rres is not None:
                stats[nm] = (rres["FDR"].to_numpy(float), rres["logFC"].to_numpy(float))
        pos, neg = fc.consensus_reference(stats, pos_lfc=a.ref_pos_lfc, neg_lfc=a.ref_neg_lfc)
        known = np.isin(sym[E], KNOWN_SEX_GENES)
        print(
            f"{tis}: reference {len(rf)}F+{len(rm)}M over {len(E)} genes"
            + (
                f" (RANDOM subsample of {int(fc.expressed_mask(rsub).sum())} expressed)"
                if a.max_genes
                else ""
            )
            + f", {len(stats)} methods, pos {pos.sum()}, neg {neg.sum()}, known sex genes {known.sum()}/{len(KNOWN_SEX_GENES)}"
            + (
                "  [pAUPRC skipped: positives < %d]" % fc.MIN_POSITIVES
                if pos.sum() < fc.MIN_POSITIVES
                else ""
            ),
            flush=True,
        )
        poolm = np.setdiff1d(male, rm)
        poolf = np.setdiff1d(female, rf)
        for n in a.ns:
            h = n // 2
            if len(poolm) < h or len(poolf) < h:
                continue
            for rep in range(a.reps):
                r2 = np.random.default_rng(1000 * n + rep)
                idx = np.concatenate([r2.choice(poolf, h, replace=False), r2.choice(poolm, h, replace=False)])
                cond = np.array([0] * h + [1] * h)
                TASKS.append((f"GTEx sex {tis}", n, rep, XL[:, idx], cond, E, sym[E], pos, neg, known))
# --- held-out and other real cohorts
HELD = [d for d in a.data if d not in ("seqc", "gtex_sex")]
if HELD:
    import heldout_sets as hs

    for src in HELD:
        for name, Xc, ctrl, case, batch, gnames in hs.comparisons(src):
            # no shared samples: with a batch column each batch is the test pool once; otherwise each group is halved
            splits = []
            if batch is not None:
                for b in np.unique(batch):
                    splits.append((np.where(batch != b)[0], np.where(batch == b)[0], f"{name} [batch {b}]"))
            else:
                # crc32 of the name gives the same split in every process
                rs = np.random.default_rng(zlib.crc32(name.encode()))
                rc, kc = rs.permutation(ctrl), rs.permutation(case)
                ref = np.concatenate([rc[: len(rc) // 2], kc[: len(kc) // 2]])
                pool = np.concatenate([rc[len(rc) // 2 :], kc[len(kc) // 2 :]])
                splits.append((ref, pool, name))
            is_case = np.zeros(Xc.shape[1], bool)
            is_case[case] = True
            for ref, pool, tag in splits:
                rc_, rk_ = ref[~is_case[ref]], ref[is_case[ref]]
                pc_, pk_ = pool[~is_case[pool]], pool[is_case[pool]]
                if min(len(rc_), len(rk_)) < a.ref_min:
                    print(
                        f"{tag}: skipped, reference side has {len(rc_)}/{len(rk_)} per group (< --ref_min {a.ref_min})",
                        flush=True,
                    )
                    continue
                ridx = np.concatenate([rc_, rk_])
                rcond = np.array([0] * len(rc_) + [1] * len(rk_))
                rsub = Xc[:, ridx]
                E = np.where(fc.expressed_mask(rsub))[0]
                # a reference spanning several batches gets batch in its design
                rb = pd.factorize(np.asarray(batch)[ridx])[0] if batch is not None else np.zeros_like(rcond)
                rkw = (
                    {"edgeR": dict(filter_expr=True, robust=True), "limma": dict(filter_expr=True)}
                    if a.official
                    else {}
                )
                stats = {}
                for nm, f in (("edgeR", run_edger), ("limma", run_limma_voom), ("wilcoxon", run_wilcoxon)):
                    rres = f(rsub[E].T, rcond, rb, tau=0.0, **rkw.get(nm, {}))
                    if rres is not None:
                        stats[nm] = (rres["FDR"].to_numpy(float), rres["logFC"].to_numpy(float))
                pos, neg = fc.consensus_reference(stats, pos_lfc=a.ref_pos_lfc, neg_lfc=a.ref_neg_lfc)
                print(
                    f"{tag}: reference {len(rc_)}v{len(rk_)} over {len(E)} genes, {len(stats)} methods, "
                    f"pos {pos.sum()}, neg {neg.sum()}, test pool {len(pc_)}v{len(pk_)}  [{hs.STATUS[src]}]"
                    + (
                        "  [pAUPRC skipped: positives < %d]" % fc.MIN_POSITIVES
                        if pos.sum() < fc.MIN_POSITIVES
                        else ""
                    ),
                    flush=True,
                )
                DATA_STATUS[tag.split()[0]] = hs.STATUS[src]
                genes = np.asarray(gnames)[E]
                # reference labels for per-gene follow-ups
                REFS.append(
                    pd.DataFrame(
                        {
                            "dataset": tag,
                            "gene": genes,
                            "ref_pos": pos,
                            "ref_neg": neg,
                            **{f"ref_{nm}_lfc": v[1] for nm, v in stats.items()},
                            **{f"ref_{nm}_fdr": v[0] for nm, v in stats.items()},
                        }
                    )
                )
                for n in a.ns:
                    h = n // 2
                    if min(len(pc_), len(pk_)) < h:
                        continue
                    for rep in range(a.reps):
                        r2 = np.random.default_rng(1000 * n + rep + (zlib.crc32(tag.encode()) % 1000))
                        idx = np.concatenate(
                            [r2.choice(pc_, h, replace=False), r2.choice(pk_, h, replace=False)]
                        )
                        TASKS.append(
                            (tag, n, rep, Xc[:, idx], np.array([0] * h + [1] * h), E, genes, pos, neg, None)
                        )

print(f"{len(TASKS)} cells over {a.jobs} workers", flush=True)
parts = Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(delayed(score_cell)(*t) for t in TASKS)
MET = [r for mm, _ in parts for r in mm]
SCORES = [r for _, ss in parts for r in ss]

met = pd.DataFrame(MET)
met.to_csv(os.path.join(R, f"fair_bench{a.tag}_metrics.csv"), index=False)
pd.DataFrame(SCORES).to_csv(os.path.join(R, f"fair_bench{a.tag}_scores.csv"), index=False)
if REFS:
    pd.concat(REFS, ignore_index=True).to_csv(
        os.path.join(R, f"fair_bench{a.tag}_reflabels.csv"), index=False
    )
pd.set_option("display.width", 220)
main = met[~met.supplement_only]
for metric in [
    "pAUPRC_0.2",
    "pAUPRC_0.5",
    "pAUPRC_1.0",
    "prec@npos",
    "prec@500",
    "prec@2000",
    "sweep_power",
] + [c for c in main.columns if c.startswith("known_")]:
    if metric not in main.columns:
        continue
    t = main.pivot_table(index=["dataset", "n"], columns="method", values=metric)
    sd = main.pivot_table(index=["dataset", "n"], columns="method", values=metric, aggfunc="std")
    if t.dropna(how="all").empty:
        continue
    print(f"\n-- {metric} (mean)")
    print(t.round(3).to_string())
    if sd.notna().to_numpy().any():
        print(f"   (standard deviation over subsample repeats)")
        print(sd.round(3).to_string())
print(f"\nwritten: results/fair_bench{a.tag}_metrics.csv (+_scores.csv, {len(SCORES)} rows)")
