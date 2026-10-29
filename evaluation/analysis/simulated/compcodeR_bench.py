"""Score PRESET and the conventional methods on simulated datasets with known truth: compcodeR (compcodeR_generate.R) or
SPsimSeq (spsimseq_generate.R, same file layout). Per design, group size, replicate and method: realized FDR and power at the
method's cut-off, power at realized FDR 5%, CAT curves, LFC slopes, false 4-fold reports and PRESET's 90% interval coverage.
    python analysis/simulated/compcodeR_bench.py --data_dir results/compcodeR_data --gpu -1 --jobs 40 --nullstrap
"""

import os, sys, glob, argparse, warnings, numpy as np, pandas as pd, torch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
RESULTS = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
ANALYSIS = os.path.join(ROOT, "analysis")
for p_ in [ROOT] + [os.path.join(ANALYSIS, d) for d in sorted(os.listdir(ANALYSIS))]:
    if p_ not in sys.path:
        sys.path.insert(0, p_)
import preset, fair_compare as fc
from benchmark import run_edger, run_deseq2, run_limma_voom
from lfc_accuracy import run_deseq2_mle_and_shrunk, mle_shrunk_scored
from benchmark import run_deseq2_scored
from gtex_realdata_eval import ctx_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--gpu", type=int, default=-1)
ap.add_argument("--nullstrap", action="store_true")
ap.add_argument("--output_dir", default=os.path.join(RESULTS, "compcodeR_bench"))
ap.add_argument(
    "--preset_decoy",
    action="store_true",
    help="add 'PRESET decoy': the package default rule (target-decoy FDR 5%%)",
)
ap.add_argument(
    "--preset_only",
    action="store_true",
    help="skip DESeq2, limma and apeglm (edgeR still runs: it defines the scored genes)",
)
A = ap.parse_args()
os.makedirs(A.output_dir, exist_ok=True)
KS = (50, 100, 200, 500)
_M = {}


def model():
    if "m" not in _M:
        torch.set_num_threads(1)
        _M["m"] = preset.load_model(A.ckpt, torch.device("cpu"))[0]
    return _M["m"]


def slopes(est, true, de):
    m = de & np.isfinite(est)
    if m.sum() < 20:
        return np.nan, np.nan
    return np.polyfit(est[m], true[m], 1)[0], np.polyfit(true[m], est[m], 1)[0]


def one(tag):
    cnt = pd.read_csv(os.path.join(A.data_dir, tag + "_counts.tsv"), sep="\t", index_col=0)
    tr = pd.read_csv(os.path.join(A.data_dir, tag + "_truth.tsv"), sep="\t").set_index("gene").loc[cnt.index]
    sa = pd.read_csv(os.path.join(A.data_dir, tag + "_samples.tsv"), sep="\t")
    # genes x samples, condition 1 = control
    cond = (sa.condition.to_numpy() == 2).astype(int)
    X = cnt.to_numpy(np.int64)
    true = tr.true_log2fc.to_numpy(float)
    de = tr.de.to_numpy() == 1
    pos = de & (np.abs(true) > A.tau)
    neg = ~de
    E = np.where(fc.expressed_mask(X))[0]
    S = X[E]
    t, p_, n_ = true[E], pos[E], neg[E]
    b = np.zeros_like(cond)
    design, spc, rep = (
        tag.split("_")[0] + "_" + tag.split("_")[1] + "_" + tag.split("_")[2],
        int(tag.split("_")[3].replace("spc", "")),
        int(tag.split("_")[4][1:]),
    )
    out = {}
    # package default on the tools' matrix
    import preset, zlib
    from gtex_realdata_eval import preset_default

    r = preset_default(
        S,
        cond,
        model(),
        tau=A.tau,
        intervals=True,
        decoy=A.preset_decoy,
        decoy_seed=zlib.crc32(tag.encode()) % (1 << 30),
    )
    out["PRESET"] = dict(
        score=r.p_de.to_numpy(),
        hi=True,
        lfc=r.lfc_mean.to_numpy(),
        call=(r.p_de > 0.9).to_numpy(),
        lo=r.lfc_lo90.to_numpy(),
        up=r.lfc_hi90.to_numpy(),
    )
    out["PRESET bfdr"] = dict(
        score=r.p_de.to_numpy(),
        hi=True,
        lfc=r.lfc_mean.to_numpy(),
        call=np.asarray(preset.calls(r, fdr=0.05, rule="bfdr"), bool),
    )
    if A.preset_decoy:
        out["PRESET decoy"] = dict(
            score=r.p_de.to_numpy(),
            hi=True,
            lfc=r.lfc_mean.to_numpy(),
            call=np.asarray(preset.calls(r, fdr=0.05, rule="decoy"), bool),
        )
    e = run_edger(S.T, cond, b, A.tau, filter_expr=True, robust=True)
    ev = np.isfinite(e.logFC.to_numpy(float)) if e is not None else np.ones(len(E), bool)
    if e is not None:
        out["edgeR TREAT"] = dict(
            score=e.PValue.to_numpy(float),
            hi=False,
            lfc=e.logFC.to_numpy(float),
            call=np.nan_to_num(e.FDR.to_numpy(float), nan=1) < 0.05,
        )

    def back(x):
        y = np.full(len(E), np.nan)
        y[ev] = x
        return y

    if A.preset_only:
        d2 = lv = apg = None
    # DESeq2 fits all genes, scored on the edgeR set
    else:
        d2 = run_deseq2_scored(S.T, cond, b, ev, A.tau, cooks=True)
    if d2 is not None:
        out["DESeq2 lfcThr"] = dict(
            score=d2.pvalue.to_numpy(float),
            hi=False,
            lfc=d2.log2FoldChange.to_numpy(float),
            call=np.nan_to_num(d2.padj.to_numpy(float), nan=1) < 0.05,
        )
    if not A.preset_only:
        lv = run_limma_voom(S.T, cond, b, A.tau, filter_expr=True)
    if lv is not None:
        out["limma treat"] = dict(
            score=lv.PValue.to_numpy(float),
            hi=False,
            lfc=lv.logFC.to_numpy(float),
            call=np.nan_to_num(
                lv["adj.P.Val"].to_numpy(float) if "adj.P.Val" in lv else lv.FDR.to_numpy(float), nan=1
            )
            < 0.05,
        )
    if not A.preset_only:
        _, apg = mle_shrunk_scored(S.T, cond, b, ev, A.tau, 1, cooks=True)
    if apg is not None:
        out["apeglm"] = dict(
            score=np.abs(np.nan_to_num(apg)), hi=True, lfc=np.nan_to_num(apg), call=np.zeros(len(E), bool)
        )
    # the estimate the DESeq2 vignette reports
    if apg is not None and "DESeq2 lfcThr" in out:
        out["DESeq2 lfcThr"]["lfc_shrunk"] = apg
    if A.nullstrap and not A.preset_only:
        from benchmark import run_nullstrap_edger

        ns = run_nullstrap_edger(S.T, cond, b, fdr=0.05)
        if ns is not None:
            out["edgeR+Nullstrap"] = dict(
                score=e.PValue.to_numpy(float), hi=False, lfc=e.logFC.to_numpy(float), call=ns
            )
    rows = []
    mc = S.mean(1)
    order_true = np.argsort(-np.abs(t))
    for nm, o in out.items():
        sc = np.where(ev, o["score"], np.nan)
        order = fc.rank_score(sc, o["lfc"], mc, higher_is_better=o["hi"])
        call = o["call"] & ev
        row = dict(
            design=design,
            spc=spc,
            rep=rep,
            method=nm,
            n_scored=int(ev.sum()),
            n_pos=int(p_[ev].sum()),
            n_called=int(call.sum()),
            # 0 when nothing is called
            op_fdr=float(n_[call].mean()) if call.any() else 0.0,
            op_power=float((call & p_).sum() / max(1, p_[ev].sum())),
            sweep_power=fc.sweep_power(order, p_, n_, 0.05),
            false4=int((np.abs(np.nan_to_num(o["lfc"])) >= 2)[n_ & ev].sum()),
            false4_called=int(((np.abs(np.nan_to_num(o["lfc"])) >= 2) & call)[n_ & ev].sum()),
            false_calls=int(call[n_ & ev].sum()),
            false4_top200=int((np.abs(np.nan_to_num(o["lfc"][order[:200]])) >= 2)[n_[order[:200]]].sum()),
            # DESeq2: same top 200, apeglm estimates
            false4_top200_shrunk=int(
                (np.abs(np.nan_to_num(o.get("lfc_shrunk", o["lfc"])[order[:200]])) >= 2)[
                    n_[order[:200]]
                ].sum()
            ),
        )
        for k in KS:
            row[f"cat@{k}"] = float(len(set(order[:k]) & set(order_true[:k])) / k)
        # CAT by |log2FC|, unscored genes last
        order_lfc = np.argsort(-np.where(ev, np.abs(np.nan_to_num(o["lfc"])), -np.inf))
        for k in KS:
            row[f"cat_lfc@{k}"] = float(len(set(order_lfc[:k]) & set(order_true[:k])) / k)
        row["calib_slope"], row["atten_slope"] = slopes(o["lfc"], t, p_ & ev)
        if "lo" in o:
            fin = np.isfinite(o["lo"])
            cov = (o["lo"] <= t) & (t <= o["up"])  # genes PRESET reports
            row["cov90_de"] = float(cov[p_ & ev & fin].mean())
            row["cov90_null"] = float(cov[n_ & ev & fin].mean())
        rows.append(row)
    return rows


tags = sorted(
    set(
        os.path.basename(f)[: -len("_counts.tsv")]
        for f in glob.glob(os.path.join(A.data_dir, "*_counts.tsv"))
    )
)
print(f"{len(tags)} datasets over {A.jobs} workers", flush=True)
if A.jobs > 1:
    from joblib import Parallel, delayed

    res = Parallel(n_jobs=A.jobs, verbose=5)(delayed(one)(t) for t in tags)
else:
    res = [one(t) for t in tags]
df = pd.DataFrame([r for rs in res for r in rs])
df.to_csv(os.path.join(A.output_dir, "compcodeR_bench.csv"), index=False)
pd.set_option("display.width", 250)
print(
    df.groupby(["design", "spc", "method"])[
        ["op_fdr", "op_power", "sweep_power", "calib_slope", "atten_slope", "false4", "cat@100"]
    ]
    .mean()
    .round(3)
    .to_string()
)
