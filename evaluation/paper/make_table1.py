"""Table 1: PRESET against edgeR, DESeq2 and limma, grouped by the kind of truth: Simulated (compcodeR, SPsimSeq), Planted (real GTEx
counts, effects planted by binomial thinning), Real (ERCC and TaqMan measurements, or the remaining samples of a large cohort).
Cells are "value ± half-width of the 95% percentile bootstrap CI". Bootstrap units: simulated dataset (stratified by design); GTEx
subsample (stratified by tissue); ERCC 3 vs 3 subsample; TaqMan gene (pooled RMSE recomputed per resample); held-out cohorts,
subsample within each comparison (subsampling uncertainty for these cohorts, not an interval over cohorts).
Bold = best value when the paired bootstrap CI of (best - runner-up) excludes 0; FDR rows bold nothing (the target is 0.05).
"n" in the result files is the total of two balanced groups; compcodeR's spc is per group. Missing sources print as \\todo{--}.
Writes $PRESET_OUTPUT/tables/table1.{tex,csv}.
    python make_table1.py [--tag v4a] [--boot 2000]
"""

import os, glob, argparse, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get("PRESET_RESULTS", os.path.join(ROOT, "results"))
TAB_DIR = os.environ.get("PRESET_OUTPUT", os.path.join(ROOT, "output")) + "/tables"
os.makedirs(TAB_DIR, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v4a")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
T = a.tag
COLS = ["PRESET", "edgeR", "DESeq2", "limma"]
rows = []
missing = []
SIM, PLANT, REAL = "Simulated", "Real counts, planted effects", "Real counts, real effects"
SWEEP = {"PRESET": "PRESET", "edgeR": "edgeR", "DESeq2": "DESeq2", "limma-voom": "limma"}
# FDR rows use PRESET's default call rule (target-decoy); Power rows rank by P(DE)
SWEEP_DC = {"PRESET decoy": "PRESET", "edgeR": "edgeR", "DESeq2": "DESeq2", "limma-voom": "limma"}
FAIR = {"PRESET": "PRESET", "edgeR TREAT": "edgeR", "DESeq2 lfcThr": "DESeq2", "limma treat": "limma"}
rng = np.random.default_rng(20261009)


def src(p):
    if os.path.exists(p):
        return True
    missing.append(p)
    return False


def concat(pattern, tagcol=None):
    fs = glob.glob(pattern)
    if not fs:
        missing.append(pattern)
        return None
    out = []
    for f in fs:
        d = pd.read_csv(f)
        d["file"] = f
        out.append(d)
    return pd.concat(out, ignore_index=True)


def boot_rows(wide, kind, better):
    """wide: one row per unit, a 'stratum' column and one column per method (kind 'mean') or (method, 'sse'/'cnt') columns (kind 'rmse').
    Returns {method: (value, lo, hi)} and the paired best-vs-runner-up CI."""
    strata = wide["stratum"].to_numpy()
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    def stat(idx):
        if kind == "mean":
            return {m: np.nanmean(wide[m].to_numpy()[idx]) for m in meths}
        return {
            m: np.sqrt(
                np.nansum(wide[(m, "sse")].to_numpy()[idx]) / np.nansum(wide[(m, "cnt")].to_numpy()[idx])
            )
            for m in meths
        }

    meths = [m for m in COLS if (m in wide.columns if kind == "mean" else (m, "sse") in wide.columns)]
    full = stat(np.arange(len(wide)))
    bs = {m: [] for m in meths}
    for _ in range(a.boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        s = stat(idx)
        for m in meths:
            bs[m].append(s[m])
    res = {m: (full[m], *np.nanpercentile(bs[m], [2.5, 97.5])) for m in meths}
    sig = None
    if better and len(meths) > 1:
        order = sorted(meths, key=lambda m: full[m], reverse=(better == "max"))
        b, r = order[0], order[1]
        d = np.array(bs[b]) - np.array(bs[r])
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        sig = dict(best=b, runner=r, diff=full[b] - full[r], lo=lo, hi=hi, excl0=bool(lo > 0 or hi < 0))
    return res, sig


DAG = ""


def add(section, dataset, npg, metric, wide=None, kind="mean", better=None, fmt="{:.2f}", nounc=()):
    r = dict(
        section=section,
        dataset=dataset,
        npg=npg,
        metric=metric,
        better=better,
        fmt=fmt,
        nounc=tuple(nounc),
        sig=None,
    )
    if wide is not None and len(wide):
        res, sig = boot_rows(wide, kind, better)
        r["sig"] = sig
        for m, (v, lo, hi) in res.items():
            r[m], r[m + "_lo"], r[m + "_hi"] = v, lo, hi
    rows.append(r)


def mean_wide(d, unit_cols, method_col, value_col, M, stratum_col=None):
    d = d[d[method_col].isin(M)].copy()
    d["m"] = d[method_col].map(M)
    w = d.pivot_table(index=unit_cols, columns="m", values=value_col, aggfunc="mean").reset_index()
    w["stratum"] = w[stratum_col] if stratum_col else 0
    return w


F = "_fair"  # runs with Bioconductor DESeq2, no shared pre-filter, continuous sweep, tau-aligned FDR


def first(*ps):
    for p_ in ps:
        if os.path.exists(p_):
            return p_
    missing.append(ps[0])
    return None


# Every method gets the same count matrix and applies its own gene filter; PRESET runs through preset.analyze with
# its default rule. The PRESET-only reruns (*_pkg) reuse the tool rows of the fair runs on the same subsamples.
PK = "_pkg"


def with_tools(new, old, keys, tool_methods, check=("edgeR TREAT", "sweep_power")):
    """PRESET* rows from `new`, tool rows from `old`; asserts the same units and identical edgeR rows (same subsamples)."""
    mcol, ccol = "method", check[1]
    a_ = new[new[mcol] == check[0]].merge(old[old[mcol] == check[0]], on=keys, suffixes=("", "_old"))
    assert len(a_) == (old[mcol] == check[0]).sum() == (new[mcol] == check[0]).sum() and np.allclose(
        a_[ccol].fillna(-1), a_[ccol + "_old"].fillna(-1)
    ), "PRESET rerun is not on the same subsamples as the tool run"
    return pd.concat(
        [new[new[mcol].str.startswith("PRESET")], old[old[mcol].isin(tool_methods)]], ignore_index=True
    )


FAIR_DC = {"PRESET decoy": "PRESET", "edgeR TREAT": "edgeR", "DESeq2 lfcThr": "DESeq2", "limma treat": "limma"}
TOOLS = ["edgeR TREAT", "DESeq2 lfcThr", "limma treat", "apeglm"]


def scored_power(d, col):
    """Power among the positives of the scored set (edgeR filterByExpr), as defined in the paper. compcodeR_bench.py divides by every
    positive with a count (n_pos_all); the others have no score, are ranked last and never enter a list with a realized FDR of 5%."""
    assert "n_pos_all" in d, "result file without n_pos_all; rerun simulated/compcodeR_bench.py"
    return d[col].to_numpy(float) * d.n_pos_all.to_numpy(float) / d.n_pos.clip(lower=1).to_numpy(float)


def fdr_excluding_small(d):
    """SPsimSeq FDR as pre-registered: DE genes with true |log2FC| <= tau are not scored, so they leave both the numerator and the
    denominator. op_fdr counts genes simulated without change over all calls and op_fdr_tau every gene with |log2FC| <= tau, so their
    difference times n_called is the number of calls on those DE genes."""
    nonde = d.op_fdr.to_numpy(float) * d.n_called.to_numpy(float)
    small = (d.op_fdr_tau.to_numpy(float) - d.op_fdr.to_numpy(float)) * d.n_called.to_numpy(float)
    assert np.allclose(nonde, np.round(nonde)) and np.allclose(small, np.round(small)), "call counts are not integers"
    labelled = d.n_called.to_numpy(float) - np.round(small)
    return np.where(labelled > 0, np.round(nonde) / np.maximum(labelled, 1), 0.0)


# ---- Simulated (no rows on PRESET's own simulator: its truth is drawn from the training prior)
p_old = first(f"{RES}/compcodeR_{T}{F}2/compcodeR_bench.csv")
p_new = first(f"{RES}/compcodeR_{T}{PK}/compcodeR_bench.csv")
if p_old and p_new:
    c_old = pd.read_csv(p_old)
    c_new = pd.read_csv(p_new)
    # all five designs of the original study
    c = with_tools(c_new, c_old, ["design", "spc", "rep"], TOOLS + ["DESeq2 lfcThr"])
    c = c[c.spc == 5]
    # DESeq2: its top 200 by p-value, counted on the apeglm estimates a DESeq2 user reports
    if "false4_top200_shrunk" in c:
        sh = c.method == "DESeq2 lfcThr"
        c.loc[sh, "false4_top200"] = c.loc[sh, "false4_top200_shrunk"]
    c["power_scored"] = scored_power(c, "sweep_power")
    add(
        SIM,
        "compcodeR",
        5,
        "FDR",
        mean_wide(c, ["design", "rep"], "method", "op_fdr", FAIR_DC, "design"),
        "mean",
        None,
    )
    add(
        SIM,
        "compcodeR",
        5,
        "Power",
        mean_wide(c, ["design", "rep"], "method", "power_scored", FAIR, "design"),
        "mean",
        "max",
    )
    add(
        SIM,
        "compcodeR",
        5,
        "False 4-fold",
        mean_wide(c, ["design", "rep"], "method", "false4_top200", FAIR, "design"),
        "mean",
        "min",
        "{:.1f}",
    )
else:
    add(SIM, "compcodeR", 5, "Power", better="max")
# SPsimSeq: semi-parametric, no negative binomial assumption; scored as compcodeR
p_sp = first(f"{RES}/spsimseq_{T}{PK}/compcodeR_bench.csv")
if p_sp:
    c = pd.read_csv(p_sp)
    c = c[c.spc == 4]
    if "false4_top200_shrunk" in c:
        sh = c.method == "DESeq2 lfcThr"
        c.loc[sh, "false4_top200"] = c.loc[sh, "false4_top200_shrunk"]
    # as pre-registered: about 7% of the DE genes have true |log2FC| <= tau and are not scored (compcodeR has none);
    # op_fdr_tau and sweep_power_tau count them as false instead (sensitivity analysis)
    c["fdr_scored"] = fdr_excluding_small(c)
    c["power_scored"] = scored_power(c, "sweep_power")
    add(
        SIM,
        "SPsimSeq",
        4,
        "FDR",
        mean_wide(c, ["design", "rep"], "method", "fdr_scored", FAIR_DC, "design"),
        "mean",
        None,
    )
    add(
        SIM,
        "SPsimSeq",
        4,
        "Power",
        mean_wide(c, ["design", "rep"], "method", "power_scored", FAIR, "design"),
        "mean",
        "max",
    )
    add(
        SIM,
        "SPsimSeq",
        4,
        "False 4-fold",
        mean_wide(c, ["design", "rep"], "method", "false4_top200", FAIR, "design"),
        "mean",
        "min",
        "{:.1f}",
    )

# ---- Planted: GTEx evaluation-half donors, effects planted by binomial thinning
E = "_evd_v2"
o = concat(f"{RES}/outl_{T}{F}{E}/*/sweep.csv")
for n in (8, 24):
    if o is None:
        add(PLANT, "GTEx lung + outliers" + DAG, n // 2, "FDR")
        continue
    d = o[(o.outlier_frac == 0.02) & (o.n == n) & (o.kind == "outlier") & (o.de_frac > 0)]
    add(
        PLANT,
        "GTEx lung + outliers" + DAG,
        n // 2,
        "FDR",
        mean_wide(d, ["file", "rep"], "method", "op_fdr", SWEEP_DC),
    )
TIS = ("lung", "whole_blood", "muscle_skeletal", "adipose_subcutaneous")
fs = sum((glob.glob(f"{RES}/nsweep_{t}_{T}{F}{E}/*/sweep.csv") for t in TIS), [])
if len(fs) >= 4 * 5 * 10:
    ns = pd.concat(
        [pd.read_csv(f).assign(file=f, tissue=os.path.relpath(f, RES).split(os.sep)[0]) for f in fs],
        ignore_index=True,
    )
    ns = ns[ns.n == 8]
    # Power with the FDR row's definition: true |lfc| >= 0.5 positive, < 0.5 negative
    add(
        PLANT,
        "GTEx, 4 tissues" + DAG,
        4,
        "Power",
        mean_wide(ns, ["tissue", "file", "rep"], "method", "sweep_power_tau", SWEEP, "tissue"),
        "mean",
        "max",
    )
    add(
        PLANT,
        "GTEx, 4 tissues" + DAG,
        4,
        "FDR",
        mean_wide(ns, ["tissue", "file", "rep"], "method", "op_fdr", SWEEP_DC, "tissue"),
        "mean",
        None,
    )
else:
    missing.append(f"nsweep_*_{T}{F}{E} ({len(fs)}/200 shards)")
    for metric, better in (("Power", "max"), ("FDR", None)):
        add(PLANT, "GTEx, 4 tissues" + DAG, 4, metric, better=better)

# ---- Real
p_old = first(f"{RES}/seqc_{T}{F}/ercc_summary.csv")
p = first(f"{RES}/seqc_{T}{PK}/ercc_summary.csv")
if p and p_old:
    e = pd.read_csv(p)
    e = e[e.cell.str.startswith("3v3")]
    eo = pd.read_csv(p_old)
    eo = eo[eo.cell.str.startswith("3v3")]
    chk = e[e.method == "edgeR TREAT"].merge(eo[eo.method == "edgeR TREAT"], on="cell", suffixes=("", "_old"))
    assert np.allclose(chk.called_A, chk.called_A_old), "ERCC rerun: tool rows differ"
    # no 1:1 row: after normalization on the endogenous genes the 1:1 spike-ins shift by about 0.72 log2 in
    # every method, so calls there follow the data rather than measure errors
    add(
        REAL,
        "SEQC ERCC spike-ins" + DAG,
        3,
        "Called (4:1)",
        mean_wide(e, ["cell"], "method", "called_A", FAIR_DC),
        "mean",
        "max",
    )
pg = first(f"{RES}/seqc_taqman_{T}{PK}_genes.csv")
if pg:
    g = pd.read_csv(pg)
    g = g[(g.contrast == "DC") & (g.n == 6)]
    parts = {}
    for m, cname in (
        ("PRESET", "PRESET"),
        ("edgeR", "edgeR_TREAT05"),
        ("DESeq2", "apeglm"),
        ("limma", "limma_treat05"),
    ):
        x = g[(g.method == cname) & g.est.notna()]
        se = ((x.est - x.true) ** 2).groupby(x.gene).sum()
        ct = x.groupby("gene").size()
        # unit = gene, summed over the subsamples
        parts[(m, "sse")] = se
        parts[(m, "cnt")] = ct.astype(float)
    w = pd.DataFrame(parts).dropna()
    w["stratum"] = 0
    add(REAL, "SEQC TaqMan qPCR" + DAG, 3, "LFC error", w, "rmse", "min", "{:.3f}")
DISEASE = [
    ("IPF vs control lung", "Pulmonary fibrosis (lung)"),
    ("CHP vs control lung", "Hypersens. pneumonitis (lung)"),
    ("sepsis vs healthy blood", "Sepsis (blood)"),
    ("Crohn's disease vs not-IBD", "Crohn's disease (ileum)"),
    ("psoriasis lesional vs healthy skin", "Psoriasis (skin)"),
    ("atopic dermatitis lesional", "Atopic dermatitis (skin)"),
]
KEYS = ["dataset", "n", "rep"]
p_old = first(f"{RES}/v4eval3_{T}{F}/fair_bench_v4eval3_{T}{F}_metrics.csv")
p = first(f"{RES}/v4eval3_{T}{PK}/fair_bench_v4eval3_{T}{PK}_metrics.csv")
if p and p_old:
    m = with_tools(pd.read_csv(p), pd.read_csv(p_old), KEYS, TOOLS)
    m = m[(m.n == 8) & ~m.insufficient_positives.astype(bool)]
    for key, label in DISEASE:
        d = m[m.dataset.str.contains(key, regex=False)]
        if d.empty:
            missing.append(f"v4eval3 dataset '{key}'")
            continue
        add(
            REAL,
            label,
            4,
            "Recall",
            mean_wide(d, ["dataset", "rep"], "method", "sweep_power", FAIR),
            "mean",
            "max",
        )
p_old = first(f"{RES}/v4eval_{T}{F}/fair_bench_v4eval_{T}{F}_metrics.csv")
p = first(f"{RES}/v4eval_{T}{PK}/fair_bench_v4eval_{T}{PK}_metrics.csv")
if p and p_old:
    m0 = with_tools(pd.read_csv(p), pd.read_csv(p_old), KEYS, TOOLS)
    m0 = m0[(m0.n == 8) & ~m0.insufficient_positives.astype(bool)]
    for name, filt in (("TCGA tumor types", "TCGA"), ("Mouse genotypes", "mouse")):
        # comparisons with too few reference positives drop out
        m = m0[m0.dataset.str.contains(filt)]
        k = m.dataset.nunique()
        add(
            REAL,
            f"{name} ({k})",
            4,
            "Recall",
            mean_wide(m, ["dataset", "rep"], "method", "sweep_power", FAIR, "dataset"),
            "mean",
            "max",
        )

# ---- outputs
t = pd.DataFrame(rows)
# row order within a section: error control, finding changes, effect sizes; the GTEx 4-tissue block first
M_RANK = {"FDR": 0, "Power": 1, "Recall": 1, "Called (4:1)": 2, "False 4-fold": 3, "LFC error": 3}
DS_FIRST = {"GTEx, 4 tissues" + DAG: -1}
t["_sec"] = t.section.map({s: i for i, s in enumerate(dict.fromkeys(t.section))})
t["_m"] = t.metric.map(M_RANK).fillna(9)
t["_appear"] = t.dataset.map({d: i for i, d in enumerate(dict.fromkeys(t.dataset))})
t["_ds"] = t.dataset.map(lambda d: DS_FIRST.get(d, None)).fillna(t.groupby("dataset")._m.transform("min"))
t = (
    t.sort_values(["_sec", "_ds", "_appear", "_m", "npg"], kind="stable")
    .drop(columns=["_sec", "_m", "_appear", "_ds"])
    .reset_index(drop=True)
)
flat = t.drop(columns=["fmt", "nounc", "sig"]).copy()
for k in ("best", "runner", "diff", "lo", "hi", "excl0"):
    flat[f"paired_{k}"] = [s.get(k) if isinstance(s, dict) else None for s in t.sig]
os.makedirs(TAB_DIR, exist_ok=True)
flat.to_csv(f"{TAB_DIR}/table1.csv", index=False)


def cell(r, c):
    v = r.get(c, np.nan)
    if pd.isna(v):
        return "\\todo{--}" if all(pd.isna(r.get(k, np.nan)) for k in COLS) else "--"
    s = r["fmt"].format(v)
    lo, hi = r.get(c + "_lo", np.nan), r.get(c + "_hi", np.nan)
    if not pd.isna(lo):
        h = (hi - lo) / 2
        hs = r["fmt"].format(h)
        hs = hs[1:] if hs.startswith("0.") else hs
        # e.g. "<.01" when the half-width rounds to zero
        if float(hs) == 0:
            hs = "$<$" + r["fmt"].format(10 ** -int(r["fmt"][3]))[1:]
        s += f"\\,{{\\scriptsize$\\pm${hs}}}"
    elif c in r["nounc"]:
        s += "$^\\dagger$"
    sig = r["sig"]
    if isinstance(sig, dict) and sig["best"] == c and sig["excl0"]:
        s = f"\\textbf{{{r['fmt'].format(v)}}}" + s[len(r["fmt"].format(v)) :]
    return s


CAT = {SIM: "Simulated", PLANT: "Planted", REAL: "Real"}
lines = [
    "\\setlength{\\tabcolsep}{4pt}\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}llclcccc@{}}\\toprule",
    "Effects & Dataset & $n$ & Metric & \\method & edgeR & DESeq2 & limma \\\\",
]
nsec = t.groupby("section", sort=False).size()
prev_sec, prev_key = None, None
for _, r in t.iterrows():
    r = r.to_dict()
    first_in_sec = r["section"] != prev_sec
    if first_in_sec:
        lines.append("\\midrule")
        prev_sec = r["section"]
        prev_key = None
    cat = f"\\multirow{{{nsec[r['section']]}}}{{*}}{{{CAT[r['section']]}}}" if first_in_sec else ""
    key = (r["dataset"], r["npg"])
    ds = r["dataset"] if r["dataset"] != (prev_key or (None,))[0] else ""
    ng = str(r["npg"]) if key != prev_key else ""
    lines.append(f"{cat} & {ds} & {ng} & {r['metric']} & " + " & ".join(cell(r, c) for c in COLS) + " \\\\")
    prev_key = key
lines.append("\\bottomrule\\end{tabular*}")
open(f"{TAB_DIR}/table1.tex", "w").write(
    "% generated by paper/scripts/make_table1.py -- do not edit by hand\n" + "\n".join(lines) + "\n"
)
pd.set_option("display.width", 250)
show = flat[
    ["dataset", "npg", "metric"]
    + [x for c in COLS for x in (c, c + "_lo", c + "_hi") if x in flat]
    + ["paired_best", "paired_runner", "paired_lo", "paired_hi", "paired_excl0"]
]
print(show.round(3).to_string(index=False))
print("\nMISSING:" if missing else "\nall sources present", *missing, sep="\n  ")
