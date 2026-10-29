"""ERCC spike-in truth, every method on the same matrix: SEQC BGI site, A (UHRR + Mix 1) vs B (HBRR + Mix 2), 5 v 5 technical
replicates. Truth = Mix 2 / Mix 1 per subgroup: A = -2, B = 0, C = +0.58, D = +1 log2. The spike-in total differs between mixes,
which shifts every ERCC by one constant; it is estimated per method as the median LFC of subgroup B (1:1) and removed before the MAE.
Cells: full 5 v 5 and --draws random 3 v 3 subsamples. Output: per-ERCC reported LFC and per-cell slopes, MAE and call rates.
    python analysis/seqc/seqc_ercc_fair.py --gpu -1 --jobs 20
"""

import os, sys, argparse, itertools, warnings, numpy as np, pandas as pd, torch

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

DATA = os.environ.get("PRESET_DATA", os.path.join(ROOT, "data"))  # see data/README.md
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None)
ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--min_count", type=float, default=200)
ap.add_argument("--draws", type=int, default=20)
ap.add_argument("--jobs", type=int, default=1)
ap.add_argument("--gpu", type=int, default=-1)
ap.add_argument("--output_dir", default=os.path.join(RESULTS, "seqc_ercc_fair"))
A = ap.parse_args()
os.makedirs(A.output_dir, exist_ok=True)
df = pd.read_csv(os.path.join(DATA, "seqc", "SEQC_BGI_AB_gene_ERCC.tsv"), sep="\t")
truth = pd.read_csv(os.path.join(DATA, "seqc", "ERCC_truth.tsv"), sep="\t").set_index("ERCC_ID")
X = df.iloc[:, 1:].to_numpy(np.int64)
ids = df.TranscriptID.to_numpy()
cols = np.array(df.columns[1:])
isA = np.array([c.startswith("A_") for c in cols])
A_idx, B_idx = np.where(isA)[0], np.where(~isA)[0]
is_ercc = np.array([i.startswith("ERCC") for i in ids])
tl = pd.Series(-truth.true_lfc.reindex(ids[is_ercc]).to_numpy(), index=ids[is_ercc])
sub = truth.subgroup.reindex(ids[is_ercc])
_M = {}


def model():
    if "m" not in _M:
        torch.set_num_threads(1)
        _M["m"] = preset.load_model(A.ckpt, torch.device("cpu"))[0]
    return _M["m"]


def cell(tag, a_cols, b_cols):
    idx = np.concatenate([a_cols, b_cols])
    sub_X = X[:, idx]
    h = len(a_cols)
    cond = np.array([0] * h + [1] * len(b_cols))
    b = np.zeros_like(cond)
    E = np.where(fc.expressed_mask(sub_X))[0]
    S = sub_X[E]
    gE = ids[E]
    eE = is_ercc[E]
    out = {}
    from gtex_realdata_eval import preset_default
    import preset  # package default on the tools' matrix

    r = preset_default(S, cond, model(), tau=A.tau, decoy_seed=0)
    out["PRESET"] = (r.lfc_mean.to_numpy(), r.p_de.to_numpy(), (r.p_de > 0.9).to_numpy())
    # default call rule
    out["PRESET decoy"] = (
        r.lfc_mean.to_numpy(),
        r.p_de.to_numpy(),
        np.asarray(preset.calls(r, fdr=0.05, rule="decoy"), bool),
    )
    e = run_edger(S.T, cond, b, A.tau, filter_expr=True, robust=True)
    ev = np.isfinite(e.logFC.to_numpy(float)) if e is not None else np.ones(len(E), bool)
    if e is not None:
        out["edgeR TREAT"] = (
            e.logFC.to_numpy(float),
            np.nan_to_num(e.FDR.to_numpy(float), nan=1),
            np.nan_to_num(e.FDR.to_numpy(float), nan=1) < 0.05,
        )

    def back(x):
        y = np.full(len(E), np.nan)
        y[ev] = x
        return y

    # DESeq2 fits all genes, scored on the edgeR set
    d2 = run_deseq2_scored(S.T, cond, b, ev, A.tau, cooks=True)
    if d2 is not None:
        out["DESeq2 lfcThr"] = (
            d2.log2FoldChange.to_numpy(float),
            d2.padj.to_numpy(float),
            np.nan_to_num(d2.padj.to_numpy(float), nan=1) < 0.05,
        )
    lv = run_limma_voom(S.T, cond, b, A.tau, filter_expr=True)
    if lv is not None:
        out["limma treat"] = (
            lv.logFC.to_numpy(float),
            lv.FDR.to_numpy(float),
            np.nan_to_num(lv.FDR.to_numpy(float), nan=1) < 0.05,
        )
    _, apg = mle_shrunk_scored(S.T, cond, b, ev, A.tau, 1, cooks=True)
    if apg is not None:
        out["apeglm"] = (apg, np.full(len(E), np.nan), np.zeros(len(E), bool))
    genes, summ = [], []
    mean_cnt = S.mean(1)
    well = eE & (mean_cnt > A.min_count)
    for nm, (lfc, stat, call) in out.items():
        lfc = np.where(ev, lfc, np.nan)
        for i in np.where(eE)[0]:
            genes.append(
                dict(
                    cell=tag,
                    method=nm,
                    ercc=gE[i],
                    subgroup=sub[gE[i]],
                    true=tl[gE[i]],
                    reported=lfc[i],
                    stat=stat[i],
                    call=bool(call[i]),
                    mean_count=mean_cnt[i],
                    well=bool(well[i]),
                )
            )
        t_w = tl[gE[well]].to_numpy()
        r_w = lfc[well]
        ok = np.isfinite(r_w)
        off = np.nanmedian(lfc[well & (sub.reindex(gE).to_numpy() == "B")])
        row = dict(
            cell=tag,
            method=nm,
            n_well=int(ok.sum()),
            offset=off,
            slope_calib=np.polyfit(r_w[ok], t_w[ok], 1)[0] if ok.sum() > 5 else np.nan,
            slope_atten=np.polyfit(t_w[ok], r_w[ok], 1)[0] if ok.sum() > 5 else np.nan,
            corr=np.corrcoef(r_w[ok], t_w[ok])[0, 1] if ok.sum() > 5 else np.nan,
            mae_offset_corrected=np.nanmean(np.abs((r_w[ok] - off) - t_w[ok])),
        )
        # B is 1:1: its call rate is the false-call rate
        for g in ("A", "B", "C", "D"):
            m = well & (sub.reindex(gE).to_numpy() == g)
            row[f"called_{g}"] = float(call[m].mean()) if m.any() else np.nan
            if nm.startswith("PRESET"):
                row[f"pde_{g}"] = float(stat[m].mean()) if m.any() else np.nan
        summ.append(row)
    return genes, summ


tasks = [("5v5", A_idx, B_idx)]
rng = np.random.default_rng(0)
combos = list(itertools.product(itertools.combinations(A_idx, 3), itertools.combinations(B_idx, 3)))
rng.shuffle(combos)
for k, (a3, b3) in enumerate(combos[: A.draws]):
    tasks.append((f"3v3_{k}", np.array(a3), np.array(b3)))
print(f"{len(tasks)} cells over {A.jobs} workers", flush=True)
if A.jobs > 1:
    from joblib import Parallel, delayed

    res = Parallel(n_jobs=A.jobs, verbose=5)(delayed(cell)(*t) for t in tasks)
else:
    res = [cell(*t) for t in tasks]
G = pd.DataFrame([g for gs, _ in res for g in gs])
Sm = pd.DataFrame([s for _, ss in res for s in ss])
G.to_csv(os.path.join(A.output_dir, "ercc_genes.csv"), index=False)
Sm.to_csv(os.path.join(A.output_dir, "ercc_summary.csv"), index=False)
Sm["design"] = Sm.cell.str.split("_").str[0]
pd.set_option("display.width", 250)
print(
    Sm.groupby(["design", "method"])[
        [
            "n_well",
            "offset",
            "slope_calib",
            "slope_atten",
            "corr",
            "mae_offset_corrected",
            "called_A",
            "called_C",
            "called_D",
        ]
    ]
    .mean()
    .round(3)
    .to_string()
)
