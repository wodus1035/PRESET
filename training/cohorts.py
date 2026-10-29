"""Loaders of the real cohorts of the second stage (Supplementary Table S3). None of them is used in any evaluation.

Each loader returns a list of (name, X genes x samples, control columns, case columns, batch or None, gene ids).
Raw files go under <PRESET_DATA>/training/ (download commands in training/README.md), except the GTEx counts
(<PRESET_DATA>/gtex/) and the BulkFormer cardiovascular collection (<PRESET_DATA>/bulkformer/cvd_adata.h5ad).
"""

import gzip
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(os.path.dirname(HERE), "evaluation")
for p_ in (EVAL, os.path.join(EVAL, "analysis", "common")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

from heldout_sets import mito_rrna_qc  # noqa: E402  same QC as the held-out cohorts

DATA = os.environ.get("PRESET_DATA", os.path.join(EVAL, "data"))
RAW = os.path.join(DATA, "training")
GTEX_SEX_TISSUES = (
    "breast_mammary_tissue",
    "esophagus_mucosa",
    "kidney_cortex",
    "liver",
    "pancreas",
    "pituitary",
    "spleen",
)


def cvd(min_per_group=40):
    """Cardiovascular disease vs control series of the BulkFormer collection (Zenodo 15744294) that hold raw counts.
    data_cache/cvd_obs_full.tsv maps each sample to its GEO series and group, data_cache/cvd_usable_series.tsv lists
    the series with integer counts."""
    import h5py

    f = h5py.File(os.path.join(DATA, "bulkformer", "cvd_adata.h5ad"), "r")
    g = f["obs"]["GSM_ID"]
    if isinstance(g, h5py.Group):
        ids = g["categories"][:].astype(str)[g["codes"][:]]
    else:
        ids = g[:].astype(str)

    obs = pd.read_csv(os.path.join(HERE, "data_cache", "cvd_obs_full.tsv"), sep="\t")
    assert (obs.gsm.to_numpy() == ids).all(), "cvd_obs_full.tsv is not in the order of the h5ad"
    usable = pd.read_csv(os.path.join(HERE, "data_cache", "cvd_usable_series.tsv"), sep="\t")
    X = f["X"][:].astype(np.int64).T
    gn = f["var"]["ensg_id"][:].astype(str)

    out = []
    for gse in usable[usable.counts_ok].gse1:
        s = obs[obs.gse1 == gse]
        c = s.index[s.bin == "control"].to_numpy()
        k = s.index[s.bin == "case"].to_numpy()
        if min(len(c), len(k)) < min_per_group:
            continue
        idx = np.concatenate([c, k])
        out.append((f"CVD {gse}", X[:, idx], np.arange(len(c)), np.arange(len(c), len(idx)), None, gn))
    return out


def psoriasis():
    """GSE268958, lesional skin at baseline vs healthy skin."""
    d = pd.read_csv(os.path.join(RAW, "GSE268958_raw_counts.tsv.gz"), sep=" ")

    # sample characteristics from the series soft file
    txt = gzip.open(os.path.join(RAW, "GSE268958_family.soft.gz"), "rt", errors="ignore").read()
    meta = []
    for blk in txt.split("^SAMPLE = ")[1:]:
        ch = dict(c.split(":", 1) for c in re.findall(r"!Sample_characteristics_ch1 = (.*)", blk) if ":" in c)
        title = re.search(r"!Sample_title = ([^,]*)", blk).group(1).strip()
        meta.append({"col": title, **{k.strip(): v.strip() for k, v in ch.items()}})

    d, _ = mito_rrna_qc(d, "GSE268958")
    m = pd.DataFrame(meta).set_index("col").reindex(d.columns)
    les = ((m.tissue == "Lesional skin") & (m.timepoint == "BL")).to_numpy()
    hc = (m.condition == "Healthy").to_numpy()
    idx = np.concatenate([np.where(hc)[0], np.where(les)[0]])
    X = d.values[:, idx].astype(np.int64)
    name = "Psoriasis lesional BL vs healthy (GSE268958)"
    return [(name, X, np.arange(hc.sum()), np.arange(hc.sum(), len(idx)), None, d.index.to_numpy())]


def hape():
    """GSE291471, high-altitude pulmonary edema vs sojourners without it."""
    d = pd.read_csv(os.path.join(RAW, "GSE291471_RNASeq_raw_count.csv.gz"), index_col=0).set_index("ENSID")
    d, _ = mito_rrna_qc(d, "GSE291471")
    grp = np.array([re.sub(r"\d+$", "", c) for c in d.columns])
    idx = np.concatenate([np.where(grp == "CON")[0], np.where(grp == "HAPE")[0]])
    nc = int((grp == "CON").sum())
    X = d.values[:, idx].astype(np.int64)
    return [
        (
            "HAPE vs sojourners (GSE291471)",
            X,
            np.arange(nc),
            np.arange(nc, len(idx)),
            None,
            d.index.to_numpy(),
        )
    ]


def ted():
    """GSE285190, thyroid eye disease vs healthy controls in batches 13, 14, 16 and 17. Healthy controls are the columns
    named TED_<batch>_H<k>, and the GH_* columns (Graves disease without eye disease) are not used."""
    d = pd.read_csv(os.path.join(RAW, "GSE285190_TED_HC_GH_ERCC_count_matrix.csv.gz"), index_col=0)
    d, _ = mito_rrna_qc(d, "GSE285190")
    grp = np.array(["HC" if re.search(r"_H\d+$", c) else c.split("_")[0] for c in d.columns])
    bat = np.array([c.split("_")[1] for c in d.columns])

    keep = np.isin(bat, ["13", "14", "16", "17"]) & np.isin(grp, ["HC", "TED"])
    X = d.values[:, keep].astype(np.int64)
    g = grp[keep]
    b = pd.factorize(bat[keep])[0]
    ctrl = np.where(g == "HC")[0]
    case = np.where(g == "TED")[0]
    return [("TED vs HC (GSE285190)", X, ctrl, case, b, d.index.to_numpy())]


def mouse_atlas():
    """GSE314817 counts (genes x samples), sample table with tissue and sex, and gene ids."""
    df = pd.read_csv(os.path.join(RAW, "GSE314817_mouse_raw_counts_wo_muscle.txt.gz"), sep="\t", index_col=0)

    # tissue and sex of each sample from the series soft file
    meta = {}
    cur = {}
    with gzip.open(os.path.join(RAW, "GSE314817_family.soft.gz"), "rt") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                cur = {"title": line.split("=", 1)[1].strip()}
            elif line.startswith("!Sample_characteristics_ch1"):
                k, v = line.split("=", 1)[1].strip().split(": ", 1)
                cur[k.lower()] = v
                if k.lower() == "age":
                    meta[cur["title"].split("_")[-1]] = cur
    m = pd.DataFrame.from_dict(meta, orient="index")
    m = m.loc[[c for c in df.columns if c in m.index]]
    df = df[m.index]
    return df.to_numpy(np.int64), m, df.index.to_numpy().astype(str)


def mouse_sex(min_per_group=16):
    """GSE314817, male vs female mice within each of 24 tissues."""
    X, m, gn = mouse_atlas()
    out = []
    for t in sorted(m.tissue.unique()):
        a = np.where((m.tissue == t) & (m.sex.str.lower() == "male"))[0]
        b = np.where((m.tissue == t) & (m.sex.str.lower() == "female"))[0]
        if min(len(a), len(b)) < min_per_group:
            continue
        idx = np.concatenate([a, b])
        out.append((f"mouse sex {t}", X[:, idx], np.arange(len(a)), np.arange(len(a), len(idx)), None, gn))
    return out


def gtex_sex(min_sex=40):
    """Female vs male donors of the bank half in GTEx tissues of the real-background bank."""
    from gtex_sex_eval import load_tissue_sex

    out = []
    for t in GTEX_SEX_TISSUES:
        sym, X, sex = load_tissue_sex(t, donor_split="bank")
        f_ = np.where(sex == 2)[0]
        m_ = np.where(sex == 1)[0]
        if min(len(f_), len(m_)) < min_sex:
            print(f"gtex_sex {t}: skipped ({len(f_)}F/{len(m_)}M)", flush=True)
            continue
        out.append((f"gtex_sex {t}", X, f_, m_, None, sym))
    return out
