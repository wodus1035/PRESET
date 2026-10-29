"""Loaders for the real cohorts other than SEQC and GTEx. Each returns a list of comparisons (name, X, ctrl, case, batch,
genes): X = genes x samples counts of all samples in the comparison, ctrl and case = column indices, batch = per-sample code
or None, genes = one id per row of X. Loaders do not subsample genes.

STATUS and EXPOSURE_NOTE record how each dataset relates to the development of PRESET. The mouse comparisons informed the
recommendation of n >= 4 per group, so they cannot verify it.

    python analysis/common/heldout_sets.py [source ...]     # print the comparisons of each source
"""

import os, gzip
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
DATA = os.environ.get("PRESET_DATA", os.path.join(ROOT, "data"))  # see data/README.md
HELD = os.path.join(DATA, "heldout")

STATUS = {
    "mouse_genotype": "held out",
    "tcga": "held out",
    "ipf": "held out",
    "sepsis": "held out",
    "crohn": "held out",
    "skin2": "held out",
}
EXPOSURE_NOTE = {
    "ipf": "GSE150910, never used",
    "sepsis": "GSE185263, never used",
    "crohn": "GSE57945, never used",
    "skin2": "GSE121212, never used",
    "tcga": "GDC STAR counts KIRC/LUAD/THCA/LIHC, not used for the prior, scenarios, architecture or training",
    "mouse_genotype": "not used for the prior, scenarios, architecture or training; borderline: evidence for the n >= 4 per group recommendation",
}
SOURCES = tuple(STATUS)


def mouse_genotype(min_per_group=12):
    out = []
    for gene in ("Dchs1", "Fat4"):
        df = pd.read_csv(os.path.join(HELD, f"GSE272152_{gene}_counts.txt.gz"), sep="\t", index_col=0)
        cols = df.columns.to_series()
        organ = cols.str.split(".").str[0].to_numpy()
        geno = cols.str.split(".").str[1].to_numpy()
        X = df.to_numpy(np.int64)
        gn = df.index.to_numpy().astype(str)
        for o in sorted(set(organ)):
            wt = np.where((organ == o) & (geno == "WT"))[0]
            het = np.where((organ == o) & (geno == "Het"))[0]
            if min(len(wt), len(het)) < min_per_group:
                continue
            idx = np.concatenate([wt, het])
            out.append(
                (f"mouse {gene} {o}", X[:, idx], np.arange(len(wt)), np.arange(len(wt), len(idx)), None, gn)
            )
    return out


def tcga(max_tumour=150, seed=0):
    """TCGA STAR counts (tcga_download.py): Solid Tissue Normal vs Primary Tumor per project, unpaired; tumors capped at max_tumour."""
    out = []
    rng = np.random.default_rng(seed)
    for proj in ("KIRC", "LUAD", "THCA", "LIHC"):
        fn = os.path.join(DATA, "tcga", f"{proj}_star_counts.tsv.gz")
        if not os.path.exists(fn):
            continue
        df = pd.read_csv(fn, sep="\t")
        meta = pd.read_csv(os.path.join(DATA, "tcga", f"{proj}_samples.tsv"), sep="\t").set_index("sample")
        cols = [c for c in df.columns if c.startswith("TCGA-")]
        st = meta.loc[cols, "sample_type"].to_numpy()
        normal = np.where(st == "Solid Tissue Normal")[0]
        tum = np.where(st == "Primary Tumor")[0]
        if len(tum) > max_tumour:
            tum = np.sort(rng.choice(tum, max_tumour, replace=False))
        idx = np.concatenate([normal, tum])
        X = df[cols].to_numpy(np.int64)[:, idx]
        out.append(
            (
                f"TCGA {proj} tumour vs normal",
                X,
                np.arange(len(normal)),
                np.arange(len(normal), len(idx)),
                None,
                df["gene_id"].to_numpy().astype(str),
            )
        )
    return out


# 12S and 16S mitochondrial rRNA, symbol or Ensembl
MITO_RRNA = ("MT-RNR1", "MT-RNR2", "ENSG00000211459", "ENSG00000210082")
MITO_RRNA_MAX_SHARE = 0.30


def mito_rrna_qc(d, tag=""):
    """Removes the two mitochondrial rRNA genes and drops samples where they exceed 30% of the library (failed rRNA
    depletion). Returns (filtered DataFrame, kept-sample mask)."""
    ids = d.index.astype(str).str.split(".").str[0]
    is_r = np.isin(ids, MITO_RRNA)
    share = d.values[is_r].sum(0) / np.maximum(d.values.sum(0), 1)
    keep = share <= MITO_RRNA_MAX_SHARE
    if (~keep).any() or is_r.any():
        print(
            f"mito rRNA QC{(' ' + tag) if tag else ''}: {int(is_r.sum())} rRNA genes removed, {int((~keep).sum())}/{len(keep)} samples dropped (share > {MITO_RRNA_MAX_SHARE})",
            flush=True,
        )
    return d.loc[~is_r, d.columns[keep]], keep


HELD3 = os.path.join(DATA, "heldout3")  # evaluation cohorts fixed before any result


def _soft_meta(path, key_from_title=None):
    """GSM -> {title, characteristics...} from a family.soft.gz."""
    import gzip, re

    txt = gzip.open(path, "rt", errors="ignore").read()
    out = {}
    for blk in txt.split("^SAMPLE = ")[1:]:
        gsm = blk.split("\n", 1)[0].strip()
        title = re.search(r"!Sample_title = (.*)", blk).group(1).strip()
        ch = {
            k.strip(): v.strip()
            for k, v in (
                c.split(":", 1) for c in re.findall(r"!Sample_characteristics_ch1 = (.*)", blk) if ":" in c
            )
        }
        out[gsm] = dict(title=title, **ch)
    return out


def ipf(min_per_group=12):
    """GSE150910: lung explants/biopsies, IPF (103) and chronic hypersensitivity pneumonitis (82) vs unused donor lungs (103); gene symbols."""
    d = pd.read_csv(os.path.join(HELD3, "GSE150910_gene-level_count_file.csv.gz"), index_col=0)
    d, _ = mito_rrna_qc(d, "GSE150910")
    grp = d.columns.str.split("_").str[0].to_numpy()
    X = d.values.astype(np.int64)
    gn = d.index.to_numpy().astype(str)
    ctrl = np.where(grp == "control")[0]
    out = []
    for g_, nm in (("ipf", "IPF"), ("chp", "CHP")):
        case = np.where(grp == g_)[0]
        if min(len(ctrl), len(case)) >= min_per_group:
            idx = np.concatenate([ctrl, case])
            out.append(
                (
                    f"{nm} vs control lung (GSE150910)",
                    X[:, idx],
                    np.arange(len(ctrl)),
                    np.arange(len(ctrl), len(idx)),
                    None,
                    gn,
                )
            )
    return out


def sepsis(min_per_group=12):
    """GSE185263: whole blood, sepsis at ICU admission (348) vs healthy (44); Ensembl ids."""
    d = pd.read_csv(os.path.join(HELD3, "GSE185263_raw_counts.csv.gz"), index_col=0)
    d, _ = mito_rrna_qc(d, "GSE185263")
    meta = _soft_meta(os.path.join(HELD3, "GSE185263_family.soft.gz"))
    state = {m["title"]: m.get("disease state", "") for m in meta.values()}
    st = np.array([state.get(c, "") for c in d.columns])
    X = d.values.astype(np.int64)
    gn = d.index.to_numpy().astype(str)
    ctrl = np.where(st == "healthy")[0]
    case = np.where(st == "sepsis")[0]
    if min(len(ctrl), len(case)) < min_per_group:
        return []
    idx = np.concatenate([ctrl, case])
    return [
        (
            "sepsis vs healthy blood (GSE185263)",
            X[:, idx],
            np.arange(len(ctrl)),
            np.arange(len(ctrl), len(idx)),
            None,
            gn,
        )
    ]


def crohn(min_per_group=12):
    """GSE57945 (RISK): pediatric ileal biopsies, Crohn's disease (218) and ulcerative colitis (62) vs not-IBD (42); one file per sample."""
    import glob as _g

    meta = _soft_meta(os.path.join(HELD3, "GSE57945_family.soft.gz"))
    cols, labs = [], []
    for fn in sorted(_g.glob(os.path.join(HELD3, "GSE57945", "GSM*_CCFA_Risk_*.txt.gz"))):
        gsm = os.path.basename(fn).split("_")[0]
        df = pd.read_csv(fn, sep="\t", index_col=0)
        cols.append(df.iloc[:, 1].rename(gsm))
        labs.append(meta[gsm].get("diagnosis", "").lower())
    d = pd.concat(cols, axis=1)
    d = d.loc[:, :]
    sym = pd.read_csv(
        sorted(_g.glob(os.path.join(HELD3, "GSE57945", "GSM*_CCFA_Risk_*.txt.gz")))[0], sep="\t", index_col=0
    ).iloc[:, 0]
    d.index = sym.reindex(d.index).fillna(d.index.to_series()).astype(str).to_numpy()
    d, keep = mito_rrna_qc(d, "GSE57945")
    labs = np.asarray(labs)[keep]
    X = d.values.astype(np.int64)
    gn = d.index.to_numpy().astype(str)
    ctrl = np.where(labs == "not ibd")[0]
    out = []
    for g_, nm in (("cd", "Crohn's disease"), ("uc", "ulcerative colitis")):
        case = np.where(labs == g_)[0]
        if min(len(ctrl), len(case)) >= min_per_group:
            idx = np.concatenate([ctrl, case])
            out.append(
                (
                    f"{nm} vs not-IBD ileum (GSE57945)",
                    X[:, idx],
                    np.arange(len(ctrl)),
                    np.arange(len(ctrl), len(idx)),
                    None,
                    gn,
                )
            )
    return out


def skin2(min_per_group=12):
    """GSE121212: skin, psoriasis (28) and atopic dermatitis (21) lesional vs healthy (38); independent of GSE268958 (used for training labels)."""
    d = pd.read_csv(os.path.join(HELD3, "GSE121212_readcount.txt.gz"), sep="\t", index_col=0)
    d = d.loc[:, [c for c in d.columns if c.strip()]]
    d, _ = mito_rrna_qc(d, "GSE121212")
    X = d.values.astype(np.int64)
    gn = d.index.to_numpy().astype(str)
    kind = np.array([c.split("_")[0] + "_" + c.split("_")[-1] for c in d.columns])
    ctrl = np.where(kind == "CTRL_healthy")[0]
    out = []
    for k_, nm in (("PSO_lesional", "psoriasis lesional"), ("AD_lesional", "atopic dermatitis lesional")):
        case = np.where(kind == k_)[0]
        if min(len(ctrl), len(case)) >= min_per_group:
            idx = np.concatenate([ctrl, case])
            out.append(
                (
                    f"{nm} vs healthy skin (GSE121212)",
                    X[:, idx],
                    np.arange(len(ctrl)),
                    np.arange(len(ctrl), len(idx)),
                    None,
                    gn,
                )
            )
    return out


LOADERS = {
    "mouse_genotype": mouse_genotype,
    "tcga": tcga,
    "ipf": ipf,
    "sepsis": sepsis,
    "crohn": crohn,
    "skin2": skin2,
}


def comparisons(which):
    if which not in LOADERS:
        raise ValueError(f"unknown source {which}; have {sorted(LOADERS)}")
    return LOADERS[which]()


if __name__ == "__main__":
    import sys

    for w in sys.argv[1:] or SOURCES:
        try:
            cs = comparisons(w)
            print(f"{w} [{STATUS[w]}] -- {EXPOSURE_NOTE[w]}\n  {len(cs)} comparisons")
            for nm, X, c, k, b, gn in cs:
                print(
                    f"    {nm:34s} genes {X.shape[0]:6d}  ctrl {len(c):4d}  case {len(k):4d}"
                    + (f"  batches {len(np.unique(b))}" if b is not None else "")
                )
        except Exception as e:
            print(f"{w}: FAILED {type(e).__name__}: {e}")
