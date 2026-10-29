"""Reading counts and metadata, and preparing the (samples x genes) integer matrix the model expects."""

import os
import numpy as np
import pandas as pd


def read_counts(path, genes_in_rows=None):
    """Read a TSV/CSV/parquet/h5ad count table as samples x genes. Unless told, genes are taken to be rows when
    there are more rows than columns."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".h5ad",):
        import anndata

        ad = anndata.read_h5ad(path)
        X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
        return pd.DataFrame(X, index=ad.obs_names, columns=ad.var_names), ad.obs
    else:
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0)
    if genes_in_rows is None:
        genes_in_rows = df.shape[0] > df.shape[1]
    if genes_in_rows:
        df = df.T
    return df, None  # samples x genes


def read_metadata(path):
    sep = "," if path.lower().endswith(".csv") else "\t"
    return pd.read_csv(path, sep=sep, index_col=0)


def design_from_metadata(meta, samples, condition, control=None, case=None, batch=None):
    """Returns (cond 0/1, batch 0/1, keep mask, labels dict) for a two-group design."""
    m = meta.loc[samples]
    col = m[condition].astype(str)
    levels = sorted(col.unique())
    if control is None or case is None:
        if len(levels) != 2:
            raise ValueError(f"condition '{condition}' has levels {levels}; pass --control and --case")
        control, case = levels[0], levels[1]
    keep = col.isin([control, case]).to_numpy()
    cond = (col.to_numpy() == case).astype(int)
    b = np.zeros(len(samples), int)
    if batch is not None:
        bl = m[batch].astype(str)
        lv = sorted(bl.unique())
        if len(lv) > 2:
            raise ValueError("v0.1 supports a single two-level batch column")
        b = (bl.to_numpy() == lv[-1]).astype(int) if len(lv) == 2 else b
    return cond, b, keep, {"control": control, "case": case}


def prepare(counts, min_count=1, min_frac=0.0, context="auto", low_frac=0.25):
    """Check for integer counts and choose the genes the model sees.

    Genes with >= 10 counts in >= 25% of samples are `expressed` and are the only ones ranked. Low-count genes help
    the model place the dataset-level null, but with few samples and many near-zero genes they inflate false calls.
    context="auto": expressed genes only below 8 samples, else also the highest-count low-count genes, up to
    low_frac of the number expressed. "expressed", "cap" and "all" force a rule.
    Returns (int64 matrix, gene names, report dict, expressed mask over the kept genes)."""
    X = counts.to_numpy()
    if not np.allclose(X, np.round(X)):
        raise ValueError("counts must be integers (raw counts, not TPM/FPKM/normalized values)")
    X = np.round(X).astype(np.int64)
    n = X.shape[0]
    any_count = (X >= min_count).sum(0) >= max(1, int(np.ceil(min_frac * n)))
    expressed = (X >= 10).sum(0) >= max(1, int(np.ceil(0.25 * n)))
    rule = context if context != "auto" else ("expressed" if n < 8 else "cap")
    if rule == "expressed":
        keep = expressed.copy()
    elif rule == "all":
        keep = any_count | expressed
    else:
        tot = X.sum(0)
        low = np.where(any_count & ~expressed)[0]
        k = min(len(low), int(low_frac * expressed.sum()))
        keep = expressed.copy()
        keep[low[np.argsort(-tot[low])[:k]]] = True
    rep = {
        "n_samples": int(n),
        "n_genes_in": int(X.shape[1]),
        "n_genes_kept": int(keep.sum()),
        "n_genes_expressed": int(expressed.sum()),
        "context_rule": rule,
        "median_library": int(np.median(X.sum(1))),
    }
    return X[:, keep], counts.columns[keep].to_numpy(), rep, expressed[keep]


# observed covariates (experimental)
COVAR_DIM = 7
COVAR_CODE_DIM = 4
COVAR_COLUMNS = (
    "continuous_z",
    "has_covar",
    "donor_code0",
    "donor_code1",
    "donor_code2",
    "donor_code3",
    "is_paired",
)


def build_covariates(meta, samples, continuous=None, donor=None, code_seed=0):
    """Build the (n_samples, 7) covariate block for run(covar=...). EXPERIMENTAL: not evaluated.

    Columns:
        0      continuous covariate standardized within this dataset (mean 0, sd 1 with ddof=1). The model was
               trained on a standardized covariate, so never pass raw values such as age in years.
        1      has_covar: 1 when a continuous covariate is given.
        2..5   donor code: 4 random numbers per donor (fixed seed); only "same donor = same code" is meaningful.
        6      is_paired: 1 when a donor column is given.

    meta: DataFrame indexed by sample id. samples: sample ids in the row order of the count matrix.
    continuous, donor: a meta column name, an array in sample order, or None. code_seed: seed of the donor codes.
    Returns float32 zeros when both are None, the same as passing no covariates.
    One continuous covariate and one donor grouping only; this is not a design matrix (sex goes in `batch`, and a
    second continuous covariate needs a retrained model).
    """
    n = len(samples)
    cov = np.zeros((n, COVAR_DIM), dtype=np.float32)

    if continuous is not None:
        x = _covar_column(meta, samples, continuous, "continuous")
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(
                "continuous covariate has missing/non-finite values; drop or impute those samples first"
            )
        sd = float(x.std(ddof=1)) if n > 1 else 0.0
        cov[:, 0] = 0.0 if sd <= 0 else (x - x.mean()) / sd  # constant covariate: left at 0
        cov[:, 1] = 1.0

    if donor is not None:
        d = _covar_column(meta, samples, donor, "donor")
        labels = pd.Index(np.asarray(d).astype(str))
        if labels.isna().any():
            raise ValueError("donor column has missing values")
        levels = sorted(labels.unique())
        codes = np.random.default_rng(code_seed).standard_normal((len(levels), COVAR_CODE_DIM))
        idx = np.array([levels.index(v) for v in labels])
        cov[:, 2 : 2 + COVAR_CODE_DIM] = codes[idx]
        cov[:, 6] = 1.0

    return cov


def _covar_column(meta, samples, spec, what):
    """A meta column name, or an array already in sample order."""
    if isinstance(spec, str):
        if meta is None:
            raise ValueError(f"{what}='{spec}' is a column name but no metadata was given")
        if spec not in meta.columns:
            raise ValueError(f"{what} column '{spec}' not in metadata (columns: {list(meta.columns)})")
        return meta.loc[samples, spec].to_numpy()
    v = np.asarray(spec)
    if len(v) != len(samples):
        raise ValueError(f"{what} has {len(v)} values but there are {len(samples)} samples")
    return v
