"""PRESET: two-group differential expression for bulk RNA-seq with a prior-fitted network.

    import preset
    res = preset.analyze(counts_df, cond)     # samples x genes raw counts of all genes; cond 0/1 (1 = case)
    hits = res[preset.calls(res, fdr=0.05)]   # rule "auto": target-decoy FDR below 16 per group, Bayesian FDR from 16
    preset.plot.volcano(res)                  # needs matplotlib
    emb = preset.embed(counts_df, cond)       # genes x d network embedding

One forward pass gives each gene a spike-and-slab posterior over the log2 fold change theta,
    q(theta) = pi0 * delta_0 + (1 - pi0) * t_4(theta; m, s),
reported as p_de = P(|theta| > tau), lfc_mean (posterior mean), lfc_given_de = m, rank, expected_false_in_list
and bayes_fdr.

Scope: two groups, trained on 2-16 samples per group (>= 4 per group recommended), one optional two-level batch
column, any organism. Pass all genes: low-count genes inform the dataset-level offset. Covariates
(io.build_covariates) are experimental and were not evaluated.
"""

import numpy as np
import pandas as pd
import torch

from .weights import load_model, resolve_weights
from .io import build_covariates
from .posterior import summarize, calls
from . import diagnose as _diag
from . import plot

__version__ = "1.0.1"
__all__ = [
    "analyze",
    "run",
    "calls",
    "embed",
    "diagnose",
    "load_model",
    "build_covariates",
    "plot",
    "__version__",
]


def _as_matrix(counts):
    if isinstance(counts, pd.DataFrame):
        return counts.to_numpy(), counts.columns.to_numpy(), counts.index.to_numpy()
    X = np.asarray(counts)
    return X, np.array([f"g{i}" for i in range(X.shape[1])]), np.array([f"s{i}" for i in range(X.shape[0])])


def analyze(counts, cond, context="auto", **run_kw):
    """Default analysis: choose genes, run the model, return one row per input gene.

    counts: samples x genes, raw integer counts of all genes (do not pre-filter). cond: 0/1 per sample (1 = case).
    context: low-count genes the model sees, see io.prepare. Genes with >= 10 counts in >= 25% of samples are ranked
    and called ("expressed"); from 8 samples on, the highest-count other genes, up to a quarter of that number, are
    added as context. Rows are in input order; genes left out have NaN posterior columns, and `reported` marks the
    ranked genes. All PRESET results in the paper use this function.
    run_kw are passed to run(). decoy defaults to "auto": the decoy passes run only when the smaller group has fewer
    than 16 samples, where calls() uses them. Pass decoy=True to get decoy_fdr at any size."""
    run_kw.setdefault("decoy", "auto")
    from .io import prepare

    if isinstance(counts, pd.DataFrame):
        names, Xin = counts.columns.to_numpy(), counts.to_numpy()
    else:
        Xin = np.asarray(counts)
        names = np.array([f"g{i}" for i in range(Xin.shape[1])])
    G = Xin.shape[1]
    pos = np.arange(G)
    Xk, kept, info, expressed = prepare(pd.DataFrame(Xin, columns=pos), context=context)
    kept = np.asarray(kept, int)
    r = run(pd.DataFrame(Xk, columns=kept), cond, report=expressed, **run_kw)
    out = pd.DataFrame(index=pos)
    r = r.drop(columns=["gene"])
    r.index = kept
    for c in r.columns:
        out[c] = r[c].reindex(pos)
    out.insert(0, "gene", names)
    out["reported"] = False
    out.loc[kept[np.asarray(expressed, bool)], "reported"] = True
    # kept[i] = input position of prepared gene i
    out.attrs.update(r.attrs)
    out.attrs["prepare"] = info
    out.attrs["kept"] = kept
    return out.reset_index(drop=True)


def run(
    counts,
    cond,
    batch=None,
    covar=None,
    tau=0.5,
    level=0.90,
    weights=None,
    device=None,
    gene_chunk=4096,
    model=None,
    intervals=False,
    slab_temperature=1.0,
    report=None,
    params=False,
    decoy=True,
    decoy_frac=0.25,
    decoy_seed=0,
    decoy_passes=8,
    decoy_plus1=True,
    decoy_variants=False,
    decoy_split="independent",
    decoy_k="floor",
    decoy_depth="raw",
    boundary=False,
    boundary_seed=None,
):
    """Posterior summaries for every gene of the matrix as given (analyze() also chooses the genes).

    counts: samples x genes, raw integer counts. cond: 0/1 per sample (1 = case).
    batch: optional 0/1 label for one two-level nuisance factor (plate, sex, ...).
    covar: optional (n_samples, 7) array from io.build_covariates. Experimental, not evaluated; not a design matrix.
    tau: effect threshold (log2) for p_de.
    report: boolean mask of the genes to rank and call (the `expressed` mask of io.prepare). Other genes still
        condition the posterior but get NaN rank and bayes_fdr. Pass it whenever the matrix holds context genes.
    intervals: add lfc_sd, lfc_median and central credible intervals at `level` (experimental, see
        posterior.summarize). slab_temperature scales the slab width for these columns only.
    params: add pi0, slab_loc, slab_scale, slab_df.
    decoy: True adds decoy_fdr, the target-decoy FDR used by calls(); "auto" runs the decoys only when the smaller
        group has fewer than posterior.AUTO_SWITCH (16) samples; False skips them. Each of decoy_passes extra passes
        (seeds decoy_seed, decoy_seed+1, ...) appends decoy_frac x m decoys, reported genes with samples re-split so
        that a true effect cancels. List FDR = (1 + #decoys >= t) / D x m / (list length); the +1 (knockoff+) avoids
        q = 0 on data with no change. The shortest callable list is 1 / (q x decoy_frac x decoy_passes) genes, 10 at
        q = 0.05. A single pass without the +1 called genes in 40-64% of label-permuted datasets.
    boundary: add boundary_fdr for calls(rule="boundary"): decoy_passes more passes with zero decoys and decoys carrying
        an effect of exactly +-tau, weighted by the posterior count of genes changed by less than tau. Holds at any
        tau (realized FDR <= 0.037 on held-out GTEx tissues, default rule up to 0.104) at lower power.
    decoy_variants, decoy_split, decoy_k, decoy_depth, boundary_seed: evaluation options. decoy_k="nearest" picks
        the re-split that leaves the smallest effect in a decoy when groups differ in size; decoy_depth="match"
        rescales moved counts by the ratio of size factors.
    Validated without batch or covariates. The LFC of genes with mean_count < 50 is strongly shrunk toward 0;
    use the ranking for those genes."""
    X, genes, samples = _as_matrix(counts)
    cond = np.asarray(cond).astype(int)
    n = X.shape[0]
    if X.shape[0] != len(cond):
        raise ValueError("counts must be samples x genes with one condition label per sample")
    if not np.allclose(X, np.round(X)):
        raise ValueError("counts must be raw integer counts")
    b = np.zeros(n, int) if batch is None else np.asarray(batch).astype(int)
    # trained on a single 0/1 batch column
    if batch is not None and not set(np.unique(b)).issubset({0, 1}):
        raise ValueError(
            f"batch must be a single two-level 0/1 column (got levels {sorted(set(np.unique(np.asarray(batch))))}); "
            "for more batches pass the largest batch contrast as 0/1 or correct the batch effect beforehand"
        )
    if model is None:
        model, dev, meta = load_model(weights, device)
    else:
        dev = next(model.parameters()).device
    cv = _check_covar(covar, n, model)
    chunk = gene_chunk if n <= 128 else max(64, int(gene_chunk * (128 / n) ** 2))
    o = _forward(model, dev, X, cond, b, cv, tau, chunk)
    df = summarize(
        o["logits"][0],
        model.head,
        tau=tau,
        level=level,
        intervals=intervals,
        slab_temperature=slab_temperature,
        report=report,
        params=params,
    )
    df.insert(0, "gene", genes)
    df.insert(1, "mean_count", X.mean(0))
    # "auto": only when calls(rule="auto") will use the decoys
    if isinstance(decoy, str):
        if decoy != "auto":
            raise ValueError(f'decoy must be True, False or "auto" (got {decoy!r})')
        from .posterior import AUTO_SWITCH

        cc = np.asarray(cond).astype(int)
        decoy = min(int((cc == 0).sum()), int((cc == 1).sum())) < AUTO_SWITCH
    if decoy:
        rep = np.ones(X.shape[1], bool) if report is None else np.asarray(report, bool)
        src = np.where(rep)[0]
        D = max(1, int(round(decoy_frac * len(src))))
        tgt, dec, pi0s, bs = [], [], [], []
        if decoy_depth not in ("raw", "match"):
            raise ValueError(f"decoy_depth must be 'raw' or 'match' (got {decoy_depth!r})")
        sf = _size_factors(X) if decoy_depth == "match" else None
        # naive decoy log2FC (decoy_variants)
        sf_d = sf if sf is not None else (_size_factors(X) if decoy_variants else None)
        dlfc = []
        for k in range(decoy_passes):
            Xd = np.hstack(
                [
                    X,
                    _decoy_genes(
                        X,
                        cond,
                        src,
                        D,
                        np.random.default_rng(decoy_seed + k),
                        common=decoy_split == "common",
                        k_rule=decoy_k,
                        sf=sf,
                        rng_thin=None if sf is None else np.random.default_rng([decoy_seed + k, 1]),
                    ),
                ]
            )
            if decoy_variants:
                nd = Xd[:, X.shape[1] :] / sf_d[:, None]
                dlfc.append(
                    float(
                        np.median(np.log2(nd[cond == 1].mean(0) + 0.5) - np.log2(nd[cond == 0].mean(0) + 0.5))
                    )
                )
            sm = summarize(
                _forward(model, dev, Xd, cond, b, cv, tau, chunk)["logits"][0],
                model.head,
                tau=tau,
                params=decoy_variants,
            )
            pd_ = sm.p_de.to_numpy()
            tgt.append(pd_[: X.shape[1]][src])
            dec.append(pd_[X.shape[1] :])
            # per-pass P(theta = 0) and P(0 < |theta| < tau)
            if decoy_variants and "pi0" in sm:
                # s + b + pi0 = 1 per pass
                pi0_ = sm.pi0.to_numpy()[: X.shape[1]][src]
                pi0s.append(pi0_)
                bs.append(np.clip(1.0 - pi0_ - tgt[-1], 0.0, 1.0))
        t_tgt, t_dec = np.mean(tgt, 0), np.concatenate(dec)
        if decoy_variants:
            df.attrs.update(
                {
                    "decoy_scores": t_dec,
                    "target_scores_mean": t_tgt,
                    "target_scores_pass0": tgt[0],
                    "decoy_targets": src,
                    "target_scores_passes": np.array(tgt),
                    "decoy_scores_passes": np.array(dec),
                    "decoy_lfc_median": float(np.mean(dlfc)),
                    "log2_sf_ratio": float(np.log2(sf_d[cond == 1]).mean() - np.log2(sf_d[cond == 0]).mean()),
                }
            )
        for col, plus in (("decoy_fdr", int(decoy_plus1)),) + (
            (("decoy_fdr_plus0", 0), ("decoy_fdr_plus1", 1)) if decoy_variants else ()
        ):
            q = np.full(X.shape[1], np.nan)
            q[src] = _decoy_qvalues(t_tgt, t_dec, plus=plus)
            df[col] = q
        # hybrid: decoys count false calls with theta = 0, the posterior those with 0 < |theta| < tau
        if decoy_variants and bs:
            pi0_m, b_m = np.mean(pi0s, 0), np.mean(bs, 0)
            df.attrs.update({"pi0_mean": pi0_m, "b_mean": b_m})
            q = np.full(X.shape[1], np.nan)
            q[src] = _hybrid_qvalues(t_tgt, t_dec, b_m, plus=int(decoy_plus1))
            df["decoy_fdr_hybrid"] = q
        df.attrs.update(
            {
                "decoy_frac": decoy_frac,
                "decoy_seed": decoy_seed,
                "n_decoys": D * decoy_passes,
                "decoy_passes": decoy_passes,
                "decoy_plus1": bool(decoy_plus1),
                "decoy_split": decoy_split,
                "decoy_k": decoy_k,
                "decoy_depth": decoy_depth,
            }
        )
    # separate passes; default columns unchanged
    if boundary:
        rep = np.ones(X.shape[1], bool) if report is None else np.asarray(report, bool)
        src = np.where(rep)[0]
        m = len(src)
        Gx = X.shape[1]
        sp = summarize(o["logits"][0], model.head, tau=tau, params=True)
        if "pi0" not in sp:
            raise ValueError("boundary=True needs a mixture-head checkpoint (pi0 of the posterior)")
        # posterior count of genes with 0 < |theta| < tau
        bg = np.clip(1.0 - sp.pi0.to_numpy()[src] - sp.p_de.to_numpy()[src], 0.0, 1.0)
        Nb = float(bg.sum())
        D0 = max(1, int(round(0.125 * m)))
        Db = max(1, int(round(0.125 * m)))
        bseed = decoy_seed if boundary_seed is None else boundary_seed
        tgt, dz, db = [], [], []
        for k in range(decoy_passes):
            r = np.random.default_rng([bseed, k])
            Z = np.asarray(_decoy_genes(X, cond, src, D0, r)).astype(np.int64)
            # least-favorable boundary: |theta_d| = tau
            B = _thin_effect(_decoy_genes(X, cond, src, Db, r), cond, tau * r.choice([-1.0, 1.0], Db), r)
            p = summarize(
                _forward(model, dev, np.hstack([X, Z, B]), cond, b, cv, tau, chunk)["logits"][0],
                model.head,
                tau=tau,
            ).p_de.to_numpy()
            tgt.append(p[:Gx][src])
            dz.append(p[Gx : Gx + D0])
            db.append(p[Gx + D0 :])
        q = np.full(Gx, np.nan)
        q[src] = _boundary_qvalues(np.mean(tgt, 0), np.concatenate(dz), np.concatenate(db), Nb, plus=1)
        df["boundary_fdr"] = q
        df.attrs.update(
            {"boundary": True, "boundary_Nb": Nb, "boundary_passes": decoy_passes, "boundary_seed": bseed}
        )
    df.attrs.update(
        {
            "tau": tau,
            "level": level,
            "n_control": int((cond == 0).sum()),
            "n_case": int((cond == 1).sum()),
            "head": model.head.kind,
            "slab_temperature": slab_temperature,
        }
    )
    if cv is not None:
        df.attrs["covar"] = {
            "has_covar": bool(cv[0, 1] == 1),
            "is_paired": bool(cv[0, 6] == 1),
            "continuous_z_range": [float(cv[:, 0].min()), float(cv[:, 0].max())],
            "experimental": True,
        }
    return df


def _forward(model, dev, X, cond, b, cv, tau, chunk):
    with torch.no_grad():
        return model.predict(
            torch.tensor(X, dtype=torch.float32, device=dev)[None],
            torch.tensor(cond, dtype=torch.float32, device=dev)[None],
            torch.tensor(b, dtype=torch.float32, device=dev)[None],
            tau=tau,
            gene_chunk=chunk,
            covar=None if cv is None else torch.tensor(cv, dtype=torch.float32, device=dev)[None],
        )


def _size_factors(X):
    """Median-of-ratios size factors of the rows of X, computed as in model.featurize."""
    with torch.no_grad():
        c = torch.tensor(np.asarray(X, dtype=np.float32))
        logg = torch.log(c + 0.5)
        ref = logg.mean(0, keepdim=True)
        ok = (c > 0).all(0, keepdim=True)
        ls = torch.nanmedian((logg - ref).masked_fill(~ok, float("nan")), dim=1).values
        ls = torch.nan_to_num(ls, nan=0.0)
        ls = ls - ls.mean()
    return np.exp(ls.numpy().astype(np.float64))


def _decoy_genes(X, cond, src, D, rng, common=False, k_rule="floor", sf=None, rng_thin=None):
    """(n_samples, D) decoys: genes drawn from `src` with samples re-split so the control positions get
    k = floor(n0 n1 / (n0 + n1)) original cases and both conditions hold the same share of each group.
    With sf, a count moved from sample j to position i is thinned or scaled by sf[i] / sf[j] (decoy_depth="match").
    """
    g0, g1 = np.where(cond == 0)[0], np.where(cond == 1)[0]
    n0, n1 = len(g0), len(g1)
    k = (n0 * n1) // (n0 + n1)
    # minimize the effect left in a decoy (ties to floor)
    if k_rule == "nearest":
        left = lambda kk: abs(kk * (1 / n0 + 1 / n1) - 1)
        if k + 1 <= min(n0, n1) and left(k + 1) < left(k) - 1e-12:
            k = k + 1
    out = np.empty((X.shape[0], D), dtype=X.dtype)
    srow = np.empty((X.shape[0], D), dtype=np.int64)
    # one re-split shared by every decoy of this pass
    if common:
        a0, c0 = rng.permutation(g0), rng.permutation(g1)
    for j, g in enumerate(rng.choice(src, D, replace=True)):
        a, c = (a0, c0) if common else (rng.permutation(g0), rng.permutation(g1))
        srow[g0, j] = np.concatenate([a[: n0 - k], c[:k]])
        srow[g1, j] = np.concatenate([a[n0 - k :], c[k:]])
        out[g0, j] = X[srow[g0, j], g]
        out[g1, j] = X[srow[g1, j], g]
    # depth-matched decoys
    if sf is not None:
        rho = np.asarray(sf, float)[:, None] / np.asarray(sf, float)[srow]
        c = np.round(out).astype(np.int64)
        down = rho <= 1.0
        new = np.empty_like(c)
        new[down] = rng_thin.binomial(c[down], rho[down])
        up = c[~down] * rho[~down]
        fl = np.floor(up)
        new[~down] = (fl + (rng_thin.random(up.shape) < up - fl)).astype(np.int64)
        out = new.astype(X.dtype)
    return out


def _decoy_qvalues(t, t_dec, plus=0):
    """Target-decoy q-values: FDR of the list {t >= t_i} = ((#decoys >= t_i) + plus) / D x m / (list length),
    made monotone over longer lists. plus=1 is the knockoff+ correction."""
    m, D = len(t), len(t_dec)
    o = np.argsort(-t, kind="stable")
    dec = np.sort(t_dec)[::-1]
    fdr = (np.searchsorted(-dec, -t[o], side="right") + plus) / D * m / np.arange(1, m + 1)
    q = np.empty(m)
    q[o] = np.minimum(np.minimum.accumulate(fdr[::-1])[::-1], 1.0)
    return q


def _thin_effect(dec, cond, theta, rng):
    """Plant log2 effects theta (D,) by thinning the control positions (up) or the case positions (down)."""
    out = np.asarray(dec).astype(np.int64).copy()
    c0 = np.asarray(cond) == 0
    c1 = ~c0
    up = theta > 0
    dn = theta < 0
    if up.any():
        out[np.ix_(c0, up)] = rng.binomial(out[np.ix_(c0, up)], 2.0 ** -theta[up][None, :])
    if dn.any():
        out[np.ix_(c1, dn)] = rng.binomial(out[np.ix_(c1, dn)], 2.0 ** theta[dn][None, :])
    return out


def _boundary_qvalues(t, t_zero, t_bnd, Nb, plus=1):
    """Boundary-decoy q-values: FDR of the list {t >= t_i} =
    [(m - Nb)(#zero decoys >= t_i + plus)/D_0 + Nb (#boundary decoys >= t_i)/D_b] / (list length), monotone over
    longer lists. Zero decoys stand for unchanged genes, boundary decoys (|theta| = tau) for the Nb genes that the
    posterior counts as changed by less than tau."""
    m = len(t)
    o = np.argsort(-t, kind="stable")
    ts = t[o]
    z = np.sort(t_zero)[::-1]
    v = np.sort(t_bnd)[::-1]
    V = (m - Nb) * (np.searchsorted(-z, -ts, side="right") + plus) / len(t_zero) + Nb * np.searchsorted(
        -v, -ts, side="right"
    ) / len(t_bnd)
    fdr = V / np.arange(1, m + 1)
    q = np.empty(m)
    q[o] = np.minimum(np.minimum.accumulate(fdr[::-1])[::-1], 1.0)
    return q


def _hybrid_qvalues(t, t_dec, b, plus=1):
    """Hybrid q-values (evaluation only): [((#decoys >= t_i) + plus) m / D + sum of b over the list] / (list length),
    with b_g = P(0 < |theta_g| < tau | data). Decoys cover false calls with theta = 0, b those with a change below tau.
    """
    m, D = len(t), len(t_dec)
    o = np.argsort(-t, kind="stable")
    dec = np.sort(t_dec)[::-1]
    v0 = (np.searchsorted(-dec, -t[o], side="right") + plus) / D * m
    vb = np.cumsum(b[o])
    fdr = (v0 + vb) / np.arange(1, m + 1)
    q = np.empty(m)
    q[o] = np.minimum(np.minimum.accumulate(fdr[::-1])[::-1], 1.0)
    return q


def embed(
    counts, cond, batch=None, covar=None, weights=None, device=None, gene_chunk=4096, model=None, stage="post"
):
    """Per-gene network embedding (d = 128 for preset_v1.0) as a genes x d DataFrame in matrix column order.

    stage="post" (default): after the latent exchange, the input of the posterior head. stage="pre": the gene's
    own encoding before the exchange. Pass the same matrix as to run(), context genes included, and select rows
    afterwards (e.g. emb[expressed]). Descriptive only: nearby genes are treated alike by the network, which has
    not been validated as a measure of co-regulation."""
    X, genes, _ = _as_matrix(counts)
    cond = np.asarray(cond).astype(int)
    n = X.shape[0]
    if X.shape[0] != len(cond):
        raise ValueError("counts must be samples x genes with one condition label per sample")
    if not np.allclose(X, np.round(X)):
        raise ValueError("counts must be raw integer counts")
    if stage not in ("post", "pre"):
        raise ValueError('stage must be "post" or "pre"')
    b = np.zeros(n, int) if batch is None else np.asarray(batch).astype(int)
    if batch is not None and not set(np.unique(b)).issubset({0, 1}):
        raise ValueError("batch must be a single two-level 0/1 column")
    if model is None:
        model, dev, _ = load_model(weights, device)
    else:
        dev = next(model.parameters()).device
    cv = _check_covar(covar, n, model)
    chunk = gene_chunk if n <= 128 else max(64, int(gene_chunk * (128 / n) ** 2))
    with torch.no_grad():
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)[None]
        h, _ = model._embed(f(np.round(X)), f(cond), f(b), chunk, covar=None if cv is None else f(cv))
        if stage == "post":
            h, _ = model.exch(h)
    return pd.DataFrame(
        h[0].float().cpu().numpy(),
        index=pd.Index(genes, name="gene"),
        columns=[f"e{i}" for i in range(h.shape[-1])],
    )


def diagnose(counts, cond, batch=None, covar=None, tau=0.5, weights=None, device=None, model=None):
    """Self-checks on this dataset (see preset.diagnose). covar: same block as run(); the null split reuses its rows
    without re-standardizing them within the subset."""
    X, _, _ = _as_matrix(counts)
    cond = np.asarray(cond).astype(int)
    b = np.zeros(len(cond), int) if batch is None else np.asarray(batch).astype(int)
    if batch is not None and not set(np.unique(b)).issubset({0, 1}):
        raise ValueError("batch must be a single two-level 0/1 column")
    if model is None:
        model, dev, _ = load_model(weights, device)
    else:
        dev = next(model.parameters()).device
    cv = _check_covar(covar, len(cond), model)
    return _diag.report(model, dev, np.round(X).astype(np.int64), cond, b, tau=tau, covar=cv)


def _check_covar(covar, n, model):
    """Validate the optional covariate block. Returns a (n, 7) float32 array or None."""
    if covar is None:
        return None
    cfg = getattr(model, "cfg", None)
    if cfg is not None and not getattr(cfg, "use_covar", False):
        raise ValueError(
            "these weights were trained without the covariate block (ModelConfig.use_covar=False); covar= cannot be used with them"
        )
    dim = getattr(cfg, "covar_dim", 7) if cfg is not None else 7
    cv = np.asarray(covar, dtype=np.float32)
    if cv.ndim != 2 or cv.shape != (n, dim):
        raise ValueError(
            f"covar must be ({n}, {dim}); got {cv.shape}. Build it with preset.io.build_covariates(meta, samples, continuous=..., donor=...)"
        )
    if not np.isfinite(cv).all():
        raise ValueError("covar contains non-finite values")
    return cv
