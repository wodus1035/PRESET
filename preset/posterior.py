"""Summaries of the per-gene posterior over log2 fold change.

Bar head: `out` = logits over 200 bins on [-8, 8]. Mixture heads (mix_t, mix_updown): `out` = parameters of a spike
at 0 with probability pi0 plus student-t slab(s). Both give the same columns. tau, the credible level and the FDR
target are post-processing and need no new forward pass.
"""

import numpy as np
import pandas as pd
import torch

from .model import student_t_cdf


def _cdf_quantiles(probs, edges, qs):
    """Quantiles of a piecewise-uniform distribution over bins. probs (G,K), edges (K+1,). Returns (G, len(qs))."""
    # (G, K+1) at edges
    cdf = np.cumsum(probs, axis=1)
    cdf = np.concatenate([np.zeros((len(probs), 1)), cdf], axis=1)
    out = np.empty((len(probs), len(qs)))
    for j, q in enumerate(qs):
        # bin index containing q (cdf has a leading 0, hence -1)
        k = np.clip((cdf < q).sum(1) - 1, 0, len(edges) - 2)
        lo, hi = cdf[np.arange(len(probs)), k], cdf[np.arange(len(probs)), k + 1]
        frac = np.clip(np.where(hi - lo > 1e-9, (q - lo) / np.maximum(hi - lo, 1e-9), 0.5), 0.0, 1.0)
        out[:, j] = edges[k] + frac * (edges[k + 1] - edges[k])
    return out


def _mixture_cdf(x, pi0, w, loc, scale, df):
    """CDF of spike(0) + student-t mixture at x (G,): pi0 * 1[x >= 0] + (1 - pi0) * sum_c w_c F_c(x). Right-continuous at the spike."""
    return pi0 * (x >= 0) + (1 - pi0) * (w * student_t_cdf((x[:, None] - loc) / scale, df)).sum(1)


def _mixture_quantiles(pi0, w, loc, scale, df, qs, iters=50):
    """Generalized inverse of the mixture CDF (smallest x with F(x) >= q) by vectorized bisection. Returns (G, len(qs)).
    At the spike the CDF jumps, so quantiles that fall into the jump are exactly 0 (pi0 = 1 gives [0, 0, 0]).
    """
    # t(2+) quantile at 1e-6 is ~50 scales from loc
    G = len(pi0)
    span = np.abs(loc).max(1) + 60.0 * scale.max(1) + 1.0
    out = np.empty((G, len(qs)))
    for j, q in enumerate(qs):
        lo, hi = -span.copy(), span.copy()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            ge = _mixture_cdf(mid, pi0, w, loc, scale, df) >= q
            hi = np.where(ge, mid, hi)
            lo = np.where(ge, lo, mid)
        x = hi
        x[np.abs(x) < 1e-7] = 0.0
        out[:, j] = x
    return out


def _summarize_bar(logits, head, tau, level, intervals):
    with torch.no_grad():
        probs = torch.softmax(logits, -1)
        centers = head.centers.to(probs.device)
        edges = head.edges.to(probs.device)
        mean = (probs * centers).sum(-1)
        var = (probs * (centers - mean[:, None]) ** 2).sum(-1)
        p_de = (probs * (centers.abs() > tau).float()).sum(-1)
        p_up = (probs * (centers > tau).float()).sum(-1)
        p_down = (probs * (centers < -tau).float()).sum(-1)
        p_pos = (probs * (centers > 0).float()).sum(-1)
    a = (1 - level) / 2
    qs = _cdf_quantiles(probs.cpu().numpy(), edges.cpu().numpy(), [a, 0.5, 1 - a]) if intervals else None
    df = pd.DataFrame(
        {
            "lfc_mean": mean.cpu().numpy(),
            "p_de": p_de.cpu().numpy(),
            "p_up": p_up.cpu().numpy(),
            "p_down": p_down.cpu().numpy(),
            "p_positive": p_pos.cpu().numpy(),
        }
    )
    if intervals:
        df["lfc_sd"] = var.sqrt().cpu().numpy()
        df["lfc_median"] = qs[:, 1]
        df[f"lfc_lo{int(level*100)}"] = qs[:, 0]
        df[f"lfc_hi{int(level*100)}"] = qs[:, 2]
    return df


def _summarize_mixture(out, head, tau, level, intervals, slab_temperature):
    """Probabilities and mean from the mixture; sd, median and interval use slab scales * slab_temperature."""
    with torch.no_grad():
        mx = head.mixture(out.detach().float())
    pi0, w, loc, sc, df = (mx[k].cpu().numpy().astype(np.float64) for k in ("pi0", "w", "loc", "scale", "df"))
    F = lambda x, s: student_t_cdf((x - loc) / s, df)  # (G,C)
    slab = 1 - pi0
    p_up = slab * (w * (1 - F(tau, sc))).sum(1)
    p_down = slab * (w * F(-tau, sc)).sum(1)
    p_pos = slab * (w * (1 - F(0.0, sc))).sum(1)
    slab_mean = (w * loc).sum(1)
    mean = slab * slab_mean
    # lfc_given_de = E[LFC | effect] (slab mean). lfc_mean = (1 - pi0) * lfc_given_de is the MSE-optimal point estimate
    # and is pulled toward 0 when the model is unsure that the gene changes.
    res = pd.DataFrame(
        {
            "lfc_mean": mean,
            "lfc_given_de": slab_mean,
            "p_de": p_up + p_down,
            "p_up": p_up,
            "p_down": p_down,
            "p_positive": p_pos,
        }
    )
    if intervals:
        scT = sc * float(slab_temperature)
        # E[t^2] for a standard t
        m2 = np.where(df > 2, df / np.maximum(df - 2, 1e-12), np.inf)
        ey2 = slab * (w * (loc**2 + scT**2 * m2)).sum(1)
        a = (1 - level) / 2
        qs = _mixture_quantiles(pi0, w, loc, scT, df, [a, 0.5, 1 - a])
        res["lfc_sd"] = np.sqrt(np.maximum(ey2 - mean**2, 0.0))
        res["lfc_median"] = qs[:, 1]
        res[f"lfc_lo{int(level*100)}"] = qs[:, 0]
        res[f"lfc_hi{int(level*100)}"] = qs[:, 2]
    return res


def summarize(
    out, head, tau=0.5, level=0.90, intervals=False, slab_temperature=1.0, report=None, params=False
):
    """Posterior summaries per gene from the raw head output `out` (G, P).

    Columns: lfc_mean, p_de = P(|LFC| > tau), p_up = P(LFC > tau), p_down = P(LFC < -tau), p_positive = P(LFC > 0),
    rank, expected_false_in_list, bayes_fdr; mixture heads add lfc_given_de. intervals=True adds lfc_sd, lfc_median
    and lfc_lo{level}/lfc_hi{level}; these are experimental and too narrow for genes with a true effect.
    report: boolean mask of the genes to rank (io.prepare's `expressed`). Other genes get NaN rank and bayes_fdr and
    are never called; unmasked low-count context genes can otherwise fill the top of the ranking. None ranks all rows.
    slab_temperature: scales the slab for the interval columns only; p_de and lfc_mean do not change.
    params: add pi0 and, for single-slab heads, slab_loc, slab_scale, slab_df."""
    kind = getattr(head, "kind", "bar")
    if kind == "bar":
        if slab_temperature != 1.0:
            raise ValueError(
                "slab_temperature applies to mixture heads only; for the bar head divide the logits instead"
            )
        df = _summarize_bar(out, head, tau, level, intervals)
    else:
        df = _summarize_mixture(out, head, tau, level, intervals, slab_temperature)
        if params:
            with torch.no_grad():
                mx = head.mixture(out.detach().float())
            df["pi0"] = mx["pi0"].cpu().numpy()
            if mx["loc"].shape[-1] == 1:
                df["slab_loc"] = mx["loc"][..., 0].cpu().numpy()
                df["slab_scale"] = mx["scale"][..., 0].cpu().numpy()
                df["slab_df"] = mx["df"][..., 0].cpu().numpy()
    p = df.p_de.to_numpy()
    rep = np.ones(len(df), bool) if report is None else np.asarray(report, bool)
    if rep.shape != (len(df),):
        raise ValueError(f"report must be a boolean mask of length {len(df)}; got {rep.shape}")
    if not rep.any():
        raise ValueError("report excludes every gene")
    idx = np.where(rep)[0][np.argsort(-p[rep], kind="stable")]  # reported genes, best first
    rank = np.full(len(df), np.nan)
    rank[idx] = np.arange(1, len(idx) + 1)
    efd = np.full(len(df), np.nan)
    efd[idx] = np.cumsum(1 - p[idx])
    df["rank"] = rank
    # expected false discoveries in the list down to this gene
    df["expected_false_in_list"] = efd
    df["bayes_fdr"] = df.expected_false_in_list / df["rank"]  # expected FDR of that list
    return df


AUTO_SWITCH = 16  # smaller-group size from which rule="auto" uses the Bayesian FDR (fixed before validation)


def calls(df, fdr=0.05, min_p=None, rule=None):
    """Boolean vector of called genes at FDR level `fdr`.

    rule=None: "auto" when df.attrs holds the group sizes stored by run(), else "decoy" if decoy_fdr is present,
        else "bfdr".
    "auto": "decoy" when the smaller group has fewer than AUTO_SWITCH (16) samples, "bfdr" from 16.
    "decoy": decoy_fdr <= fdr (target-decoy FDR, needs run(..., decoy=True)).
    "bfdr": the longest top-ranked list whose bayes_fdr is <= fdr.
    "boundary": boundary_fdr <= fdr (run(..., boundary=True)); conservative, holds at any tau.
    "p": p_de > min_p (default 0.9).

    Why the switch at 16: decoys mimic only exactly unchanged genes, while in large samples most false calls are
    genes changed by less than tau. With effects planted into GTEx backgrounds the decoy rule's realized FDR was
    3.2% at 4-12 per group but rose to 0.07-0.13 at 50 per group; on held-out tissues the auto rule gave 2.7-6.7%
    from 16 to 50 per group (decoy 4.5-14.0%). The decoy rule cannot call lists shorter than
    1 / (fdr x decoy_frac x decoy_passes) genes (10 at the defaults), and single datasets can exceed the target,
    e.g. with very sparse effects at 4 per group or null data with outlying counts."""
    if rule is None:
        rule = (
            "auto"
            if ("n_control" in df.attrs and "n_case" in df.attrs)
            else ("decoy" if "decoy_fdr" in df else "bfdr")
        )
    if rule == "auto":
        if "n_control" not in df.attrs or "n_case" not in df.attrs:
            raise ValueError(
                'rule="auto" needs the group sizes stored by preset.run (df.attrs n_control, n_case)'
            )
        n_min = min(int(df.attrs["n_control"]), int(df.attrs["n_case"]))
        rule = "bfdr" if n_min >= AUTO_SWITCH or "decoy_fdr" not in df else "decoy"
    if rule == "decoy":
        if "decoy_fdr" not in df:
            raise ValueError('rule="decoy" needs the decoy_fdr column: run preset.run(..., decoy=True)')
        return (df.decoy_fdr <= fdr).fillna(False).to_numpy(bool)
    # evaluation only (run(..., decoy_variants=True))
    if rule in ("decoy_plus0", "decoy_plus1"):
        col = "decoy_fdr_" + rule.split("_")[1]
        if col not in df:
            raise ValueError(f'rule="{rule}" needs preset.run(..., decoy_variants=True)')
        return (df[col] <= fdr).fillna(False).to_numpy(bool)
    # evaluation only
    if rule == "hybrid":
        if "decoy_fdr_hybrid" not in df:
            raise ValueError(
                'rule="hybrid" needs the decoy_fdr_hybrid column: run preset.run(..., decoy_variants=True)'
            )
        return (df.decoy_fdr_hybrid <= fdr).fillna(False).to_numpy(bool)
    if rule == "boundary":
        if "boundary_fdr" not in df:
            raise ValueError(
                'rule="boundary" needs the boundary_fdr column: run preset.run(..., boundary=True)'
            )
        return (df.boundary_fdr <= fdr).fillna(False).to_numpy(bool)
    if rule == "p":
        return (df.p_de > (0.9 if min_p is None else min_p)).to_numpy()
    ok = df.bayes_fdr <= fdr
    if not ok.any():
        return np.zeros(len(df), bool)
    k = int(df["rank"][ok].max())
    return (df["rank"] <= k).to_numpy()
