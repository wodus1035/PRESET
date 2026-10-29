"""Plots for PRESET results (pip install "preset[plot]"; UMAP maps also need umap-learn).

    res = preset.run(X, cond, report=expressed)
    preset.plot.volcano(res)
    emb = preset.embed(X, cond)                 # same X as run
    ax, xy = preset.plot.embedding(emb, res)

Both draw on a new figure unless ax= is given.
"""

import numpy as np
import pandas as pd

UP, DOWN, GREY, INK = "#c0392b", "#2a78d6", "#bdbdbd", "#222222"


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('preset.plot needs matplotlib:  pip install "preset[plot]"') from e
    return plt


def _reported(res):
    """Mask of ranked rows; context genes (rank NaN) are not drawn."""
    return res["rank"].notna().to_numpy() if "rank" in res else np.ones(len(res), bool)


def _calls(res, called, fdr, rule):
    if called is not None:
        c = np.asarray(called, bool)
        if len(c) != len(res):
            raise ValueError(f"called has {len(c)} entries, res has {len(res)} rows")
        return c
    from .posterior import calls

    return calls(res, fdr=fdr, rule=rule)


def _rule_name(res, rule):
    r = rule if rule is not None else ("decoy" if "decoy_fdr" in res else "bfdr")
    return {"decoy": "target-decoy", "bfdr": "Bayesian", "p": "P"}.get(r, r)


def volcano(res, called=None, fdr=0.05, rule=None, label=10, y="p_de", ax=None, s=4, title=True):
    """Volcano plot of a run() or analyze() result; returns the Axes.

    x: lfc_mean, shrunk toward 0 when evidence is weak. y: "p_de" (default), or "decoy_fdr" / "bayes_fdr" on a
    -log10 scale. Genes called by calls(res, fdr, rule), or by the boolean array `called`, are red (up) and blue
    (down), and the `label` called genes with the largest |lfc_mean| are named. Only ranked genes are drawn.
    """
    plt = _plt()
    keep = _reported(res)
    c = _calls(res, called, fdr, rule)[keep]
    r = res[keep]
    x = r["lfc_mean"].to_numpy(float)
    tau = res.attrs.get("tau", 0.5)
    if y == "p_de":
        yy = r["p_de"].to_numpy(float)
        ylab = rf"$P(|\log_2\mathrm{{FC}}| > {tau:g})$"
    elif y in ("decoy_fdr", "bayes_fdr"):
        if y not in r:
            raise ValueError(f"column {y} not in res (decoy_fdr needs preset.run(..., decoy=True))")
        yy = -np.log10(np.clip(r[y].to_numpy(float), 1e-6, 1.0))
        ylab = rf"$-\log_{{10}}$ {y.replace('_', ' ')}"
    else:
        raise ValueError('y must be "p_de", "decoy_fdr" or "bayes_fdr"')
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 3.9))
    up, dn = c & (x > 0), c & (x < 0)
    ax.scatter(x[~c], yy[~c], s=s, color=GREY, alpha=0.45, lw=0, rasterized=True)
    ax.scatter(
        x[up], yy[up], s=s + 2, color=UP, alpha=0.8, lw=0, rasterized=True, label=f"up ({int(up.sum()):,})"
    )
    ax.scatter(
        x[dn],
        yy[dn],
        s=s + 2,
        color=DOWN,
        alpha=0.8,
        lw=0,
        rasterized=True,
        label=f"down ({int(dn.sum()):,})",
    )
    if label and c.any() and "gene" in r:
        _label_extremes(ax, x, yy, c, r["gene"].astype(str).to_numpy(), label)
    ax.axvline(0, color=GREY, lw=0.6, zorder=0)
    ax.set_xlabel(r"posterior mean $\log_2$ fold change")
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(
            (
                title
                if isinstance(title, str)
                else f"{int(c.sum()):,} genes called ({_rule_name(res, rule)} FDR {fdr:g})"
            ),
            fontsize=9,
        )
    if c.any():
        ax.legend(frameon=False, fontsize=8, loc="best", markerscale=2)
    return ax


def _label_extremes(ax, x, y, c, names, k):
    """Name the called genes with the largest |x| on both sides, stacked from the top on the inner side of each point."""
    lo, hi = np.nanmin(y), np.nanmax(y)
    step = 0.065 * (hi - lo if hi > lo else 1.0)
    sides = {sg: np.where(c & (np.sign(x) == sg))[0] for sg in (1, -1)}
    k = min(k, int(c.sum()))
    n_up = int(round(k * len(sides[1]) / max(1, len(sides[1]) + len(sides[-1]))))
    # both sides named when both have calls
    if k >= 2 and len(sides[1]) and len(sides[-1]):
        n_up = min(max(n_up, 1), k - 1)
    for sign, n in ((1, n_up), (-1, k - n_up)):
        side = sides[sign]
        if not len(side) or n <= 0:
            continue
        for j, i in enumerate(side[np.argsort(-np.abs(x[side]), kind="stable")][:n]):
            ty = hi - (j + 0.5) * step
            tx = x[i] - sign * 0.06 * (np.nanmax(np.abs(x)) or 1.0)
            ax.annotate(
                names[i],
                (x[i], y[i]),
                xytext=(tx, ty),
                textcoords="data",
                ha="right" if sign > 0 else "left",
                va="center",
                fontsize=7,
                color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkA=0, shrinkB=1),
            )


def _project(E, method, random_state):
    if method in ("auto", "umap"):
        try:
            import umap

            return umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=random_state).fit_transform(E), "UMAP"
        except ImportError:
            if method == "umap":
                raise ImportError("method='umap' needs umap-learn:  pip install umap-learn")
            import warnings

            warnings.warn(
                "umap-learn is not installed; drawing the first two principal components instead (pip install umap-learn for UMAP)"
            )
    if method not in ("auto", "pca"):
        raise ValueError('method must be "auto", "umap" or "pca"')
    Z = E - E.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :2] * S[:2], "PC"


def embedding(
    emb,
    res=None,
    color="p_de",
    method="auto",
    called=None,
    fdr=0.05,
    rule=None,
    min_mean_count=None,
    label=0,
    ax=None,
    s=3,
    random_state=0,
):
    """2-D map of the embedding from embed(); returns (Axes, DataFrame of 2-D coordinates).

    emb: genes x d in matrix column order. res: the run() result for the same matrix; when given, only ranked genes
    are drawn, colored by "p_de", "lfc_mean" or "called" (up / down / not called).
    min_mean_count: drop genes with a lower mean raw count (needs res). method: "umap", "pca" or "auto" (UMAP if
    installed). label: name this many genes with the highest p_de. The map is descriptive only."""
    plt = _plt()
    E = np.asarray(emb, float)
    names = np.asarray(emb.index if isinstance(emb, pd.DataFrame) else np.arange(len(E))).astype(str)
    keep = np.ones(len(E), bool)
    r = None
    if res is not None:
        if len(res) != len(E):
            raise ValueError(f"emb has {len(E)} rows and res {len(res)}; compute both from the same matrix")
        keep &= _reported(res)
        if min_mean_count is not None:
            keep &= res["mean_count"].to_numpy(float) >= min_mean_count
        r = res[keep]
    elif color != "none":
        color = "none"
    xy, kind = _project(E[keep], method, random_state)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0))
    if color == "p_de":
        v = r["p_de"].to_numpy(float)
        o = np.argsort(v)
        sc = ax.scatter(
            xy[o, 0], xy[o, 1], c=v[o], cmap="viridis", vmin=0, vmax=1, s=s, lw=0, rasterized=True
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(
            rf"$P(|\log_2\mathrm{{FC}}| > {res.attrs.get('tau', 0.5):g})$"
        )
    elif color == "lfc_mean":
        v = r["lfc_mean"].to_numpy(float)
        lim = max(0.5, float(np.nanpercentile(np.abs(v), 99)))
        o = np.argsort(np.abs(v))
        sc = ax.scatter(
            xy[o, 0], xy[o, 1], c=v[o], cmap="RdBu_r", vmin=-lim, vmax=lim, s=s, lw=0, rasterized=True
        )
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02).set_label(r"posterior mean $\log_2$FC")
    elif color == "called":
        c = _calls(res, called, fdr, rule)[keep]
        x = r["lfc_mean"].to_numpy(float)
        ax.scatter(xy[~c, 0], xy[~c, 1], s=s, color=GREY, alpha=0.5, lw=0, rasterized=True)
        for m, col, lab in ((c & (x > 0), UP, "up"), (c & (x < 0), DOWN, "down")):
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=s + 2,
                color=col,
                lw=0,
                rasterized=True,
                label=f"{lab} ({int(m.sum()):,})",
            )
        if c.any():
            ax.legend(frameon=False, fontsize=8, markerscale=2)
    elif color == "none":
        ax.scatter(xy[:, 0], xy[:, 1], s=s, color=INK, alpha=0.5, lw=0, rasterized=True)
    else:
        raise ValueError('color must be "p_de", "lfc_mean", "called" or "none"')
    gnames = r["gene"].astype(str).to_numpy() if r is not None and "gene" in r else names[keep]
    if label and r is not None:
        top = np.argsort(-r["p_de"].to_numpy(float), kind="stable")[:label]
        # stack the names top-down in a column beside the points
        top = top[np.argsort(-xy[top, 1], kind="stable")]
        for j, i in enumerate(top):
            ax.annotate(
                gnames[i],
                (xy[i, 0], xy[i, 1]),
                xytext=(18, 9 * (len(top) / 2 - j)),
                textcoords="offset points",
                fontsize=7,
                va="center",
                color=INK,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5, shrinkA=0, shrinkB=1),
            )
    ax.set_xlabel(f"{kind} 1")
    ax.set_ylabel(f"{kind} 2")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax, pd.DataFrame(xy, index=gnames, columns=[f"{kind}1", f"{kind}2"])
