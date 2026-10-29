"""Self-checks of whether PRESET's probabilities can be trusted on a given dataset.

null_split   : split one group in half at random and run the model; every call is a false positive.
donor_axis   : share of within-group variance on the first principal axis. Above ~0.4 a common factor (donor
               program, strong batch) dominates and effects along it are called conservatively.
depth        : median library size and mean count per gene; validated down to ~25 counts per gene.
depth_ratio  : log2 ratio of the groups' mean size factors. Decoys move raw counts between samples, so above
               |log2 r| = 0.3 the decoy cut-off is conservative (at r = 2, power 0.06-0.20 instead of 0.18-0.46,
               realized FDR <= 0.04). The ranking and the posterior are not affected.
"""

import numpy as np


def null_split(model, dev, X, cond, batch, tau=0.5, reps=5, seed=0, min_group=4, covar=None):
    """Fraction of genes with p_de > 0.9 when one group is split at random, one value per repeat."""
    import torch

    rng = np.random.default_rng(seed)
    out = []
    for g in (0, 1):
        idx = np.where(cond == g)[0]
        if len(idx) < min_group:
            continue
        for _ in range(reps):
            s = rng.permutation(idx)
            h = len(s) // 2
            sub = s[: 2 * h]
            c = np.array([0] * h + [1] * h)
            with torch.no_grad():
                o = model.predict(
                    torch.tensor(X[sub], dtype=torch.float32, device=dev)[None],
                    torch.tensor(c, dtype=torch.float32, device=dev)[None],
                    torch.tensor(batch[sub], dtype=torch.float32, device=dev)[None],
                    tau=tau,
                    covar=(
                        None
                        if covar is None
                        else torch.tensor(covar[sub], dtype=torch.float32, device=dev)[None]
                    ),
                )
            out.append(float((o["p_de"][0].cpu().numpy() > 0.9).mean()))
    return out


def donor_axis(X, cond):
    y = np.log1p(X / X.sum(1, keepdims=True) * 1e6)
    R = y.copy()
    for g in (0, 1):
        m = cond == g
        if m.sum() > 1:
            R[m] -= R[m].mean(0, keepdims=True)
    s = np.linalg.svd(R, compute_uv=False)
    share = s**2 / max((s**2).sum(), 1e-12)
    return float(share[0]), float(share[1]) if len(share) > 1 else 0.0


def depth_ratio(X, cond):
    """log2 ratio of the mean size factor of group 1 over group 0."""
    from . import _size_factors

    sf = _size_factors(X)
    return float(np.log2(sf[cond == 1]).mean() - np.log2(sf[cond == 0]).mean())


def report(model, dev, X, cond, batch, tau=0.5, covar=None):
    lr = depth_ratio(X, cond)
    pc1, pc2 = donor_axis(X, cond)
    nulls = null_split(model, dev, X, cond, batch, tau=tau, covar=covar)
    lib = X.sum(1)
    per_gene = X.mean()
    flags = []
    if nulls and max(nulls) > 0.01:
        flags.append(
            f"null-split false calls up to {max(nulls)*100:.1f}% of genes: probabilities are not trustworthy on this data"
        )
    if pc1 > 0.4:
        flags.append(
            f"dominant within-group axis (PC1 share {pc1:.2f}): effects along it will be called conservatively; rely on ranking / expected false discoveries"
        )
    if per_gene < 25:
        flags.append(
            f"shallow counts (mean {per_gene:.0f}/gene): validated down to ~25; below that effect sizes shrink toward 0"
        )
    if min((cond == 0).sum(), (cond == 1).sum()) < 2:
        flags.append("fewer than 2 samples in a group: not supported")
    if abs(lr) > 0.3:
        flags.append(
            f"unequal sequencing depth between groups (log2 size-factor ratio {lr:+.2f}): the decoy cut-off is conservative "
            "here and may call too few genes; rely on ranking / expected false discoveries"
        )
    return {
        "null_split_fraction_called": nulls,
        "within_group_pc1_share": pc1,
        "within_group_pc2_share": pc2,
        "median_library": int(np.median(lib)),
        "mean_count_per_gene": float(per_gene),
        "log2_depth_ratio": lr,
        "flags": flags,
    }
