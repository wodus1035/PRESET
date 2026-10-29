"""Add labels without DE methods to the real-cohort bank (train.py --cons_label observed).

For fold f of a cohort the label of a gene is the log2 ratio of the normalized group means on the other folds:
    size factors  median of ratios on the reference samples
    lfc_obs       log2((mean_case + 0.5) / (mean_ctrl + 0.5)) of the normalized counts
    se_obs        sqrt(var1 / n1 + var0 / n0) of log2(normalized + 0.5) within each group
A gene is labeled when its reference mean is at least MIN_MEAN in one group and se_obs <= MAX_SE, and its label is
exactly 0 when |lfc_obs| < ZERO_LFC and se_obs < ZERO_SE. Every other gene is masked.
    python training/add_observed_labels.py
"""

import argparse
import glob
import json
import os

import numpy as np

MIN_MEAN = 10.0
MAX_SE = 0.5
ZERO_LFC = 0.1
ZERO_SE = 0.15


def size_factors(X):
    """Median of ratios over the genes counted in every sample, or library sizes when fewer than 200 such genes."""
    pos = (X > 0).all(1)
    if pos.sum() < 200:
        s = X.sum(0).astype(float)
        return s / np.exp(np.log(s).mean())
    lg = np.log(X[pos].astype(float))
    ref = lg.mean(1, keepdims=True)
    return np.exp(np.median(lg - ref, axis=0))


def observed_labels(X, cond, fold, K):
    lab_obs = np.full((K, X.shape[0]), np.nan, np.float32)
    se_obs = np.full((K, X.shape[0]), np.nan, np.float32)
    for f in range(K):
        # Normalized counts of the reference folds
        ref = fold != f
        Xr = X[:, ref]
        cr = cond[ref]
        N = Xr / size_factors(Xr)[None, :]

        # Log2 ratio of the group means and its standard error
        m0 = N[:, cr == 0].mean(1)
        m1 = N[:, cr == 1].mean(1)
        L = np.log2(N + 0.5)
        v0 = L[:, cr == 0].var(1, ddof=1)
        v1 = L[:, cr == 1].var(1, ddof=1)
        se = np.sqrt(v1 / (cr == 1).sum() + v0 / (cr == 0).sum())
        lfc = np.log2((m1 + 0.5) / (m0 + 0.5))

        # Labels
        expressed = (m0 >= MIN_MEAN) | (m1 >= MIN_MEAN)
        scored = expressed & (se <= MAX_SE)
        zero = scored & (np.abs(lfc) < ZERO_LFC) & (se < ZERO_SE)
        lab = np.full(X.shape[0], np.nan, np.float32)
        lab[scored] = lfc[scored]
        lab[zero] = 0.0

        lab_obs[f] = lab
        se_obs[f] = se.astype(np.float32)
    return lab_obs, se_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bank", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "consensus_bank")
    )
    a = ap.parse_args()

    for fn in sorted(glob.glob(os.path.join(a.bank, "*.npz"))):
        z = dict(np.load(fn, allow_pickle=False))
        meta = json.loads(str(z["meta"]))
        X = z["X"].astype(np.int64)
        cond = z["cond"].astype(int)
        fold = z["fold"].astype(int)
        z["lab_obs"], z["se_obs"] = observed_labels(X, cond, fold, meta["K"])

        meta["obs_rules"] = dict(min_mean=MIN_MEAN, max_se=MAX_SE, zero_lfc=ZERO_LFC, zero_se=ZERO_SE)
        z["meta"] = json.dumps(meta)
        np.savez_compressed(fn, **z)
        print(
            f"{meta['name']}: {int(np.isfinite(z['lab_obs']).sum(1).mean())} labeled genes per fold",
            flush=True,
        )


if __name__ == "__main__":
    main()
