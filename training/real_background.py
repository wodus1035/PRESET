"""Real-background training tasks: GTEx count matrices of the training tissues and donors, with planted condition effects.

Real disease effects overlap the donor-to-donor variation within each group, which the SCM prior does not generate.
A task takes 2n samples of one tissue, calls the first n controls and the rest cases, and plants effects of known size
by binomial thinning (see _inject). The bank holds only donors of the `bank` half of the GTEx donor split, so the
GTEx evaluations (evaluation half) never share a donor with training.

Effects planted in one task (independent draws):
  huge    : with prob. p_huge, 0.1-1% of genes get |LFC| 2-8 and every other gene stays null (exclusive)
  sparse  : with prob. 0.6, a fraction U(0.02, 0.30) of genes get N(0, sd U(0.5, 2.0))
  aligned : with prob. p_aligned, the genes loading most on one of the first three within-group PCs get effects along
            the loading, sd U(aligned_sd), on a fraction U(aligned_frac) of genes
  dense   : with prob. 0.4, a fraction U(0.3, 0.7) of genes get N(0, sd U(0.1, 0.4))
At least MIN_NULL_FRAC of the genes keep an effect of exactly zero.
"""

import json

import numpy as np
import pandas as pd
import torch

MIN_NULL_FRAC = 0.30  # same floor as the SCM prior, the null genes identify the compositional offset


def donor_of(sample_ids):
    """GTEx sample id -> donor id (the GTEX-XXXXX prefix)."""
    ids = pd.Series(np.asarray(sample_ids, dtype=str))
    return ids.str.extract(r"^(GTEX-[^-]+)")[0].to_numpy()


def load_donor_split(path):
    """Donor split json, with the donor ids of the `bank` and `eval` halves."""
    with open(path) as f:
        return json.load(f)


def _inject(cnt, lfc, h, rng):
    """Give the genes with lfc != 0 a fold change of 2^lfc between the control half (columns :h) and the case half (h:).

    Binomial thinning of whichever half has to go down. If X | lambda ~ Poisson(lambda), Binomial(X, c) is Poisson(c lambda)
    under the same mixture, so a negative binomial count NB(mu, phi) becomes NB(c mu, phi) with its dispersion and its
    donor-level realization kept. Up-regulated genes thin the control half by 2^-lfc and down-regulated genes thin the
    case half by 2^lfc, and null genes are not touched."""
    m = lfc != 0
    if not m.any():
        return cnt

    up = m & (lfc > 0)
    dn = m & (lfc < 0)
    if up.any():
        cnt[up, :h] = rng.binomial(cnt[up, :h].astype(np.int64), (2.0 ** -lfc[up])[:, None])
    if dn.any():
        cnt[dn, h:] = rng.binomial(cnt[dn, h:].astype(np.int64), (2.0 ** lfc[dn])[:, None])
    return cnt


def _fix_depth(cnt_before, cnt_after, rng):
    """Thin the planted matrix back to the library-size ratios the samples had before planting.

    Thinning only removes reads, so planting changes the depth ratio between the two halves in the direction of the
    planted effects. With r_i = total_after_i / total_before_i and kappa = min_i r_i, sample i is thinned by
    kappa / r_i <= 1, which leaves every sample at kappa * total_before_i."""
    t0 = cnt_before.sum(0).astype(np.float64)
    t1 = cnt_after.sum(0).astype(np.float64)
    r = t1 / np.maximum(t0, 1.0)
    kappa = float(r.min())

    out = cnt_after
    for j in range(out.shape[1]):
        p_ = kappa / max(r[j], 1e-12)
        if p_ < 1.0:
            out[:, j] = rng.binomial(out[:, j].astype(np.int64), p_)
    return out


class RealBank:
    """Tasks from a bank of real count matrices (build_real_bank.py, build_mouse_bank.py). donor_split: path of the GTEx
    split json; every sample whose donor is not in its `bank` half is dropped, so a bank built from other donors cannot be
    used by mistake. None for a bank without GTEx donors."""

    def __init__(
        self, path, donor_split, p_huge=0.35, p_aligned=0.75, aligned_sd=(0.5, 3.0), aligned_frac=(0.02, 0.30)
    ):
        z = np.load(path, allow_pickle=True)
        self.path = path
        self.tissues = [str(t) for t in z["tissues"]]
        self.X = {t: z[f"X_{t}"] for t in self.tissues}
        self.samples = {t: z[f"samples_{t}"].astype(str) for t in self.tissues}

        # Keep the bank-half donors only
        if donor_split is not None:
            bank = set(load_donor_split(donor_split)["bank"])
            for t in self.tissues:
                m = np.array([d in bank for d in donor_of(self.samples[t])])
                if not m.all():
                    self.X[t] = self.X[t][:, m]
                    self.samples[t] = self.samples[t][m]
                if self.X[t].shape[1] == 0:
                    raise ValueError(f"{t}: no bank donors left after applying the donor split")

        self.p_huge = p_huge
        self.p_sparse = 0.6
        self.p_aligned = p_aligned
        self.p_dense = 0.4
        self.min_count = 10  # a gene is expressed if this count is reached in a quarter of the samples
        self.sparse_frac = (0.02, 0.30)
        self.aligned_sd = tuple(aligned_sd)
        self.aligned_frac = tuple(aligned_frac)
        self.dense_frac = (0.3, 1.0 - MIN_NULL_FRAC)

    def _plant(self, cnt, lfc, h, rng):
        planted = _inject(cnt.copy(), lfc, h, rng)
        return _fix_depth(cnt, planted, rng)

    def _one(self, rng, n_per_group, G):
        # Samples of one tissue and G expressed genes
        n = 2 * n_per_group
        ok = [t for t in self.tissues if self.X[t].shape[1] >= n]
        if not ok:
            ok = [max(self.tissues, key=lambda t: self.X[t].shape[1])]
        t = ok[rng.integers(len(ok))]
        X = self.X[t]
        n = min(n, X.shape[1])
        n_per_group = n // 2
        n = 2 * n_per_group

        cols = rng.choice(X.shape[1], n, replace=False)
        cnt = X[:, cols].astype(np.float64)
        expressed = np.where((cnt >= self.min_count).sum(1) >= max(2, n // 4))[0]
        genes = rng.choice(expressed, G, replace=len(expressed) < G)
        cnt = cnt[genes]
        h = n_per_group
        lfc = np.zeros(G)

        # A few very large effects on an otherwise null background
        if self.p_huge > 0 and rng.random() < self.p_huge:
            k = max(1, int(rng.uniform(0.001, 0.01) * G))
            g_ = rng.choice(G, k, replace=False)
            lfc[g_] = rng.choice([-1.0, 1.0], k) * rng.uniform(2.0, 8.0, k)
            cnt = self._plant(cnt, lfc, h, rng)
            return cnt.T, lfc

        # Sparse effects
        if rng.random() < self.p_sparse:
            de = rng.random(G) < rng.uniform(*self.sparse_frac)
            lfc[de] += rng.normal(0, rng.uniform(0.5, 2.0), de.sum())

        # Effects along a within-group principal component
        if rng.random() < self.p_aligned:
            y = np.log1p(cnt)
            R = y - y.mean(1, keepdims=True)
            _, _, vt = np.linalg.svd(R.T, full_matrices=False)
            k = rng.integers(0, min(3, vt.shape[0]))
            load = vt[k]
            n_top = max(1, int(rng.uniform(*self.aligned_frac) * G))
            top = np.argsort(-np.abs(load))[:n_top]
            size = np.abs(rng.normal(0, rng.uniform(*self.aligned_sd), len(top)))
            lfc[top] += np.sign(load[top]) * size * rng.choice([-1.0, 1.0])

        # Dense small effects
        if rng.random() < self.p_dense:
            sub = rng.random(G) < rng.uniform(*self.dense_frac)
            lfc[sub] += rng.normal(0, rng.uniform(0.1, 0.4), int(sub.sum()))

        # Null floor over whatever combination fired
        nz = lfc != 0
        if nz.mean() > 1.0 - MIN_NULL_FRAC:
            n_drop = int(round((nz.mean() - (1.0 - MIN_NULL_FRAC)) * G))
            drop = rng.choice(np.where(nz)[0], n_drop, replace=False)
            lfc[drop] = 0.0

        cnt = self._plant(cnt, lfc, h, rng)
        return cnt.T, lfc

    def sample_batch(self, B, n_per_group, G, rng, device):
        out = [self._one(rng, n_per_group, G) for _ in range(B)]
        counts = torch.tensor(np.stack([o[0] for o in out]), dtype=torch.float32, device=device)
        lfc = torch.tensor(np.stack([o[1] for o in out]), dtype=torch.float32, device=device)
        cond = (
            torch.cat([torch.zeros(n_per_group), torch.ones(n_per_group)])
            .to(device)
            .expand(B, -1)
            .contiguous()
        )
        return {
            "counts": counts,
            "cond": cond,
            "batch": torch.zeros_like(cond),
            "lfc_true": lfc,
            "is_null": torch.zeros(B, device=device),
        }
