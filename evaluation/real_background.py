"""Real-background training data: GTEx count matrices of tissues not used in evaluation, with planted condition
effects of known size.

Real disease effects overlap within-group donor variation, which the synthetic prior does not generate, and the
model discounted effects planted along within-group PC1 (power 0.64 -> 0.20; edgeR 0.56 -> 0.33).

sample_batch() returns the batch layout of the synthetic prior (counts (B,n,G), cond, batch, lfc_true, is_null,
has_batch) plus real_tissue, real_kind and real_samples. Effects per dataset (independent draws):
  sparse  : with prob p_sparse, a fraction ~ U(sparse_frac) of genes gets N(0, sd ~ U(0.5, 2.0))
  aligned : with prob p_aligned, the top fraction ~ U(aligned_frac) of genes by |loading| on within-group PC k
            (k in 1..3) gets sign(loading) * |N(0, sd ~ U(aligned_sd))| * a random overall sign
  dense   : with prob p_dense, a fraction ~ U(dense_frac) of genes gets N(0, sd ~ U(0.1, 0.4))
  huge    : with prob p_huge, 0.1-1% of genes get |lfc| ~ U(2, 8) and no other effects
At least MIN_NULL_FRAC of genes keep an effect of exactly 0. `inject` sets how effects are planted (see _inject).

GTEx donors are split 50/50 at random into `bank` and `eval` (data_cache/gtex_donor_split_v3.json). Training uses
`bank` donors only, so evaluations with donor_split="eval" are donor-disjoint from training.
"""

import os, glob, json, numpy as np, pandas as pd, torch

GTEX = os.path.join(
    os.environ.get("PRESET_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")), "gtex"
)
EVAL_TISSUES = {
    "lung",
    "thyroid",
    "whole_blood",
    "muscle_skeletal",
    "adipose_subcutaneous",
    "skin_sun_exposed_lower_leg",
    "brain_cortex",
    "colon_transverse",
    "stomach",
    "heart_left_ventricle",
    "artery_aorta",
    "nerve_tibial",
}
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
BANK = os.path.join(_CACHE, "gtex_real_bank.npz")  # all donors of the 11 training tissues
BANK_V3 = os.path.join(_CACHE, "gtex_real_bank_v3.npz")  # `bank` donors only
# held-out donors of the 11 training tissues
BANK_V3_EVAL = os.path.join(_CACHE, "gtex_real_bank_v3_eval.npz")
# {"bank": [...], "eval": [...], "seed", "date", ...}
DONOR_SPLIT_V3 = os.path.join(_CACHE, "gtex_donor_split_v3.json")
MIN_NULL_FRAC = 0.30  # floor on exactly-null genes, which identify the dataset-level offset
INJECT_MODES = ("poisson", "multiplicative", "thinning")


def donor_of(sample_ids):
    """GTEx sample id -> donor id (`GTEX-XXXXX` prefix). Works on a list/array/Series of strings."""
    return pd.Series(np.asarray(sample_ids, dtype=str)).str.extract(r"^(GTEX-[^-]+)")[0].to_numpy()


def load_donor_split(path=DONOR_SPLIT_V3):
    """Donor split dict with keys bank and eval (lists of donor ids)."""
    with open(path) as f:
        return json.load(f)


def read_gct_samples(path):
    """Sample ids (column header) of a GTEx .gct.gz without loading the matrix."""
    import gzip

    with gzip.open(path, "rt") as f:
        f.readline()
        f.readline()
        return f.readline().rstrip("\n").split("\t")[2:]


def build_bank(min_samples=60, verbose=True, out=BANK, donors=None, store_samples=False):
    """Write a bank npz {X_<tissue>: genes x samples int32, tissues} from the non-evaluation GTEx tissues.
    donors: optional donor ids to keep. store_samples: add samples_<tissue> so a donor split can be checked later.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mats, ids, names = {}, {}, []
    keep_set = None if donors is None else set(donors)
    for f in sorted(glob.glob(os.path.join(GTEX, "gene_reads_v10_*.gct.gz"))):
        t = os.path.basename(f)[len("gene_reads_v10_") : -len(".gct.gz")]
        if t in EVAL_TISSUES:
            continue
        df = pd.read_csv(f, sep="\t", skiprows=2, compression="gzip")
        samples = [c for c in df.columns if c not in ("Name", "Description")]
        X = df[samples].to_numpy(np.int32)
        samples = np.array(samples)
        if keep_set is not None:
            m = np.array([d in keep_set for d in donor_of(samples)])
            X = X[:, m]
            samples = samples[m]
        if X.shape[1] < min_samples:
            if verbose:
                print(f"  {t}: {X.shape[1]} samples < {min_samples}, skipped", flush=True)
            continue
        mats[t] = X
        ids[t] = samples
        names.append(t)
        if verbose:
            print(f"  {t}: {X.shape[1]} samples", flush=True)
    arrays = {f"X_{t}": mats[t] for t in names}
    if store_samples:
        arrays.update({f"samples_{t}": ids[t] for t in names})
    np.savez_compressed(out, **arrays, tissues=np.array(names))
    return names


def _fix_depth(cnt_before, cnt_after, rng):
    """Restore the library-size ratios the samples had before planting.

    Thinning only removes reads, so planting shifts the depth ratio between the halves in the direction of the
    effects (log2 0.040 with 10% of genes planted). Each sample is thinned once more by kappa / r_i, with
    r_i = total_after_i / total_before_i and kappa = min_i r_i. Thinning twice is still thinning, so the count
    distribution and each donor's realization are kept."""
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


def _inject(cnt, lfc, h, rng, mode="poisson"):
    """Give genes with lfc != 0 a fold change of 2^lfc between the control half (columns :h) and the case half (h:).

    mode="thinning" (exact, used for training and evaluation): binomial thinning of the half that goes down, the
    control half by 2^-lfc for up genes and the case half by 2^lfc for down genes. Thinning NB(mu, phi) counts gives
    NB(c*mu, phi) and keeps each donor's realization.
    mode="poisson": Poisson(count * 2^lfc) on the case half. It leaks: at mean counts 1-10 resampled genes get 0.010
    more P(DE) than untouched ones at a true lfc of 0 (see Hejblum et al. 2024 on injection artifacts).
    mode="multiplicative": round(count * 2^lfc) on the case half; zeros stay zero and low-count variance is wrong.
    """
    if mode not in INJECT_MODES:
        raise ValueError(f"inject must be one of {INJECT_MODES}, got {mode!r}")
    m = lfc != 0
    if not m.any():
        return cnt
    if mode == "thinning":
        up = m & (lfc > 0)
        dn = m & (lfc < 0)
        # control half down
        if up.any():
            cnt[up, :h] = rng.binomial(cnt[up, :h].astype(np.int64), (2.0 ** -lfc[up])[:, None])
        # case half down
        if dn.any():
            cnt[dn, h:] = rng.binomial(cnt[dn, h:].astype(np.int64), (2.0 ** lfc[dn])[:, None])
        return cnt
    scaled = cnt[m, h:] * (2.0 ** lfc[m])[:, None]
    cnt[m, h:] = rng.poisson(scaled) if mode == "poisson" else np.round(scaled)
    return cnt


class RealBank:
    """Sampler of real-background datasets. path: bank npz. inject: one of INJECT_MODES. donor_split: None, a split
    json path or dict; columns whose donor is not in split[side] are dropped (needs samples_<tissue> in the npz).
    """

    def __init__(
        self,
        path,
        inject,
        donor_split,
        side="bank",
        fix_depth=True,
        p_sparse=0.6,
        p_aligned=0.6,
        p_dense=0.4,
        p_huge=0.0,
        min_count=10,
        aligned_sd=(0.5, 1.5),
        aligned_frac=(0.05, 0.30),
        p_het=0.0,
        het_sd=(0.3, 1.0),
        het_side="case",
        sparse_frac=(0.02, 0.30),
        dense_frac=(0.3, 1.0 - MIN_NULL_FRAC),
        frac_log=False,
    ):
        """path, inject and donor_split have no defaults so every caller states the bank, planting mode and donor
        split it uses; RealBank.v3() and RealBank.v3_eval() give the paper settings."""
        if inject not in INJECT_MODES:
            raise ValueError(f"inject must be one of {INJECT_MODES}, got {inject!r}")
        z = np.load(path, allow_pickle=True)
        self.tissues = list(z["tissues"])
        self.X = {t: z[f"X_{t}"] for t in self.tissues}
        self.samples = {t: z[f"samples_{t}"].astype(str) for t in self.tissues if f"samples_{t}" in z.files}
        self.path, self.inject, self.donor_split, self.side = path, inject, None, side
        if donor_split is not None:
            split = (
                load_donor_split(donor_split) if isinstance(donor_split, (str, os.PathLike)) else donor_split
            )
            bank = set(split[side])
            self.donor_split = split
            for t in self.tissues:
                if t not in self.samples:
                    raise ValueError(
                        f"{path} stores no sample ids for {t}; cannot apply a donor split (rebuild with store_samples=True)"
                    )
                m = np.array([d in bank for d in donor_of(self.samples[t])])
                if not m.all():
                    self.X[t] = self.X[t][:, m]
                    self.samples[t] = self.samples[t][m]
                if self.X[t].shape[1] == 0:
                    raise ValueError(f"{t}: no `{side}` donors left after applying the donor split")
        self.p_sparse, self.p_aligned, self.p_dense, self.p_huge, self.min_count = (
            p_sparse,
            p_aligned,
            p_dense,
            p_huge,
            min_count,
        )
        # aligned effect SD and gene-fraction ranges (the defaults leave strong programs on few genes out of range)
        self.aligned_sd, self.aligned_frac = tuple(aligned_sd), tuple(aligned_frac)
        # frac_log: draw the sparse and aligned fractions log-uniformly
        self.sparse_frac, self.dense_frac, self.frac_log = (
            tuple(sparse_frac),
            tuple(dense_frac),
            bool(frac_log),
        )
        # with prob p_het the effect varies between samples around the label (see _plant); real disease cohorts show
        # extra case-side variance in DE genes, which a constant effect lacks
        self.p_het, self.het_sd = p_het, tuple(het_sd)
        # het_side: "case" = case samples, "high" = the higher side (case for up, control for down genes), "both" = every
        # sample. The model discounts effects whose higher side carries donor variation, which hides real down-regulated genes.
        if het_side not in ("case", "high", "both"):
            raise ValueError(f"het_side must be case|high|both, got {het_side!r}")
        self.het_side = het_side
        self.fix_depth = fix_depth

    @classmethod
    def v3(cls, path=BANK_V3, donor_split=DONOR_SPLIT_V3, inject="thinning", p_aligned=0.75, **kw):
        """Training bank: `bank` donors with split check, binomial thinning, p_aligned 0.75."""
        return cls(path=path, donor_split=donor_split, inject=inject, p_aligned=p_aligned, **kw)

    @classmethod
    def v3_eval(cls, p_huge=0.35, **kw):
        """Evaluation bank: held-out donors of the 11 training tissues, binomial thinning. These are held-out donors,
        not held-out tissues; report the two separately."""
        return cls(
            path=BANK_V3_EVAL,
            inject="thinning",
            donor_split=DONOR_SPLIT_V3,
            side="eval",
            p_aligned=0.75,
            p_huge=p_huge,
            **kw,
        )

    def provenance(self):
        """Bank settings to store with every result table."""
        return {
            "bank_path": os.path.basename(self.path),
            "inject": self.inject,
            "donor_side": self.side,
            "donor_split": (
                os.path.basename(self.donor_split.get("split", ""))
                if isinstance(self.donor_split, dict)
                else None
            )
            or ("applied" if self.donor_split else "none"),
            "n_tissues": len(self.tissues),
            "n_samples": int(sum(v.shape[1] for v in self.X.values())),
        }

    def donors(self, tissue=None):
        """Donor ids in the bank after the split; needs samples_<tissue>."""
        ts = self.tissues if tissue is None else [tissue]
        return sorted(set(np.concatenate([donor_of(self.samples[t]) for t in ts])))

    def _plant(self, cnt, lfc, h, rng):
        """Constant effect via _inject or, with prob p_het, a sample-specific one by thinning. Draws no random numbers
        for this when p_het == 0, so the stream matches a bank without heterogeneity."""
        if self.p_het > 0 and rng.random() < self.p_het:
            n = cnt.shape[1]
            s = rng.uniform(*self.het_sd)
            nz = lfc != 0
            out = cnt.copy()
            # amplitude m per case sample, E[m] = 1, so the label is the mean effect
            if self.het_side == "case":
                m = np.exp(rng.normal(0, s, n - h) - s * s / 2)
                e = np.zeros((len(lfc), n))
                e[:, h:] = lfc[:, None] * m[None, :]
            # additive log2 offset per sample, mean 0
            else:
                m = rng.normal(0, s, n) / np.log(2.0)  # s is a natural-log SD
                # centered within each side so the mean effect equals the label
                m[:h] -= m[:h].mean()
                m[h:] -= m[h:].mean()
                e = np.zeros((len(lfc), n))
                e[:, h:] = lfc[:, None]
                if self.het_side == "both":
                    e[nz] += m[None, :]
                else:
                    up = nz & (lfc > 0)
                    dn = nz & (lfc < 0)
                    e[up, h:] += m[None, h:]
                    e[dn, :h] += m[None, :h]
            # shift so the highest-effect sample is untouched; factors <= 1
            f = 2.0 ** (e - np.maximum(e.max(1, keepdims=True), 0.0))
            out[nz] = rng.binomial(out[nz].astype(np.int64), f[nz])
            return out
        return _inject(cnt.copy(), lfc, h, rng, self.inject)

    def _maybe_fix_depth(self, before, after, rng):
        return _fix_depth(before, after, rng) if self.fix_depth else after

    @staticmethod
    def _frac(lo_hi, rng, log):
        lo, hi = lo_hi
        return (
            float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            if log and lo > 0
            else float(rng.uniform(lo, hi))
        )

    def _one(self, rng, n_per_group, G):
        n = 2 * n_per_group
        ok = [t for t in self.tissues if self.X[t].shape[1] >= n] or [
            max(self.tissues, key=lambda t: self.X[t].shape[1])
        ]
        t = ok[rng.integers(len(ok))]
        X = self.X[t]
        n = min(n, X.shape[1])
        n_per_group = n // 2
        n = 2 * n_per_group
        cols = rng.choice(X.shape[1], n, replace=False)
        cnt = X[:, cols].astype(np.float64)
        # for nuisance labels (SMCENTER, sex, Hardy); no rng use
        self._last_samples = self.samples[t][cols] if t in self.samples else None
        expressed = np.where((cnt >= self.min_count).sum(1) >= max(2, n // 4))[0]
        genes = rng.choice(expressed, G, replace=len(expressed) < G)
        cnt = cnt[genes]
        h = n_per_group
        lfc = np.zeros(G)
        kinds = []
        if self.p_huge > 0 and rng.random() < self.p_huge:
            # near-null dataset with a few very large effects (like GTEx sex genes), exclusive with the other modes
            k = max(1, int(rng.uniform(0.001, 0.01) * G))
            g_ = rng.choice(G, k, replace=False)
            lfc[g_] = rng.choice([-1.0, 1.0], k) * rng.uniform(2.0, 8.0, k)
            kinds.append("huge")
            cnt = self._maybe_fix_depth(cnt, self._plant(cnt, lfc, h, rng), rng)
            return cnt.T, lfc, t, "+".join(kinds)
        if rng.random() < self.p_sparse:
            de = rng.random(G) < self._frac(self.sparse_frac, rng, self.frac_log)
            lfc[de] += rng.normal(0, rng.uniform(0.5, 2.0), de.sum())
            kinds.append("sparse")
        if rng.random() < self.p_aligned:
            y = np.log1p(cnt)
            R = y - y.mean(1, keepdims=True)
            _, _, vt = np.linalg.svd(R.T, full_matrices=False)
            k = rng.integers(0, min(3, vt.shape[0]))
            load = vt[k]
            top = np.argsort(-np.abs(load))[
                : max(1, int(self._frac(self.aligned_frac, rng, self.frac_log) * G))
            ]
            lfc[top] += (
                np.sign(load[top])
                * np.abs(rng.normal(0, rng.uniform(*self.aligned_sd), len(top)))
                * rng.choice([-1.0, 1.0])
            )
            kinds.append(f"alignedPC{k+1}")
        if rng.random() < self.p_dense:
            # a subset only: the floor below keeps MIN_NULL_FRAC of genes exactly null
            sub = rng.random(G) < self._frac(self.dense_frac, rng, False)
            lfc[sub] += rng.normal(0, rng.uniform(0.1, 0.4), int(sub.sum()))
            kinds.append("dense")
        nz = lfc != 0
        if nz.mean() > 1.0 - MIN_NULL_FRAC:
            drop = rng.choice(
                np.where(nz)[0], int(round((nz.mean() - (1.0 - MIN_NULL_FRAC)) * G)), replace=False
            )
            lfc[drop] = 0.0
            kinds.append("nullfloor")
        cnt = self._maybe_fix_depth(cnt, self._plant(cnt, lfc, h, rng), rng)
        return cnt.T, lfc, t, "+".join(kinds) or "null"

    def sample_batch(self, B, n_per_group, G, rng, device):
        out, samp = [], []
        for _ in range(B):
            out.append(self._one(rng, n_per_group, G))
            samp.append(self._last_samples)
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
            "has_batch": torch.zeros(B, device=device),
            "real_tissue": [o[2] for o in out],
            "real_kind": [o[3] for o in out],
            "real_samples": samp,
        }
