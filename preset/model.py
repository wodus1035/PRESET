"""PRESET network.

    counts (B,n,G), design (B,n,2)
      -> per-gene tokens over samples             (B*G, n, d)
      -> GeneEncoder: attention over samples       -> h_g   (B,G,d)
      -> LatentExchange: genes <-> K latent tokens -> h_g'  (B,G,d), latents (B,K,d)
      -> head (ModelConfig.head):
           "bar"        BarHead: categorical over LFC bins           -> logits (B,G,nbins)
           "mix_t"      MixTHead: spike at 0 + student-t slab         -> params (B,G,3|4)
           "mix_updown" MixUpDownHead: spike at 0 + up/down t slabs   -> params (B,G,6|7)

Heads share forward, nll, mean, prob_abs_gt and `kind`; mixture heads add mixture(), read by posterior.summarize.
The model is permutation-invariant over genes and over samples, and its cost is linear in G.
"""

from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d: int = 128
    n_heads: int = 4
    enc_layers: int = 2
    exch_layers: int = 3
    n_latents: int = 128
    n_bins: int = 200
    lfc_range: float = 8.0  # bins cover [-lfc_range, lfc_range] in log2
    dropout: float = 0.0
    # optional parts; the defaults let checkpoints without these keys load
    use_stats: bool = False  # add an embedding of per-gene classical statistics (gene_stats)
    evidence_head: bool = False  # head also reads the pre-exchange and stats embeddings
    use_covar: bool = False  # append the covariate block (io.build_covariates) to each sample token
    covar_dim: int = 7
    sc_feats: int = 0  # extra per-(gene, sample) features for pseudobulk input; 0 = bulk model
    tail_scale: float = 0.0  # > 0: half-normal tails on the two edge bins; 0 clamps at +-lfc_range
    head: str = "bar"  # see HEADS
    head_hidden: int = 128  # MLP width of the mixture heads (0 = one linear layer)
    slab_df: float = 4.0  # student-t df of the slab(s) when slab_df_learn is False
    slab_df_learn: bool = False  # per-gene df = 2 + softplus(raw)


# features
def featurize(
    counts: torch.Tensor,
    cond: torch.Tensor,
    batch: torch.Tensor,
    covar: torch.Tensor = None,
    extra: torch.Tensor = None,
) -> torch.Tensor:
    """(B,n,G) counts -> (B,G,n,F) per-gene sample tokens. Size factors are median-of-ratios over genes
    nonzero in all samples; raw depth is also given so the model can correct them."""
    B, n, G = counts.shape
    eps = 1e-8
    log_c = torch.log1p(counts)
    logg = torch.log(counts + 0.5)
    ref = logg.mean(1, keepdim=True)  # (B,1,G)
    ok = (counts > 0).all(1, keepdim=True)  # genes nonzero in all samples
    ratio = (logg - ref).masked_fill(~ok, float("nan"))
    log_sf = torch.nanmedian(ratio, dim=2).values  # (B,n)
    log_sf = torch.nan_to_num(log_sf, nan=0.0)
    log_sf = log_sf - log_sf.mean(1, keepdim=True)

    x = torch.log1p(counts / torch.exp(log_sf).unsqueeze(-1))  # normalized log expr (B,n,G)
    gene_mean = x.mean(1, keepdim=True)  # (B,1,G)
    x_c = x - gene_mean
    depth = torch.log(counts.sum(2, keepdim=True) + eps)  # (B,n,1)
    depth = depth - depth.mean(1, keepdim=True)

    feats = torch.stack(
        [
            x_c,  # centered expression
            gene_mean.expand(B, n, G),  # gene level (dispersion trend)
            log_sf.unsqueeze(-1).expand(B, n, G),
            depth.expand(B, n, G),
            cond.unsqueeze(-1).expand(B, n, G),
            batch.unsqueeze(-1).expand(B, n, G),
            log_c,  # raw scale, keeps count noise info
        ],
        dim=-1,
    )  # (B,n,G,F)
    if covar is not None:
        feats = torch.cat([feats, covar.unsqueeze(2).expand(B, n, G, covar.size(-1))], dim=-1)
    # (B,n,G,E) single-cell per-(gene, sample) features
    if extra is not None:
        feats = torch.cat([feats, extra], dim=-1)
    return feats.permute(0, 2, 1, 3).contiguous()  # (B,G,n,F)


N_FEATS = 7


def gene_stats(counts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """(B,n,G) counts, (B,n) cond -> (B,G,S) classical statistics on normalized counts: naive log2FC, Welch t
    on log2 scale, pooled log-variance, log mean, moment NB log-dispersion, zero fraction, log n per group."""
    B, n, G = counts.shape
    eps = 1e-6
    logg = torch.log(counts + 0.5)
    ref = logg.mean(1, keepdim=True)
    ok = (counts > 0).all(1, keepdim=True)
    log_sf = torch.nan_to_num(
        torch.nanmedian((logg - ref).masked_fill(~ok, float("nan")), dim=2).values, nan=0.0
    )
    # (B,n,G)
    log_sf = log_sf - log_sf.mean(1, keepdim=True)
    norm = counts / torch.exp(log_sf).unsqueeze(-1)
    # (B,G)
    m = cond.unsqueeze(-1).expand(B, n, G) > 0.5
    n1 = m.float().sum(1).clamp(min=1)
    n0 = (~m).float().sum(1).clamp(min=1)
    mean1 = (norm * m).sum(1) / n1
    mean0 = (norm * (~m)).sum(1) / n0
    lfc = torch.log2(mean1 + 0.5) - torch.log2(mean0 + 0.5)
    x = torch.log2(norm + 1.0)
    xm1 = (x * m).sum(1) / n1
    xm0 = (x * (~m)).sum(1) / n0
    v1 = (((x - xm1.unsqueeze(1)) ** 2) * m).sum(1) / (n1 - 1).clamp(min=1)
    v0 = (((x - xm0.unsqueeze(1)) ** 2) * (~m)).sum(1) / (n0 - 1).clamp(min=1)
    t = (xm1 - xm0) / torch.sqrt(v1 / n1 + v0 / n0 + 0.05)
    t = t.clamp(-20, 20)
    logvar = torch.log(0.5 * (v1 + v0) + eps)
    mean_all = norm.mean(1)
    var_within = (v1 * (n1 - 1) + v0 * (n0 - 1)) / (n1 + n0 - 2).clamp(min=1)
    # MoM dispersion on the count scale, within groups
    cvar = (((norm - mean1.unsqueeze(1)) ** 2) * m).sum(1) / (n1 - 1).clamp(min=1) + (
        ((norm - mean0.unsqueeze(1)) ** 2) * (~m)
    ).sum(1) / (n0 - 1).clamp(min=1)
    disp = ((0.5 * cvar - mean_all) / (mean_all**2 + eps)).clamp(min=1e-3, max=100.0)
    logdisp = torch.log(disp)
    zero = (counts == 0).float().mean(1)
    # (B,G,S)
    return torch.stack(
        [lfc, t, logvar, torch.log1p(mean_all), logdisp, zero, torch.log(n1), torch.log(n0)], dim=-1
    )


N_STATS = 8


# attention blocks
class Block(nn.Module):
    """Pre-norm attention block; if kv is None it is self-attention."""

    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv=None):
        kvn = self.ln_q(q) if kv is None else self.ln_kv(kv)
        q = q + self.attn(self.ln_q(q), kvn, kvn, need_weights=False)[0]
        return q + self.ff(self.ln_ff(q))


class GeneEncoder(nn.Module):
    """Set transformer over samples -> one embedding per gene."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        n_in = N_FEATS + (cfg.covar_dim if cfg.use_covar else 0) + getattr(cfg, "sc_feats", 0)
        self.inp = nn.Sequential(nn.Linear(n_in, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d))
        self.blocks = nn.ModuleList([Block(cfg.d, cfg.n_heads, cfg.dropout) for _ in range(cfg.enc_layers)])
        self.pool_q = nn.Parameter(torch.randn(1, 1, cfg.d) / math.sqrt(cfg.d))
        self.pool = Block(cfg.d, cfg.n_heads, cfg.dropout)

    # (B,G,n,F)
    def forward(self, feats):
        B, G, n, _ = feats.shape
        h = self.inp(feats.view(B * G, n, -1))
        for blk in self.blocks:
            h = blk(h)
        q = self.pool_q.expand(B * G, 1, -1)
        hg = self.pool(q, h)[:, 0] + h.mean(1)  # attn-pool + mean-pool
        return hg.view(B, G, -1)


class LatentExchange(nn.Module):
    """Perceiver-style exchange: genes -> latent tokens -> genes."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, cfg.n_latents, cfg.d) / math.sqrt(cfg.d))
        self.read = nn.ModuleList([Block(cfg.d, cfg.n_heads, cfg.dropout) for _ in range(cfg.exch_layers)])
        self.self_ = nn.ModuleList([Block(cfg.d, cfg.n_heads, cfg.dropout) for _ in range(cfg.exch_layers)])
        self.write = nn.ModuleList([Block(cfg.d, cfg.n_heads, cfg.dropout) for _ in range(cfg.exch_layers)])

    # (B,G,d)
    def forward(self, hg):
        z = self.latents.expand(hg.size(0), -1, -1)
        for r, s, w in zip(self.read, self.self_, self.write):
            z = r(z, hg)  # latents attend to genes
            z = s(z)  # latents mix
            hg = w(hg, z)  # genes read back
        return hg, z


class BarHead(nn.Module):
    """Categorical over fixed LFC bins (TabPFN-style regression head)."""

    kind = "bar"

    def __init__(self, cfg: ModelConfig, in_dim: int = None):
        super().__init__()
        in_dim = in_dim or cfg.d
        self.out = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.n_bins))
        edges = torch.linspace(-cfg.lfc_range, cfg.lfc_range, cfg.n_bins + 1)
        self.register_buffer("edges", edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.tail_scale = float(cfg.tail_scale)
        # edge bins carry half-normal tails: their mean moves outward by s*sqrt(2/pi)
        if self.tail_scale > 0:
            centers = centers.clone()
            off = self.tail_scale * math.sqrt(2.0 / math.pi)
            centers[0] = edges[0] - off
            centers[-1] = edges[-1] + off
        self.register_buffer("centers", centers)

    def forward(self, hg):
        return self.out(hg)  # (B,G,n_bins)

    def nll(self, logits, target, weight=None):
        idx = torch.bucketize(target.clamp(self.edges[0] + 1e-4, self.edges[-1] - 1e-4), self.edges) - 1
        ce = F.cross_entropy(logits.flatten(0, 1), idx.flatten(), reduction="none").view_as(target)
        # TabPFN-style: beyond the edges the target is scored by a half-normal density inside the edge bin
        if self.tail_scale > 0:
            # -log density up to a constant
            over = (target.abs() - self.edges[-1]).clamp(min=0.0)
            ce = ce + 0.5 * (over / self.tail_scale) ** 2
        if weight is None:
            return ce.mean()
        return (ce * weight).sum() / weight.sum().clamp_min(1.0)  # weight 0 = gene left out of the loss

    def probs(self, logits):
        return logits.softmax(-1)

    def mean(self, logits):
        return (self.probs(logits) * self.centers).sum(-1)

    def prob_abs_gt(self, logits, tau):
        """P(|LFC| > tau) from the bin probabilities."""
        p = self.probs(logits)
        return (p * (self.centers.abs() > tau)).sum(-1)


# spike-and-slab heads
ZERO_TOL = 1e-6  # |target| below this is scored by the spike (unchanged training genes are exactly 0)
LOG_SIG_MIN, LOG_SIG_MAX = -7.0, 5.0
SIG_FLOOR = 1e-3  # sigma = SIG_FLOOR + exp(clamp(log_sigma)) >= 1e-3
LOC_MAX = 10.0  # slab locations clamped to [-10, 10] (log2)
DF_MAX = 200.0


def student_t_logpdf(y, loc, scale, df):
    """log density of a student-t(df, loc, scale) at y; all arguments broadcast."""
    z = (y - loc) / scale
    return (
        torch.lgamma((df + 1) / 2)
        - torch.lgamma(df / 2)
        - 0.5 * torch.log(df * math.pi)
        - torch.log(scale)
        - (df + 1) / 2 * torch.log1p(z * z / df)
    )


def student_t_cdf(x, df):
    """CDF of the standard student-t, numpy (scipy). torch.distributions.StudentT has no cdf, so mixture summaries run on CPU."""
    from scipy.stats import t as _t

    return _t.cdf(np.asarray(x, dtype=np.float64), np.asarray(df, dtype=np.float64))


class _MixtureHead(nn.Module):
    """MLP readout of mixture parameters; nll, mean and prob_abs_gt use mixture().

    Loss: -log pi0 for y == 0, else -log(1 - pi0) - log p_slab(y).
    mixture(out) -> dict(logit0, pi0, w (...,C), loc, scale, df) with sum_c w = 1."""

    kind = "mix"
    n_comp = 1

    def __init__(self, cfg: ModelConfig, in_dim: int, n_out: int):
        super().__init__()
        in_dim = in_dim or cfg.d
        self.df_fixed = float(cfg.slab_df)
        self.df_learn = bool(cfg.slab_df_learn)
        n_out = n_out + (1 if self.df_learn else 0)
        if cfg.head_hidden > 0:
            self.out = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, cfg.head_hidden),
                nn.GELU(),
                nn.Linear(cfg.head_hidden, n_out),
            )
        else:
            self.out = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, n_out))
        self.n_out = n_out

    def forward(self, hg):
        return self.out(hg)  # (B,G,n_out)

    def _scale(self, log_sig):
        return SIG_FLOOR + torch.exp(log_sig.clamp(LOG_SIG_MIN, LOG_SIG_MAX))

    def _df(self, out):
        if self.df_learn:
            return (2.0 + F.softplus(out[..., -1])).clamp(max=DF_MAX)
        return torch.full_like(out[..., 0], self.df_fixed)

    def mixture(self, out, slab_temperature: float = 1.0):
        raise NotImplementedError

    def nll(self, out, target, weight=None):
        """weight (B,G): 1 = scored, 0 = left out of the loss (genes without a label)."""
        mx = self.mixture(out)
        y = target.unsqueeze(-1)
        lp = torch.log(mx["w"].clamp_min(1e-12)) + student_t_logpdf(y, mx["loc"], mx["scale"], mx["df"])
        lp_slab = torch.logsumexp(lp, dim=-1)
        is0 = target.abs() < ZERO_TOL
        # -log pi0  |  -log(1-pi0) - log p_slab
        nll = torch.where(is0, F.softplus(-mx["logit0"]), F.softplus(mx["logit0"]) - lp_slab)
        if weight is None:
            return nll.mean()
        return (nll * weight).sum() / weight.sum().clamp_min(1.0)

    def mean(self, out):
        mx = self.mixture(out)
        return (1 - mx["pi0"]) * (mx["w"] * mx["loc"]).sum(-1)

    @torch.no_grad()
    def prob_abs_gt(self, out, tau):
        """P(|LFC| > tau) = (1 - pi0) * sum_c w_c [1 - F_c(tau) + F_c(-tau)], on CPU via scipy."""
        mx = self.mixture(out)
        pi0, w, loc, sc, df = (
            mx[k].detach().float().cpu().numpy().astype(np.float64)
            for k in ("pi0", "w", "loc", "scale", "df")
        )
        p = (1 - pi0) * (
            w * (1 - student_t_cdf((tau - loc) / sc, df) + student_t_cdf((-tau - loc) / sc, df))
        ).sum(-1)
        return torch.as_tensor(p, dtype=out.dtype, device=out.device)


class MixTHead(_MixtureHead):
    """Spike at 0 with probability sigmoid(logit0) plus one student-t slab. Outputs: [logit0, mu, log_sigma, (df_raw)]."""

    kind = "mix_t"
    n_comp = 1

    def __init__(self, cfg: ModelConfig, in_dim: int = None):
        super().__init__(cfg, in_dim, 3)

    def mixture(self, out, slab_temperature: float = 1.0):
        logit0 = out[..., 0]
        loc = out[..., 1].clamp(-LOC_MAX, LOC_MAX)
        scale = self._scale(out[..., 2]) * slab_temperature
        return {
            "logit0": logit0,
            "pi0": torch.sigmoid(logit0),
            "w": torch.ones_like(loc)[..., None],
            "loc": loc[..., None],
            "scale": scale[..., None],
            "df": self._df(out)[..., None],
        }


class MixUpDownHead(_MixtureHead):
    """Spike at 0 plus two student-t slabs with loc > 0 (up) and loc < 0 (down).
    Outputs: [logit0, logit_up_vs_down, mu_up_raw, log_sigma_up, mu_down_raw, log_sigma_down, (df_raw)]."""

    kind = "mix_updown"
    n_comp = 2

    def __init__(self, cfg: ModelConfig, in_dim: int = None):
        super().__init__(cfg, in_dim, 6)

    def mixture(self, out, slab_temperature: float = 1.0):
        logit0 = out[..., 0]
        w_up = torch.sigmoid(out[..., 1])
        loc_up = F.softplus(out[..., 2]).clamp(max=LOC_MAX)
        loc_dn = -F.softplus(out[..., 4]).clamp(max=LOC_MAX)
        sc_up = self._scale(out[..., 3]) * slab_temperature
        sc_dn = self._scale(out[..., 5]) * slab_temperature
        df = self._df(out)
        return {
            "logit0": logit0,
            "pi0": torch.sigmoid(logit0),
            "w": torch.stack([w_up, 1 - w_up], -1),
            "loc": torch.stack([loc_up, loc_dn], -1),
            "scale": torch.stack([sc_up, sc_dn], -1),
            "df": torch.stack([df, df], -1),
        }


class MixNearZeroHead(_MixtureHead):
    """Experimental head without a spike: a narrow near-zero component plus a broad slab,
        q(theta) = w0 * t_30(theta; m0, s0) + (1 - w0) * t_nu(theta; m, s),  |m0| <= NZ_LOC_MAX, s0 in [NZ_SCALE_MIN, NZ_SCALE_MAX].
    The width floor keeps the near-zero part from collapsing to a point mass. pi0 = 0, so rank 'any change' by
    max(P(theta > 0), P(theta < 0)). Outputs: [logit_w0, m0_raw, log_s0, m, log_s]."""

    kind = "mix_nz"
    n_comp = 2
    NZ_LOC_MAX, NZ_SCALE_MIN, NZ_SCALE_MAX, NZ_DF = 0.3, 0.05, 0.25, 30.0

    def __init__(self, cfg: ModelConfig, in_dim: int = None):
        super().__init__(cfg, in_dim, 5)

    def mixture(self, out, slab_temperature: float = 1.0):
        w0 = torch.sigmoid(out[..., 0])
        m0 = self.NZ_LOC_MAX * torch.tanh(out[..., 1])
        s0 = self.NZ_SCALE_MIN + (self.NZ_SCALE_MAX - self.NZ_SCALE_MIN) * torch.sigmoid(out[..., 2])
        m = out[..., 3].clamp(-LOC_MAX, LOC_MAX)
        s = self._scale(out[..., 4]) * slab_temperature
        df_slab = self._df(out)
        df0 = torch.full_like(df_slab, self.NZ_DF)
        # no exact spike: pi0 = sigmoid(-30) ~ 1e-13
        logit0 = torch.full_like(w0, -30.0)
        return {
            "logit0": logit0,
            "pi0": torch.zeros_like(w0),
            "w": torch.stack([w0, 1 - w0], -1),
            "loc": torch.stack([m0, m], -1),
            "scale": torch.stack([s0, s], -1),
            "df": torch.stack([df0, df_slab], -1),
        }

    def nll(self, out, target, weight=None):
        """-log q(y); zero labels are scored by the density at 0."""
        mx = self.mixture(out)
        y = target.unsqueeze(-1)
        lp = torch.log(mx["w"].clamp_min(1e-12)) + student_t_logpdf(y, mx["loc"], mx["scale"], mx["df"])
        nll = -torch.logsumexp(lp, dim=-1)
        if weight is None:
            return nll.mean()
        return (nll * weight).sum() / weight.sum().clamp_min(1.0)


HEADS = {"bar": BarHead, "mix_t": MixTHead, "mix_updown": MixUpDownHead, "mix_nz": MixNearZeroHead}


def make_head(cfg: ModelConfig, in_dim: int = None):
    if cfg.head not in HEADS:
        raise ValueError(f"unknown head {cfg.head!r}; choose from {sorted(HEADS)}")
    return HEADS[cfg.head](cfg, in_dim)


class PRESET(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = GeneEncoder(cfg)
        self.exch = LatentExchange(cfg)
        if cfg.use_stats:
            self.stats_norm = nn.LayerNorm(N_STATS)
            self.stats_proj = nn.Sequential(nn.Linear(N_STATS, cfg.d), nn.GELU(), nn.Linear(cfg.d, cfg.d))
        head_in = cfg.d * (3 if cfg.evidence_head else 1)
        self.head = make_head(cfg, head_in)

    def _embed(self, counts, cond, batch, gene_chunk=None, covar=None, extra=None):
        if self.cfg.use_covar:
            if covar is None:
                covar = torch.zeros(
                    counts.size(0),
                    counts.size(1),
                    self.cfg.covar_dim,
                    device=counts.device,
                    dtype=counts.dtype,
                )
        else:
            covar = None
        E = getattr(self.cfg, "sc_feats", 0)
        if E and extra is None:
            extra = torch.zeros(*counts.shape, E, device=counts.device, dtype=counts.dtype)
        if not E:
            extra = None
        feats = featurize(counts, cond, batch, covar, extra)
        if gene_chunk:
            hg = torch.cat(
                [self.enc(feats[:, i : i + gene_chunk]) for i in range(0, feats.size(1), gene_chunk)], 1
            )
        else:
            hg = self.enc(feats)
        se = None
        if self.cfg.use_stats:
            se = self.stats_proj(self.stats_norm(gene_stats(counts, cond)))
            hg = hg + se
        return hg, se

    def _readout(self, hg_pre, hg_post, se):
        if not self.cfg.evidence_head:
            return self.head(hg_post)
        if se is None:
            se = torch.zeros_like(hg_pre)
        return self.head(torch.cat([hg_post, hg_pre, se], -1))

    def forward(self, counts, cond, batch, covar=None, extra=None):
        hg_pre, se = self._embed(counts, cond, batch, covar=covar, extra=extra)
        hg, latents = self.exch(hg_pre)
        return self._readout(hg_pre, hg, se), latents

    @torch.no_grad()
    def predict(self, counts, cond, batch, tau=0.5, gene_chunk=4096, covar=None, extra=None):
        """Inference on all genes: encode in chunks, exchange once over all genes. "logits" is the raw head output
        (bin logits or mixture parameters) that posterior.summarize reads."""
        hg_pre, se = self._embed(counts, cond, batch, gene_chunk, covar=covar, extra=extra)
        hg, _ = self.exch(hg_pre)
        logits = self._readout(hg_pre, hg, se)
        return {
            "lfc_mean": self.head.mean(logits),
            "p_de": self.head.prob_abs_gt(logits, tau),
            "logits": logits,
            "head": self.head.kind,
        }
