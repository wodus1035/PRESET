"""Structural causal model (SCM) prior that generates the simulated training datasets.

Graph of one dataset:
    condition c_i, batch b_i, donor latent z_i
        -> TF activity t_ki = A_c[k] c_i + A_b[k] b_i + A_z[k] z_i + noise
            -> log mu_gi = a_g + sum_k W_gk t_ki + beta_g c_i + gamma_g b_i
                -> count_gi ~ NB(mu_gi, alpha_g), allocated from a read depth that does not depend on the condition

The label of gene g is the total effect of the condition on log2 mu at fixed batch and donor,
    lfc_g = (beta_g + (W A_c)_g) / ln 2.

All datasets of a batch share the number of samples and genes; every other quantity is drawn per dataset.
"""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

LN2 = math.log(2.0)


@dataclass
class PriorConfig:
    G: int = 2000
    K: int = 64  # TFs
    dz: int = 2  # donor latent dimensions
    n_per_group: Tuple[int, int] = (2, 8)
    p_null: float = 0.25  # datasets without any condition effect

    # baseline expression (natural log)
    a_mu: Tuple[float, float] = (4.0, 6.5)
    a_sd: Tuple[float, float] = (1.0, 2.2)
    a_min: float = 1.5

    # TF layer
    tf_per_gene_probs: Tuple[float, ...] = (0.40, 0.35, 0.20, 0.05)  # genes regulated by 0, 1, 2, 3 TFs
    tf_popularity_conc: float = 0.3
    w_logsd: float = 0.5
    p_repress: float = 0.3
    frac_responsive_tf: Tuple[float, float] = (0.05, 0.25)
    tf_effect_sd: Tuple[float, float] = (0.3, 1.5)
    tf_noise_sd: Tuple[float, float] = (0.05, 0.3)
    z_effect_sd: Tuple[float, float] = (0.15, 0.5)

    # direct condition effects
    de_frac: Tuple[float, float] = (0.01, 0.30)
    beta_sd: Tuple[float, float] = (0.3, 1.5)

    # batch
    p_has_batch: float = 0.5
    batch_confound: Tuple[float, float] = (0.0, 0.8)
    gamma_frac: Tuple[float, float] = (0.05, 0.30)
    gamma_sd: Tuple[float, float] = (0.2, 0.8)
    batch_tf_frac: Tuple[float, float] = (0.02, 0.15)
    batch_tf_sd: Tuple[float, float] = (0.2, 0.8)

    # observation, alpha_g = a0 * (baseline_mu / 100)^(-b) + a1 / baseline_mu
    disp_a0: Tuple[float, float] = (0.005, 0.4)
    disp_b: Tuple[float, float] = (0.0, 0.4)
    disp_a1: Tuple[float, float] = (0.1, 2.0)
    disp_logsd: Tuple[float, float] = (0.2, 0.9)
    sf_logsd: Tuple[float, float] = (0.1, 0.5)
    min_null_frac: float = (
        0.30  # share of genes with exactly zero total effect (identifies the compositional offset)
    )

    p_outlier: Tuple[float, float] = (0.0, 0.002)
    outlier_mult: Tuple[float, float] = (4.0, 12.0)

    # condition effect along the donor-variation axis
    p_aligned: float = 0.0
    aligned_effect_sd: Tuple[float, float] = (0.5, 2.5)
    aligned_axis_boost: Tuple[float, float] = (2.0, 4.0)

    # observed covariates and paired designs
    p_covar: float = 0.0  # donor axis 0 observed as a continuous covariate, imbalanced between conditions
    covar_cont_noise: Tuple[float, float] = (0.0, 0.5)
    covar_confound: Tuple[float, float] = (0.0, 1.0)
    p_paired: float = 0.0  # the same donors in both conditions, donor identity observed
    covar_code_dim: int = 4

    # numerical guards
    alpha_min: float = 1e-4
    alpha_max: float = 20.0
    log_mu_min: float = -6.0
    log_mu_max: float = 14.0


def _u(lo_hi, shape, device):
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(shape, device=device)


def _lu(lo_hi, shape, device):
    """Log-uniform draw."""
    lo, hi = lo_hi
    return torch.exp(_u((math.log(lo), math.log(hi)), shape, device))


def _sample_baseline(cfg: PriorConfig, B: int, device):
    """Baseline expression with a dataset-specific center and spread, redrawn below a_min (genes that pass a filter)."""
    center = _u(cfg.a_mu, (B, 1), device)
    spread = _u(cfg.a_sd, (B, 1), device)
    a = center + spread * torch.randn(B, cfg.G, device=device)

    for _ in range(8):
        low = a < cfg.a_min
        if not low.any():
            break
        redraw = center + spread * torch.randn(B, cfg.G, device=device)
        a = torch.where(low, redraw, a)

    low = a < cfg.a_min
    a = torch.where(low, cfg.a_min + 0.25 * torch.randn_like(a).abs(), a)
    return a


def _sparse_loadings(cfg: PriorConfig, B: int, device) -> torch.Tensor:
    """W (B, G, K). Each gene loads on 0-3 distinct TFs, and TF popularity is skewed."""
    G, K = cfg.G, cfg.K
    probs = torch.tensor(cfg.tf_per_gene_probs, dtype=torch.float32, device=device)
    n_tf = torch.multinomial(probs.expand(B * G, -1), 1).view(B, G)

    pop = torch.distributions.Dirichlet(torch.full((K,), cfg.tf_popularity_conc, device=device)).sample((B,))
    idx = torch.multinomial(pop.repeat_interleave(G, 0), 3, replacement=False).view(B, G, 3)

    keep = torch.arange(3, device=device).view(1, 1, 3) < n_tf.unsqueeze(-1)
    mag = torch.exp(cfg.w_logsd * torch.randn(B, G, 3, device=device))
    sign = torch.where(torch.rand(B, G, 3, device=device) < cfg.p_repress, -1.0, 1.0)

    W = torch.zeros(B, G, K, device=device)
    W.scatter_add_(2, idx, mag * sign * keep)
    return W


@torch.no_grad()
def generate_batch(cfg: PriorConfig, B: int, device="cpu") -> Dict[str, torch.Tensor]:
    G, K = cfg.G, cfg.K

    npg = int(torch.randint(cfg.n_per_group[0], cfg.n_per_group[1] + 1, (1,)).item())
    n = 2 * npg

    # Samples: condition, batch and donor latent
    c = torch.cat([torch.zeros(npg), torch.ones(npg)]).to(device).expand(B, n)

    has_batch = (torch.rand(B, device=device) < cfg.p_has_batch).float()
    confound = _u(cfg.batch_confound, (B, 1), device)
    p_b = 0.5 + confound * (c - 0.5)  # P(b=1 | c=0) = 0.5 - confound/2, P(b=1 | c=1) = 0.5 + confound/2
    b = (torch.rand(B, n, device=device) < p_b).float()

    # random label permutation, so that "batch = 1" carries no meaning
    flip = torch.rand(B, 1, device=device) < 0.5
    b = torch.where(flip, 1.0 - b, b)
    b = b * has_batch.unsqueeze(1)
    z = torch.randn(B, n, cfg.dz, device=device)

    # Observed covariates and paired designs
    if cfg.p_paired > 0:
        is_paired = torch.rand(B, device=device) < cfg.p_paired
    else:
        is_paired = torch.zeros(B, dtype=torch.bool, device=device)
    if is_paired.any():
        z_pair = torch.cat([z[:, :npg], z[:, :npg]], dim=1)
        z = torch.where(is_paired.view(B, 1, 1), z_pair, z)

    if cfg.p_covar > 0:
        has_covar = (torch.rand(B, device=device) < cfg.p_covar).float()
        conf = (
            _u(cfg.covar_confound, (B, 1), device)
            * has_covar.unsqueeze(1)
            * (~is_paired).float().unsqueeze(1)
        )
        z0 = z[:, :, :1] + (conf * (c - 0.5)).unsqueeze(-1)  # donor axis 0 imbalanced between conditions
        z = torch.cat([z0, z[:, :, 1:]], dim=2)
    else:
        has_covar = torch.zeros(B, device=device)

    x_cont = z[:, :, 0] + _u(cfg.covar_cont_noise, (B, 1), device) * torch.randn(B, n, device=device)
    x_cont = (x_cont - x_cont.mean(1, keepdim=True)) / (x_cont.std(1, keepdim=True) + 1e-6)
    x_cont = x_cont * has_covar.unsqueeze(1)

    codes = torch.randn(B, npg, cfg.covar_code_dim, device=device)  # random code per donor
    donor = torch.arange(n, device=device) % npg
    code_feat = codes[:, donor] * is_paired.float().view(B, 1, 1)
    covar = torch.cat(
        [
            x_cont.unsqueeze(-1),
            has_covar.view(B, 1, 1).expand(B, n, 1),
            code_feat,
            is_paired.float().view(B, 1, 1).expand(B, n, 1),
        ],
        dim=-1,
    )

    # Datasets without any condition effect
    is_null = (torch.rand(B, device=device) < cfg.p_null).float()
    signal = (1.0 - is_null).view(B, 1)

    # TF layer
    W = _sparse_loadings(cfg, B, device)
    resp = (torch.rand(B, K, device=device) < _u(cfg.frac_responsive_tf, (B, 1), device)).float()
    A_c = resp * torch.randn(B, K, device=device) * _u(cfg.tf_effect_sd, (B, 1), device) * signal

    batch_resp = (torch.rand(B, K, device=device) < _u(cfg.batch_tf_frac, (B, 1), device)).float()
    A_b = batch_resp * torch.randn(B, K, device=device) * _u(cfg.batch_tf_sd, (B, 1), device)
    A_b = A_b * has_batch.unsqueeze(1)

    A_z = torch.randn(B, K, cfg.dz, device=device) * _u(cfg.z_effect_sd, (B, 1, 1), device)

    # aligned datasets: donor axis 0 dominates the within-group variation and the condition effect follows it
    if cfg.p_aligned > 0:
        is_aligned = torch.rand(B, device=device) < cfg.p_aligned
        boost = torch.where(
            is_aligned, _u(cfg.aligned_axis_boost, (B,), device), torch.ones(B, device=device)
        )
        A_z = torch.cat([A_z[:, :, :1] * boost.view(B, 1, 1), A_z[:, :, 1:]], dim=2)

        axis = A_z[:, :, 0]
        axis = axis / (axis.std(1, keepdim=True) + 1e-6)
        direction = torch.randint(0, 2, (B, 1), device=device).float() * 2 - 1
        A_c_al = resp * axis * direction * _u(cfg.aligned_effect_sd, (B, 1), device) * signal
        A_c = torch.where(is_aligned.unsqueeze(1), A_c_al, A_c)

    t_noise = _u(cfg.tf_noise_sd, (B, 1, 1), device) * torch.randn(B, n, K, device=device)
    t = (
        c.unsqueeze(-1) * A_c.unsqueeze(1)
        + b.unsqueeze(-1) * A_b.unsqueeze(1)
        + torch.einsum("bnd,bkd->bnk", z, A_z)
        + t_noise
    )
    module_term = torch.einsum("bnk,bgk->bng", t, W)
    module_lfc = torch.einsum("bgk,bk->bg", W, A_c)

    # Genes: baseline, direct condition effects and batch effects
    a = _sample_baseline(cfg, B, device)

    de_mask = (torch.rand(B, G, device=device) < _u(cfg.de_frac, (B, 1), device)).float() * signal
    beta = de_mask * torch.randn(B, G, device=device) * _u(cfg.beta_sd, (B, 1), device)

    gamma_mask = (torch.rand(B, G, device=device) < _u(cfg.gamma_frac, (B, 1), device)).float()
    gamma = gamma_mask * torch.randn(B, G, device=device) * _u(cfg.gamma_sd, (B, 1), device)
    gamma = gamma * has_batch.unsqueeze(1)

    log_mu_raw = (
        a.unsqueeze(1)
        + module_term
        + c.unsqueeze(-1) * beta.unsqueeze(1)
        + b.unsqueeze(-1) * gamma.unsqueeze(1)
    )
    log_mu = log_mu_raw.clamp(cfg.log_mu_min, cfg.log_mu_max)
    mu = log_mu.exp()

    # Null floor: the genes with the smallest effects become exactly null on both paths (beta and modules).
    # Only the condition part of the module term is removed, so these genes keep their donor and batch co-variation.
    if cfg.min_null_frac > 0:
        tot = beta + module_lfc
        need = int(round(cfg.min_null_frac * G)) - int((tot == 0).sum(1).min().item())
        if need > 0:
            score = tot.abs() + (tot == 0).float() * 1e9  # genes that are already null stay null
            idx = score.argsort(dim=1)[:, :need]
            m = torch.zeros(B, G, device=device, dtype=torch.bool).scatter_(1, idx, True)

            beta = beta.masked_fill(m, 0.0)
            drop = (module_lfc * m.float()).unsqueeze(1) * c.unsqueeze(-1)
            module_lfc = module_lfc.masked_fill(m, 0.0)
            log_mu_raw = (
                a.unsqueeze(1)
                + module_term
                - drop
                + c.unsqueeze(-1) * beta.unsqueeze(1)
                + b.unsqueeze(-1) * gamma.unsqueeze(1)
            )
            log_mu = log_mu_raw.clamp(cfg.log_mu_min, cfg.log_mu_max)
            mu = log_mu.exp()

    lfc_true = (beta + module_lfc) / LN2

    # Dispersion, from the baseline abundance rather than the condition-shifted one
    mu_bar = torch.exp(a).clamp_min(0.1)
    disp_a0 = _lu(cfg.disp_a0, (B, 1), device)
    disp_b = _u(cfg.disp_b, (B, 1), device)
    disp_a1 = _u(cfg.disp_a1, (B, 1), device)
    trend = disp_a0 * (mu_bar / 100.0).pow(-disp_b) + disp_a1 / mu_bar
    alpha = trend * torch.exp(_u(cfg.disp_logsd, (B, 1), device) * torch.randn(B, G, device=device))
    alpha = alpha.clamp(cfg.alpha_min, cfg.alpha_max)

    # Size factors, geometric mean 1 per dataset
    log_sf = _u(cfg.sf_logsd, (B, 1), device) * torch.randn(B, n, device=device)
    log_sf = log_sf - log_sf.mean(1, keepdim=True)
    sf = torch.exp(log_sf)

    # Fixed read depth: mu becomes a within-sample composition times a depth that does not depend on the condition,
    # so genes that do not change lose reads when others gain, as on a sequencer
    tot_mu = mu.sum(-1, keepdim=True).clamp_min(1e-8)
    D = torch.exp(torch.log(tot_mu).mean(1, keepdim=True))
    rate_mu = mu / tot_mu * D * sf.unsqueeze(-1)

    # Negative binomial counts as a Gamma-Poisson mixture, Var(Y) = mu + alpha mu^2
    shape = (1.0 / alpha).unsqueeze(1).expand_as(rate_mu)
    lam = torch.distributions.Gamma(shape, shape / rate_mu.clamp_min(1e-8)).sample()
    counts = torch.poisson(lam.clamp(0.0, 1e7))

    # Outlier counts, rounded so that a fractional part does not mark them
    out_mask = torch.rand_like(counts) < _u(cfg.p_outlier, (B, 1, 1), device)
    out_mult = _u(cfg.outlier_mult, (B, 1, 1), device)
    counts = torch.where(out_mask, (counts * out_mult).round(), counts)

    # Datasets in which condition and batch are completely aliased
    same = (b == c).all(1)
    opposite = (b == (1.0 - c)).all(1)
    exact_confound = (same | opposite) & has_batch.bool()

    return {
        "counts": counts,
        "cond": c,
        "batch": b,
        "covar": covar,
        "lfc_true": lfc_true,
        "is_null": is_null,
        "beta_true": beta,
        "module_lfc_true": module_lfc / LN2,
        "exact_confound": exact_confound.float(),
    }
