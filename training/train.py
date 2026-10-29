"""Train PRESET on simulated datasets, real GTEx backgrounds with planted effects and, in the second stage, real cohorts.

The defaults are the first stage of the released model. stage1.sh and stage2.sh give the commands of both stages.
    python training/train.py --output_dir outputs/stage1
"""

import argparse
import math
import os
import random
import time

import numpy as np
import torch

from preset.model import ModelConfig, PRESET
from prior import PriorConfig, generate_batch

# Dataset-level scenarios drawn on top of the base prior (Supplementary Table S1). Each overrides some PriorConfig ranges.
#   TISSUE  two different tissues, most genes change
#   TECH    technical replicates, very low within-group noise
#   BIO     biological replicates from different donors, overdispersion that does not decay with expression
#   SPARSE  a handful of changed genes and no module response
#   CLIN    clinical case-control, moderate effects on 5-25% of genes with donor-level noise
REGIME_TISSUE = dict(
    de_frac=(0.5, 0.70), beta_sd=(1.0, 3.0), frac_responsive_tf=(0.3, 0.9), tf_effect_sd=(0.8, 2.5)
)
REGIME_TECH = dict(
    z_effect_sd=(0.02, 0.08),
    tf_noise_sd=(0.01, 0.04),
    sf_logsd=(0.03, 0.1),
    disp_a0=(0.002, 0.03),
    disp_a1=(0.02, 0.2),
    disp_logsd=(0.1, 0.3),
)
REGIME_BIO = dict(
    disp_a0=(0.2, 1.0),
    disp_b=(0.0, 0.15),
    disp_a1=(0.3, 3.0),
    disp_logsd=(0.5, 1.2),
    z_effect_sd=(0.4, 1.2),
    tf_noise_sd=(0.2, 0.6),
)
REGIME_SPARSE = dict(de_frac=(0.0005, 0.01), frac_responsive_tf=(0.0, 0.0), beta_sd=(1.0, 4.0))
REGIME_CLIN = dict(
    de_frac=(0.05, 0.25),
    beta_sd=(0.4, 1.0),
    frac_responsive_tf=(0.03, 0.15),
    tf_effect_sd=(0.3, 0.8),
    disp_a0=(0.15, 0.5),
    disp_b=(0.0, 0.15),
    disp_a1=(0.3, 2.0),
    disp_logsd=(0.4, 0.9),
    z_effect_sd=(0.4, 1.0),
    tf_noise_sd=(0.2, 0.5),
)

MIN_NULL_FRAC = 0.30  # share of genes that stay exactly unchanged
ONOFF_FRAC = (0.05, 0.30)  # fraction of genes switched off in one condition
ONOFF_SHIFT = (4.0, 12.0)  # log2 decrease of the switched-off condition


def curriculum(cfg, step, warm):
    """Widen batch confounding, outliers and donor variation from zero to the full prior over `warm` steps."""
    f = min(1.0, step / max(1, warm))
    out = PriorConfig(**vars(cfg))

    out.p_has_batch = cfg.p_has_batch * f
    out.batch_confound = (0.0, cfg.batch_confound[1] * f)
    out.p_outlier = (0.0, cfg.p_outlier[1] * f)

    z0, z1 = cfg.z_effect_sd
    out.z_effect_sd = (z0, z0 + (z1 - z0) * f)
    return out


def apply_regimes(cfg, rng, p_tissue, p_tech, p_bio, p_sparse, p_clin):
    """Draw the scenarios of one step. CLIN excludes the others, SPARSE and TISSUE exclude each other (SPARSE drawn first),
    and BIO and TECH exclude each other (BIO drawn first)."""
    tag = []
    if p_clin > 0 and rng.random() < p_clin:
        for k, v in REGIME_CLIN.items():
            setattr(cfg, k, v)
        cfg.p_null = 0.0
        return cfg, "clin"

    if p_sparse > 0 and rng.random() < p_sparse:
        for k, v in REGIME_SPARSE.items():
            setattr(cfg, k, v)
        cfg.p_null = 0.0
        tag.append("sparse")
    elif p_tissue > 0 and rng.random() < p_tissue:
        for k, v in REGIME_TISSUE.items():
            setattr(cfg, k, v)
        tag.append("tissue")

    if p_bio > 0 and rng.random() < p_bio:
        for k, v in REGIME_BIO.items():
            setattr(cfg, k, v)
        tag.append("bio")
    elif p_tech > 0 and rng.random() < p_tech:
        for k, v in REGIME_TECH.items():
            setattr(cfg, k, v)
        tag.append("tech")

    return cfg, "+".join(tag) or "base"


@torch.no_grad()
def apply_onoff(d):
    """Switch a random subset of the null genes (almost) off in one random condition, by binomial thinning, and update
    the labels. Datasets without condition effects are left untouched."""
    counts, cond, lfc = d["counts"], d["cond"], d["lfc_true"]
    B, n, G = counts.shape
    dev = counts.device

    # fraction of genes, capped so that MIN_NULL_FRAC of the genes stay null
    frac = torch.empty(B, device=dev).uniform_(*ONOFF_FRAC)
    null_frac = (lfc == 0).float().mean(1)
    frac = torch.minimum(frac, (null_frac - MIN_NULL_FRAC).clamp_min(0.0))
    sel = (torch.rand(B, G, device=dev) < frac.unsqueeze(1)) & (d["is_null"].view(B, 1) < 0.5)
    sel = sel & (lfc == 0)

    # which condition is off, and by how much
    off_in_cond1 = torch.rand(B, G, device=dev) < 0.5
    shift = torch.empty(B, G, device=dev).uniform_(*ONOFF_SHIFT) * sel
    is_off = (cond.unsqueeze(-1) > 0.5) == off_in_cond1.unsqueeze(1)
    factor = torch.where(is_off & sel.unsqueeze(1), torch.exp2(-shift.unsqueeze(1)), torch.ones_like(counts))

    out = dict(d)
    out["counts"] = torch.binomial(counts.clamp_min(0.0), factor.clamp(0.0, 1.0))
    out["lfc_true"] = lfc + torch.where(off_in_cond1, -shift, shift)
    return out


@torch.no_grad()
def apply_lowtail(d, rng, frac_range, mean_range):
    """Thin a random subset of genes to a log-uniform target mean count in mean_range, with the same factor for every
    sample, so the labels are unchanged. Real matrices carry many genes with mean counts of 1-30, which the prior and the
    GTEx bank rarely produce. rng: random.Random for the per-dataset fraction."""
    counts = d["counts"]
    B, n, G = counts.shape
    dev = counts.device

    frac = torch.tensor([rng.uniform(*frac_range) for _ in range(B)], device=dev)
    sel = torch.rand(B, G, device=dev) < frac.unsqueeze(1)
    lo, hi = math.log(mean_range[0]), math.log(mean_range[1])
    target = torch.exp(lo + (hi - lo) * torch.rand(B, G, device=dev))

    # genes already below their target are left alone
    cur = counts.mean(1)
    factor = torch.where(sel & (cur > target), target / cur.clamp_min(1e-8), torch.ones_like(cur))
    changed = factor < 1
    new = torch.poisson(counts * factor.unsqueeze(1))

    out = dict(d)
    out["counts"] = torch.where(changed.unsqueeze(1).expand_as(counts), new, counts)
    return out


def sample_G(rng, G_min, G_max):
    """Log-uniform number of genes in [G_min, G_max]."""
    if G_min == G_max:
        return G_min
    g = int(round(math.exp(rng.uniform(math.log(G_min), math.log(G_max)))))
    return min(max(g, G_min), G_max)


def effective_B(B, G, tokens_per_step):
    """Datasets per step, so that datasets times genes stays under tokens_per_step."""
    if tokens_per_step <= 0:
        return B
    return max(1, min(B, tokens_per_step // max(1, G)))


def make_scheduler(opt, total_steps, warmup_steps):
    """Linear warm-up followed by cosine decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def checkpoint_dict(step, model, opt, sched, mcfg, pcfg, args, rep=None):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "mcfg": vars(mcfg),
        "pcfg": vars(pcfg),
        "args": vars(args),
    }
    if rep is not None:
        ckpt["eval"] = rep
    return ckpt


@torch.no_grad()
def evaluate(model, cfg, device, n_batches, B, tau):
    """Loss and calls on simulated validation datasets from the full prior. The first stage keeps the checkpoint with
    the lowest loss."""
    model.eval()
    nll_sum = 0.0
    mae_sum = 0.0
    calls = [0, 0, 0]  # called, false calls, true changes

    for _ in range(n_batches):
        d = generate_batch(cfg, B, device)
        out, _ = model(d["counts"], d["cond"], d["batch"], covar=d["covar"])
        target = d["lfc_true"]

        nll_sum += model.head.nll(out, target).item()
        mae_sum += (model.head.mean(out) - target).abs().mean().item()

        sel = model.head.prob_abs_gt(out, tau) > 0.9
        truth = target.abs() > tau
        calls[0] += sel.sum().item()
        calls[1] += (sel & ~truth).sum().item()
        calls[2] += truth.sum().item()

    model.train()
    return {
        "nll": nll_sum / n_batches,
        "mae": mae_sum / n_batches,
        "FDP@0.9": calls[1] / max(1, calls[0]),
        "power@0.9": (calls[0] - calls[1]) / max(1, calls[2]),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()

    # run
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--ckpt", default="preset.pt", help="also writes <name>_best.pt, the lowest validation loss"
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--gpu", type=int, default=0, help="-1 for the CPU")
    ap.add_argument("--init_from", default=None, help="checkpoint whose weights start the run (second stage)")
    ap.add_argument("--wandb_project", default=None, help="log to Weights & Biases under this project")
    ap.add_argument("--wandb_name", default=None)

    # optimization
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_warmup", type=int, default=1000)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--B", type=int, default=4, help="largest number of datasets per step")
    ap.add_argument("--tokens_per_step", type=int, default=16000, help="datasets x genes per step")
    ap.add_argument(
        "--G_min", type=int, default=2000, help="genes per dataset, log-uniform in [G_min, G_max]"
    )
    ap.add_argument("--G_max", type=int, default=8000)
    ap.add_argument("--n_max", type=int, default=16, help="largest number of samples per group (smallest 2)")
    ap.add_argument("--warm", type=int, default=1000, help="steps over which the nuisance structure widens")
    ap.add_argument(
        "--warm_regime", type=int, default=15000, help="steps over which the probabilities below ramp in"
    )

    # simulated datasets
    ap.add_argument("--K", type=int, default=64, help="TFs of the prior")
    ap.add_argument("--p_tissue", type=float, default=0.3)
    ap.add_argument("--p_tech", type=float, default=0.25)
    ap.add_argument("--p_bio", type=float, default=0.4)
    ap.add_argument("--p_sparse", type=float, default=0.3)
    ap.add_argument("--p_clin", type=float, default=0.15)
    ap.add_argument("--p_onoff", type=float, default=0.3)
    ap.add_argument("--p_aligned", type=float, default=0.3)
    ap.add_argument("--p_covar", type=float, default=0.3)
    ap.add_argument("--p_paired", type=float, default=0.15)
    ap.add_argument("--p_lowtail", type=float, default=0.5)
    ap.add_argument("--lowtail_frac", nargs=2, type=float, default=[0.1, 0.3])
    ap.add_argument("--lowtail_mean", nargs=2, type=float, default=[1.0, 30.0])
    ap.add_argument("--p_thin", type=float, default=0.2, help="binomial thinning of every count of a batch")
    ap.add_argument("--thin_range", nargs=2, type=float, default=[0.01, 0.3])

    # real GTEx backgrounds
    ap.add_argument("--p_real", type=float, default=0.5)
    ap.add_argument("--real_bank", default=os.path.join(here, "data", "gtex_real_bank_v3.npz"))
    ap.add_argument(
        "--donor_split",
        default=os.path.join(os.path.dirname(here), "evaluation", "data_cache", "gtex_donor_split_v3.json"),
    )
    ap.add_argument("--real_p_huge", type=float, default=0.35)
    ap.add_argument("--real_p_aligned", type=float, default=0.75)
    ap.add_argument("--real_aligned_sd", nargs=2, type=float, default=[0.5, 3.0])
    ap.add_argument("--real_aligned_frac", nargs=2, type=float, default=[0.02, 0.3])
    ap.add_argument(
        "--real_bank2",
        default=None,
        help="second bank with the same planting and no donor split (build_mouse_bank.py)",
    )
    ap.add_argument(
        "--real_bank2_frac",
        type=float,
        default=0.0,
        help="share of the real-background batches drawn from it",
    )

    # real cohorts (second stage)
    ap.add_argument("--p_cons", type=float, default=0.0)
    ap.add_argument("--cons_dir", default=os.path.join(here, "data", "consensus_bank"))
    ap.add_argument("--cons_label", choices=["consensus", "observed"], default="consensus")
    ap.add_argument(
        "--cons_exclude", nargs="*", default=[], help="leave out cohorts whose names contain these strings"
    )

    # validation
    ap.add_argument("--G", type=int, default=2000, help="genes of the validation datasets")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--eval_batches", type=int, default=16)
    ap.add_argument("--eval_B", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("device:", device, flush=True)

    # Prior. Validation uses the full prior without the per-step scenarios.
    pcfg = PriorConfig(G=args.G, K=args.K)
    pcfg.n_per_group = (pcfg.n_per_group[0], args.n_max)
    eval_cfg = PriorConfig(**vars(pcfg))

    # Model
    mcfg = ModelConfig(
        use_stats=True,
        evidence_head=True,
        use_covar=True,
        head="mix_t",
        head_hidden=128,
        slab_df=4.0,
    )
    model = PRESET(mcfg).to(device)
    if args.init_from:
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init["model"])
        print(f"initialized from {args.init_from} (step {init.get('step', '?')})", flush=True)

    run = None
    if args.wandb_project:
        import wandb

        config = {**vars(args), "model": vars(mcfg), "prior": vars(pcfg)}
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=config)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = make_scheduler(opt, total_steps=args.steps, warmup_steps=args.lr_warmup)

    os.makedirs(args.output_dir, exist_ok=True)
    root, ext = os.path.splitext(args.ckpt)
    latest_path = os.path.join(args.output_dir, root + (ext or ".pt"))
    best_path = os.path.join(args.output_dir, root + "_best" + (ext or ".pt"))
    best_nll = float("inf")

    # Random streams of the scenario draws, the gene numbers, the low-count tail and the real tasks
    regime_rng = random.Random(args.seed)
    g_rng = random.Random(args.seed + 1)
    lowtail_rng = random.Random(args.seed + 2)
    real_np_rng = np.random.default_rng(args.seed)

    real_bank = None
    if args.p_real > 0:
        from real_background import RealBank

        real_bank = RealBank(
            path=args.real_bank,
            donor_split=args.donor_split,
            p_huge=args.real_p_huge,
            p_aligned=args.real_p_aligned,
            aligned_sd=tuple(args.real_aligned_sd),
            aligned_frac=tuple(args.real_aligned_frac),
        )
        print(f"GTEx bank: {real_bank.path}, tissues {real_bank.tissues}", flush=True)

    real_bank2 = None
    if real_bank is not None and args.real_bank2 and args.real_bank2_frac > 0:
        real_bank2 = RealBank(
            path=args.real_bank2,
            donor_split=None,
            p_huge=args.real_p_huge,
            p_aligned=args.real_p_aligned,
            aligned_sd=tuple(args.real_aligned_sd),
            aligned_frac=tuple(args.real_aligned_frac),
        )
        print(f"second bank: {real_bank2.path}, tissues {real_bank2.tissues}", flush=True)

    cons_bank = None
    if args.p_cons > 0:
        from consensus_bank import ConsensusBank

        cons_bank = ConsensusBank(
            path=args.cons_dir, label_mode=args.cons_label, exclude=tuple(args.cons_exclude)
        )
        print(
            f"cohort bank: {cons_bank.path}, {len(cons_bank.cohorts)} cohorts, {args.cons_label} labels",
            flush=True,
        )

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        # Scenario of this step
        train_cfg = curriculum(pcfg, step, args.warm)
        ramp = min(1.0, step / max(1, args.warm_regime))
        train_cfg, regime = apply_regimes(
            train_cfg,
            regime_rng,
            p_tissue=args.p_tissue * ramp,
            p_tech=args.p_tech * ramp,
            p_bio=args.p_bio * ramp,
            p_sparse=args.p_sparse * ramp,
            p_clin=args.p_clin * ramp,
        )
        train_cfg.p_aligned = args.p_aligned * ramp
        train_cfg.p_covar = args.p_covar * ramp
        train_cfg.p_paired = args.p_paired * ramp

        # Genes and datasets of this step
        G_step = sample_G(g_rng, args.G_min, args.G_max)
        B_step = effective_B(args.B, G_step, args.tokens_per_step)
        train_cfg.G = G_step

        # Source of the batch: a real cohort, a GTEx background or the prior
        is_cons = cons_bank is not None and regime_rng.random() < args.p_cons * ramp
        is_real = (not is_cons) and real_bank is not None and regime_rng.random() < args.p_real * ramp
        if is_cons:
            npg = int(real_np_rng.integers(train_cfg.n_per_group[0], train_cfg.n_per_group[1] + 1))
            d = cons_bank.sample_batch(B_step, npg, G_step, real_np_rng, device)
        elif is_real:
            npg = int(real_np_rng.integers(train_cfg.n_per_group[0], train_cfg.n_per_group[1] + 1))
            if real_bank2 is not None and regime_rng.random() < args.real_bank2_frac:
                d = real_bank2.sample_batch(B_step, npg, G_step, real_np_rng, device)
            else:
                d = real_bank.sample_batch(B_step, npg, G_step, real_np_rng, device)
        else:
            d = generate_batch(train_cfg, B_step, device)
            if args.p_onoff > 0 and regime_rng.random() < args.p_onoff * ramp:
                d = apply_onoff(d)

        # Count edits: low-count tail (not on real batches) and depth thinning
        if args.p_lowtail > 0 and not is_cons and not is_real:
            if lowtail_rng.random() < args.p_lowtail * ramp:
                d = apply_lowtail(d, lowtail_rng, tuple(args.lowtail_frac), tuple(args.lowtail_mean))
        if args.p_thin > 0 and regime_rng.random() < args.p_thin * ramp:
            keep = float(regime_rng.uniform(args.thin_range[0], args.thin_range[1]))
            d["counts"] = torch.binomial(d["counts"].float(), torch.full_like(d["counts"].float(), keep))

        # Step. Real-cohort batches mask the genes without a label.
        out, _ = model(d["counts"], d["cond"], d["batch"], covar=d.get("covar"))
        loss = model.head.nll(out, d["lfc_true"], weight=d.get("lfc_mask"))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            lr = opt.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(
                f"step {step:7d}  G {G_step:5d}  B {B_step:2d}  nll {loss.item():.4f}  "
                f"grad {float(grad_norm):.3f}  lr {lr:.2e}  {elapsed:7.0f}s",
                flush=True,
            )
            if run is not None:
                run.log(
                    {"train/nll": loss.item(), "train/grad_norm": float(grad_norm), "train/lr": lr}, step=step
                )

        # Validation, latest and best checkpoints
        if step % args.eval_every == 0:
            rep = evaluate(model, eval_cfg, device, n_batches=args.eval_batches, B=args.eval_B, tau=args.tau)
            print("  eval " + "  ".join(f"{k} {v:.4f}" for k, v in rep.items()), flush=True)
            if run is not None:
                run.log({f"eval/{k}": v for k, v in rep.items()}, step=step)

            torch.save(checkpoint_dict(step, model, opt, sched, mcfg, pcfg, args, rep), latest_path)
            if rep["nll"] < best_nll:
                best_nll = rep["nll"]
                torch.save(checkpoint_dict(step, model, opt, sched, mcfg, pcfg, args, rep), best_path)
                print(f"  new best: nll={best_nll:.4f}  saved to {best_path}", flush=True)

    torch.save(checkpoint_dict(args.steps, model, opt, sched, mcfg, pcfg, args), latest_path)
    if run is not None:
        run.finish()
    print(f"latest checkpoint: {latest_path}")
    print(f"best checkpoint:   {best_path} (validation nll {best_nll:.4f})")


if __name__ == "__main__":
    main()
