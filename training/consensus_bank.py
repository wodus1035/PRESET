"""Real-cohort training tasks of the second stage, with labels from the other folds of the same cohort.

Each cohort of the bank (build_consensus_bank.py) is split into folds. A task draws n controls and n cases from one fold,
and its labels come from the remaining folds, so the input and the label never share a sample. Labels are
    exactly 0          for genes that every conventional method called unchanged,
    the agreed log2FC  for genes called changed with agreeing effect sizes,
    NaN                for every other gene, which is masked out of the loss.
The counts are the real counts, without planting.

label_mode "observed" uses the log2 ratio of the reference folds instead (add_observed_labels.py), so no DE method
enters the labels. exclude leaves out the cohorts whose names contain one of the given strings.
"""

import glob
import json
import os

import numpy as np
import torch


class ConsensusBank:
    def __init__(self, path, label_mode="consensus", exclude=()):
        key = {"consensus": "lab", "observed": "lab_obs"}[label_mode]
        self.path = path
        self.min_count = 10  # a gene is expressed if this count is reached in a quarter of the samples
        self.cohorts = []

        for fn in sorted(glob.glob(os.path.join(path, "*.npz"))):
            z = np.load(fn, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            if any(e in meta["name"] for e in exclude):
                continue
            if key not in z.files:
                raise KeyError(f"{fn} has no {key} (add_observed_labels.py adds the observed labels)")

            X = z["X"]
            cond = z["cond"].astype(int)
            fold = z["fold"].astype(int)
            lab = z[key]

            # largest n per group that each fold can supply
            caps = []
            for f in range(meta["K"]):
                in_f = fold == f
                caps.append(int(min((cond[in_f] == 0).sum(), (cond[in_f] == 1).sum())))
            if max(caps) < 2:
                continue

            self.cohorts.append(
                dict(name=meta["name"], X=X, cond=cond, fold=fold, lab=lab, caps=caps, K=meta["K"])
            )

        if not self.cohorts:
            raise FileNotFoundError(f"no cohorts under {path}")

    def _one(self, rng, n_per_group, G):
        c = self.cohorts[rng.integers(len(self.cohorts))]
        f = int(rng.integers(c["K"]))
        n = int(min(n_per_group, c["caps"][f]))
        if n < 2:
            # fold too small for this n, take the fold with the largest capacity
            f = int(np.argmax(c["caps"]))
            n = int(min(n_per_group, c["caps"][f]))

        # Samples of fold f
        infold = np.where(c["fold"] == f)[0]
        ctrl = rng.choice(infold[c["cond"][infold] == 0], n, replace=False)
        case = rng.choice(infold[c["cond"][infold] == 1], n, replace=False)
        cols = np.concatenate([ctrl, case])
        cnt = c["X"][:, cols].astype(np.float64)

        # G expressed genes and their labels from the other folds
        expressed = np.where((cnt >= self.min_count).sum(1) >= max(2, (2 * n) // 4))[0]
        genes = rng.choice(expressed, G, replace=len(expressed) < G)
        cnt = cnt[genes]
        lab = c["lab"][f, genes]
        mask = np.isfinite(lab).astype(np.float32)
        lfc = np.where(mask > 0, lab, 0.0).astype(np.float32)
        return cnt.T, lfc, mask, n

    def sample_batch(self, B, n_per_group, G, rng, device):
        # all B tasks share one n so that the tensors stack, the first task sets it
        out = [self._one(rng, n_per_group, G)]
        n = out[0][3]
        while len(out) < B:
            o = self._one(rng, n, G)
            if o[3] == n:
                out.append(o)

        counts = torch.tensor(np.stack([o[0] for o in out]), dtype=torch.float32, device=device)
        lfc = torch.tensor(np.stack([o[1] for o in out]), dtype=torch.float32, device=device)
        mask = torch.tensor(np.stack([o[2] for o in out]), dtype=torch.float32, device=device)
        cond = torch.cat([torch.zeros(n), torch.ones(n)]).to(device).expand(B, -1).contiguous()
        return {
            "counts": counts,
            "cond": cond,
            "batch": torch.zeros_like(cond),
            "lfc_true": lfc,
            "lfc_mask": mask,
            "is_null": torch.zeros(B, device=device),
        }
