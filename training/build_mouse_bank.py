"""Build the low-dispersion bank of the retraining in Supplementary Note S6 (train.py --real_bank2).

The 24 tissues of the GSE314817 mouse atlas (inbred C57BL/6 mice of both sexes and six ages, 80-104 samples per tissue),
whose within-tissue dispersion is far below that of the GTEx bank. GSE314817 is already a cohort of the second stage.
    python training/build_mouse_bank.py
writes training/data/mouse_atlas_bank_v1.npz with X_<tissue> (genes x samples, int32), samples_<tissue>, tissues and genes.
"""

import os

import numpy as np

import cohorts


def main():
    out = os.path.join(cohorts.HERE, "data", "mouse_atlas_bank_v1.npz")
    X, m, gn = cohorts.mouse_atlas()

    arrays = {}
    names = []
    for t in sorted(m.tissue.unique()):
        idx = np.where(m.tissue.to_numpy() == t)[0]
        k = t.replace(" ", "_")
        arrays[f"X_{k}"] = X[:, idx].astype(np.int32)
        arrays[f"samples_{k}"] = m.index.to_numpy()[idx].astype(str)
        names.append(k)
        print(f"  {k}: {len(idx)} samples", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **arrays, tissues=np.array(names), genes=gn)
    print("written", out, len(names), "tissues", X.shape[0], "genes", flush=True)


if __name__ == "__main__":
    main()
