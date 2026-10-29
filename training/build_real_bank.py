"""Build the GTEx bank of the real-background tasks from the GTEx v10 gene read counts.

The bank holds the 11 GTEx tissues that no evaluation uses, restricted to the donors of the `bank` half of the donor
split (evaluation/data_cache/gtex_donor_split_v3.json), so training and the GTEx evaluations share no donor.
    python training/build_real_bank.py
reads <PRESET_DATA>/gtex/gene_reads_v10_<tissue>.gct.gz (download commands in training/README.md) and writes
training/data/gtex_real_bank_v3.npz with X_<tissue> (genes x samples, int32), samples_<tissue> and tissues.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from real_background import donor_of, load_donor_split

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TISSUES = [
    "breast_mammary_tissue",
    "esophagus_mucosa",
    "kidney_cortex",
    "liver",
    "ovary",
    "pancreas",
    "pituitary",
    "prostate",
    "spleen",
    "testis",
    "uterus",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gtex_dir",
        default=os.path.join(os.environ.get("PRESET_DATA", os.path.join(REPO, "evaluation", "data")), "gtex"),
    )
    ap.add_argument(
        "--split", default=os.path.join(REPO, "evaluation", "data_cache", "gtex_donor_split_v3.json")
    )
    ap.add_argument("--out", default=os.path.join(HERE, "data", "gtex_real_bank_v3.npz"))
    args = ap.parse_args()

    bank = set(load_donor_split(args.split)["bank"])
    arrays = {}
    for t in TISSUES:
        # Counts of one tissue, bank-half donors only
        df = pd.read_csv(
            os.path.join(args.gtex_dir, f"gene_reads_v10_{t}.gct.gz"),
            sep="\t",
            skiprows=2,
            compression="gzip",
        )
        samples = np.array([c for c in df.columns if c not in ("Name", "Description")])
        keep = np.array([d in bank for d in donor_of(samples)])
        arrays[f"X_{t}"] = df[samples[keep]].to_numpy(np.int32)
        arrays[f"samples_{t}"] = samples[keep]
        print(
            f"  {t}: {keep.sum()} samples of {len(np.unique(donor_of(samples[keep])))} bank donors",
            flush=True,
        )

    meta = {"split": os.path.basename(args.split)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **arrays, tissues=np.array(TISSUES), meta=np.array(json.dumps(meta)))
    print("written", args.out)


if __name__ == "__main__":
    main()
