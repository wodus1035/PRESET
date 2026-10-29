"""SEQC ERCC input of seqc_ercc_fair.py: BGI site, samples A (UHRR + ERCC Mix 1) and B (HBRR + ERCC Mix 2), five libraries each.
Reads from <PRESET_DATA>/seqc/: GSE47774_SEQC_ILM_BGI.txt (GEO), gene2refseq.gz (NCBI Gene), cms_095046.txt (Thermo Fisher ERCC
mix concentrations). Writes SEQC_BGI_AB_gene_ERCC.tsv (lanes summed per library, RefSeq NM_/NR_ summed per Entrez gene) and
ERCC_truth.tsv (true_lfc = log2 nominal Mix 1 / Mix 2 ratio, i.e. A over B).
    python analysis/data_prep/prep_seqc_ercc.py"""

import os, re, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
D = os.path.join(os.environ.get("PRESET_DATA", os.path.join(ROOT, "data")), "seqc")

df = pd.read_csv(os.path.join(D, "GSE47774_SEQC_ILM_BGI.txt"), sep="\t")
groups = {}
for c in df.columns[1:]:
    m = re.search(r"SEQC_ILM_BGI_([A-F])_(\d+)_", c)
    if m:
        groups.setdefault(f"{m.group(1)}_{m.group(2)}", []).append(c)
out = pd.DataFrame({"TranscriptID": df["TranscriptID"]})
for k, cols in groups.items():
    out[k] = df[cols].sum(axis=1)
cnt = out[["TranscriptID"] + [f"A_{i}" for i in range(1, 6)] + [f"B_{i}" for i in range(1, 6)]]
sample_cols = [c for c in cnt.columns if c != "TranscriptID"]

# human rows only
m = pd.concat(
    [
        ch[ch.iloc[:, 0] == "9606"]
        for ch in pd.read_csv(
            os.path.join(D, "gene2refseq.gz"),
            sep="\t",
            compression="gzip",
            usecols=[0, 1, 3],
            dtype=str,
            chunksize=2_000_000,
        )
    ]
)
m.columns = [c.lstrip("#") for c in m.columns]
m = m[["GeneID", "RNA_nucleotide_accession.version"]].copy()
m["GeneID"] = m["GeneID"].astype(int)
m["RNA"] = m["RNA_nucleotide_accession.version"].str.replace(r"\.\d+$", "", regex=True)
human = cnt[cnt["TranscriptID"].str.startswith(("NM_", "NR_"))].copy()
ercc = cnt[cnt["TranscriptID"].str.startswith("ERCC_")].copy()
human = human.merge(m[["GeneID", "RNA"]], left_on="TranscriptID", right_on="RNA", how="inner")
gene = human.groupby("GeneID", as_index=False)[sample_cols].sum()
gene["TranscriptID"] = "GENE_" + gene["GeneID"].astype(str)
gene = gene[["TranscriptID"] + sample_cols]
pd.concat([gene, ercc], ignore_index=True).to_csv(
    os.path.join(D, "SEQC_BGI_AB_gene_ERCC.tsv"), sep="\t", index=False
)
print("human genes", len(gene), "ERCC", len(ercc), flush=True)

t = pd.read_csv(os.path.join(D, "cms_095046.txt"), sep="\t")
t.columns = [
    "re_sort_id",
    "ERCC_ID",
    "subgroup",
    "mix1_attomol_ul",
    "mix2_attomol_ul",
    "expected_fc_mix1_over_mix2",
    "log2_mix1_over_mix2",
][: len(t.columns)]
t["ERCC_ID"] = t["ERCC_ID"].str.replace("-", "_")
for c in ("mix1_attomol_ul", "mix2_attomol_ul"):
    t[c] = t[c].astype(float)
# nominal subgroup ratios (the file rounds 2/3 to 0.67)
t["expected_fc_mix1_over_mix2"] = t["subgroup"].map({"A": 4.0, "B": 1.0, "C": 2 / 3, "D": 0.5})
t["true_lfc"] = np.log2(t["expected_fc_mix1_over_mix2"])
t[
    ["ERCC_ID", "subgroup", "mix1_attomol_ul", "mix2_attomol_ul", "expected_fc_mix1_over_mix2", "true_lfc"]
].to_csv(os.path.join(D, "ERCC_truth.tsv"), sep="\t", index=False)
print("ERCC truth", len(t))
