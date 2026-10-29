"""SEQC/MAQC gene-level counts and two truths for samples A-D that no DE method helped define.
counts   GSE47774_SEQC_ILM_BGI.txt: lanes summed per library (A1-A5 ... D1-D5); every RefSeq transcript summed per Entrez gene
         (mygene.info), so the model sees the full transcriptome; unmapped accessions stay as their own rows
truth 1  TaqMan log2(B/A), mean of 4 qPCR replicates (MAQC_TaqMan_1044.tsv, GSE5350), where all A and B replicates are Present
truth 2  mixing design C = 0.75 A + 0.25 B, D = 0.25 A + 0.75 B: log2(D/C) from the TaqMan A and B levels (small effects)
  python analysis/data_prep/prep_seqc_taqman.py   -> data/seqc/seqc_gene_counts_ABCD.tsv, data/seqc/seqc_taqman_truth.tsv
"""

import os, sys, json, urllib.request, urllib.parse, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # evaluation/
D = os.path.join(os.environ.get("PRESET_DATA", os.path.join(ROOT, "data")), "seqc")
COUNTS = os.path.join(D, "GSE47774_SEQC_ILM_BGI.txt")
TAQ = os.path.join(ROOT, "data_cache", "seqc", "MAQC_TaqMan_1044.tsv")  # GEO GSE5350, included in data_cache
MIX = {"C": (0.75, 0.25), "D": (0.25, 0.75)}


# mygene.info answers queried in October 2026, cached and included
def map_refseq(accs, chunk=500, cache=os.path.join(ROOT, "data_cache", "seqc", "refseq_to_entrez.tsv")):
    """RefSeq accession (version stripped) -> Entrez gene ID via mygene.info, cached on disk."""
    have = {}
    if os.path.exists(cache):
        c = pd.read_csv(cache, sep="\t", dtype=str)
        have = dict(zip(c.accession, c.entrezgene))
    todo = [a for a in accs if a not in have]
    for i in range(0, len(todo), chunk):
        part = todo[i : i + chunk]
        body = urllib.parse.urlencode(
            {"q": ",".join(part), "scopes": "refseq", "fields": "entrezgene", "species": "human"}
        ).encode()
        req = urllib.request.Request(
            "https://mygene.info/v3/query",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        for rec in json.loads(urllib.request.urlopen(req, timeout=180).read()):
            e = rec.get("entrezgene")
            if e is not None:
                have.setdefault(str(rec.get("query")), str(e))
        if (i // chunk) % 10 == 0:
            print(f"  mygene {min(i + chunk, len(todo))}/{len(todo)}", flush=True)
    pd.DataFrame({"accession": list(have), "entrezgene": [have[k] for k in have]}).to_csv(
        cache, sep="\t", index=False
    )
    return have


def main():
    taq = pd.read_csv(TAQ, sep="\t")
    # R wrote them as numeric -> "1543.0"
    taq = taq[taq.EntrezID.notna()].copy()
    taq["EntrezID"] = taq.EntrezID.astype("int64").astype(str)
    val = {s: taq[[f"{s}{r}_value" for r in range(1, 5)]].to_numpy(float) for s in "ABCD"}
    det = {s: taq[[f"{s}{r}_detection" for r in range(1, 5)]].to_numpy(str) for s in "ABCD"}
    ok = (det["A"] == "P").all(1) & (det["B"] == "P").all(1)
    a = val["A"].mean(1)
    b = val["B"].mean(1)
    truth = pd.DataFrame(
        {
            "EntrezID": taq.EntrezID.to_numpy(),
            "Symbol": taq.Symbol,
            "taqman_A": a,
            "taqman_B": b,
            "present_AB": ok,
        }
    )
    for s_ in "CD":
        truth[f"taqman_{s_}_obs"] = val[s_].mean(1)
    n_probe = len(truth)
    truth = truth.groupby("EntrezID", as_index=False).agg(  # several probes for some genes
        Symbol=("Symbol", "first"),
        taqman_A=("taqman_A", "mean"),
        taqman_B=("taqman_B", "mean"),
        taqman_C_obs=("taqman_C_obs", "mean"),
        taqman_D_obs=("taqman_D_obs", "mean"),
        present_AB=("present_AB", "all"),
        n_probes=("Symbol", "size"),
    )
    for s_, (wa, wb) in MIX.items():
        truth[f"mix_{s_}"] = wa * truth.taqman_A + wb * truth.taqman_B
    truth["lfc_BA"] = np.log2(
        truth.taqman_B.where(truth.taqman_B > 0) / truth.taqman_A.where(truth.taqman_A > 0)
    )
    truth["lfc_DC_mix"] = np.log2(truth.mix_D / truth.mix_C)  # D vs C
    truth["lfc_DC_obs"] = np.log2(truth.taqman_D_obs / truth.taqman_C_obs)
    print(
        f"TaqMan: {n_probe} probes -> {len(truth)} genes, present in all A+B replicates {int(truth.present_AB.sum())}",
        flush=True,
    )

    hdr = open(COUNTS).readline().rstrip("\n").split("\t")[1:]
    keep = {}  # column index -> library label "A_1"
    for j, c in enumerate(hdr):
        # SEQC_ILM_BGI_<sample>_<lib>_L0k_<barcode>_<flowcell>
        p_ = c.split("_")
        if len(p_) > 5 and p_[3] in "ABCD":
            keep[j] = f"{p_[3]}_{p_[4]}"
    libs = sorted(set(keep.values()))
    print(f"counts: {len(hdr)} lanes -> {len(libs)} libraries", flush=True)

    accs = [l.split("\t", 1)[0].split(".")[0] for l in open(COUNTS).readlines()[1:]]
    accs = [a for a in accs if not a.startswith("ERCC")]
    a2e = map_refseq(sorted(set(accs)))
    print(f"mygene: {len(a2e)} of {len(set(accs))} accessions mapped to an Entrez id", flush=True)

    # Entrez ID (or unmapped accession) -> lane counts
    agg = {}
    with open(COUNTS) as f:
        f.readline()
        for line in f:
            tid, rest = line.split("\t", 1)
            base = tid.split(".")[0]
            if base.startswith("ERCC"):
                continue  # ERCC: see prep_seqc_ercc.py
            rid = a2e.get(base, base)
            v = np.fromstring(rest, sep="\t")
            if rid in agg:
                agg[rid] += v
            else:
                agg[rid] = v
    print(
        f"counts: {len(agg)} rows after aggregating transcripts by gene (unmapped accessions kept as their own row)",
        flush=True,
    )

    idx = {l: [j for j, lab in keep.items() if lab == l] for l in libs}
    rows = {e: np.array([v[idx[l]].sum() for l in libs]) for e, v in agg.items()}
    C = pd.DataFrame(rows, index=libs).T.astype(np.int64)
    C.index.name = "EntrezID"
    C.to_csv(os.path.join(D, "seqc_gene_counts_ABCD.tsv"), sep="\t")
    truth[truth.EntrezID.isin(C.index)].to_csv(
        os.path.join(D, "seqc_taqman_truth.tsv"), sep="\t", index=False
    )
    t = truth[truth.EntrezID.isin(C.index) & truth.present_AB & truth.lfc_BA.notna()]
    print(
        f"\nwritten: seqc_gene_counts_ABCD.tsv ({C.shape[0]} genes x {C.shape[1]} libraries), seqc_taqman_truth.tsv"
    )
    print(
        f"usable A/B truth genes: {len(t)}; |lfc_BA| median {t.lfc_BA.abs().median():.3f}, >1: {(t.lfc_BA.abs() > 1).sum()}, <0.5: {(t.lfc_BA.abs() < 0.5).sum()}"
    )
    print(
        f"mixing truth |lfc_DC_mix| median {t.lfc_DC_mix.abs().median():.3f}, >1: {(t.lfc_DC_mix.abs() > 1).sum()}, <0.5: {(t.lfc_DC_mix.abs() < 0.5).sum()}"
    )


if __name__ == "__main__":
    main()
