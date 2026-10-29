"""Download TCGA STAR raw counts (GDC API, open access; unstranded column) and write one integer count matrix per project with a
sample table (barcode, sample_type, case_id) for tumor/normal and matched-pair selection.
    python analysis/data_prep/tcga_download.py --projects TCGA-KIRC TCGA-LUAD TCGA-THCA TCGA-LIHC --out data/tcga
"""

import os, sys, io, json, argparse, gzip, tarfile, time, requests, pandas as pd, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--projects", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_tumour", type=int, default=400)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
API = "https://api.gdc.cancer.gov"


def list_files(project):
    filt = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    params = {
        "filters": json.dumps(filt),
        "fields": "file_id,file_name,cases.case_id,cases.submitter_id,cases.samples.sample_type,cases.samples.submitter_id,cases.samples.portions.analytes.aliquots.submitter_id",
        "format": "JSON",
        "size": "5000",
    }
    r = requests.get(f"{API}/files", params=params, timeout=120)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    rows = []
    for h in hits:
        c = h["cases"][0]
        s = c["samples"][0]
        rows.append(
            dict(
                file_id=h["file_id"],
                file_name=h["file_name"],
                case_id=c["case_id"],
                case=c["submitter_id"],
                sample=s["submitter_id"],
                sample_type=s["sample_type"],
            )
        )
    return pd.DataFrame(rows)


def fetch(file_id, tries=4):
    for t in range(tries):
        try:
            r = requests.get(f"{API}/data/{file_id}", timeout=300)
            r.raise_for_status()
            return r.content
        except Exception as e:
            time.sleep(5 * (t + 1))
    raise RuntimeError(file_id)


def parse_star(content):
    txt = content.decode()
    df = pd.read_csv(io.StringIO(txt), sep="\t", comment="#")
    df = df[~df.gene_id.str.startswith("N_")]
    return df.set_index("gene_id")["unstranded"].astype(np.int64), df.set_index("gene_id")["gene_name"]


for proj in A.projects:
    tag = proj.replace("TCGA-", "")
    meta = list_files(proj)
    normal = meta[meta.sample_type == "Solid Tissue Normal"]
    tumour = meta[meta.sample_type == "Primary Tumor"]
    # all normals, tumors with a matched normal, then other tumors (shuffled, seed 0) up to --max_tumour
    paired = tumour[tumour.case_id.isin(normal.case_id)]
    rest = tumour[~tumour.case_id.isin(normal.case_id)].sample(frac=1, random_state=0)
    keep = (
        pd.concat([normal, paired, rest])
        .drop_duplicates("sample")
        .head(len(normal) + len(paired) + max(0, A.max_tumour - len(paired)))
    )
    print(
        f"{proj}: files {len(meta)}, normals {len(normal)}, tumours {len(tumour)} (paired {len(paired)}), downloading {len(keep)}",
        flush=True,
    )
    cols, names = {}, None
    for i, row in enumerate(keep.itertuples()):
        s, gn = parse_star(fetch(row.file_id))
        cols[row.sample] = s
        names = gn if names is None else names
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(keep)}", flush=True)
    M = pd.DataFrame(cols)
    M.insert(0, "gene_name", names.reindex(M.index).to_numpy())
    M.to_csv(os.path.join(A.out, f"{tag}_star_counts.tsv.gz"), sep="\t", compression="gzip")
    keep[["sample", "sample_type", "case", "case_id", "file_id"]].to_csv(
        os.path.join(A.out, f"{tag}_samples.tsv"), sep="\t", index=False
    )
    print(f"{proj}: wrote {M.shape[0]} genes x {M.shape[1] - 1} samples", flush=True)
