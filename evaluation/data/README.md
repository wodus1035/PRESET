# Data

`bash evaluation/data/download.sh` downloads every dataset below into this folder (or `PRESET_DATA`) and prepares the SEQC inputs. Names of parts (`gtex`, `heldout3`, `heldout`, `tcga`, `seqc`) download only those.

| Folder | Source | Rows of Table 1 |
|---|---|---|
| `gtex/` | GTEx v10 gene read counts of lung, whole blood, skeletal muscle and subcutaneous adipose, and subject phenotypes (GTEx Portal, phs000424) | GTEx |
| `heldout3/` | GEO GSE150910 (IPF and CHP lung), GSE185263 (sepsis blood), GSE57945 (Crohn's disease and ulcerative colitis ileum), GSE121212 (psoriasis and atopic dermatitis skin) | disease cohorts |
| `heldout/` | GEO GSE272152 (mouse *Dchs1* and *Fat4* heterozygotes) | mouse |
| `tcga/` | GDC STAR counts of TCGA-KIRC, LUAD, THCA and LIHC | TCGA |
| `seqc/` | GEO GSE47774 (SEQC, BGI site), NCBI gene2refseq, ERCC Mix 1 and 2 concentrations (Thermo Fisher) | ERCC, TaqMan |

Only the GTEx donors of the evaluation half are used (`data_cache/gtex_donor_split_v3.json`). `data_cache/seqc/` holds the MAQC TaqMan panel of GEO GSE5350 as one table and the RefSeq to Entrez mapping that `prep_seqc_taqman.py` obtained from mygene.info. The part `tcga_sim` (TCGA-BRCA, the source of SPsimSeq) is needed only without the simulated datasets of the release.
