#!/usr/bin/env bash
# Download the public data of Table 1 into $PRESET_DATA (default evaluation/data) and prepare the SEQC inputs.
#   bash evaluation/data/download.sh               # gtex heldout3 heldout tcga seqc
#   bash evaluation/data/download.sh gtex seqc     # some of them; tcga_sim only without the simulated datasets of the release
# Files already present are skipped, so an interrupted run can be repeated.
set -e
EVAL="$(cd "$(dirname "$0")/.." && pwd)"
D="${PRESET_DATA:-$EVAL/data}"
PYTHON="${PYTHON:-python}"
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series

get() {
    [ -s "$D/$1" ] || curl -fL --create-dirs -o "$D/$1" "$2"
}

gtex() {
    # GTEx v10 gene read counts of four tissues and the subject phenotypes (open access, phs000424)
    local G=https://storage.googleapis.com/adult-gtex
    for t in lung whole_blood muscle_skeletal adipose_subcutaneous; do
        get gtex/gene_reads_v10_$t.gct.gz $G/bulk-gex/v10/rna-seq/counts-by-tissue/gene_reads_v10_$t.gct.gz
    done
    get gtex/GTEx_v10_SubjectPhenotypes.txt $G/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt
}

heldout3() {
    # four disease cohorts
    get heldout3/GSE150910_gene-level_count_file.csv.gz $GEO/GSE150nnn/GSE150910/suppl/GSE150910_gene-level_count_file.csv.gz
    get heldout3/GSE185263_raw_counts.csv.gz $GEO/GSE185nnn/GSE185263/suppl/GSE185263_raw_counts.csv.gz
    get heldout3/GSE185263_family.soft.gz $GEO/GSE185nnn/GSE185263/soft/GSE185263_family.soft.gz
    get heldout3/GSE121212_readcount.txt.gz $GEO/GSE121nnn/GSE121212/suppl/GSE121212_readcount.txt.gz
    get heldout3/GSE57945_family.soft.gz $GEO/GSE57nnn/GSE57945/soft/GSE57945_family.soft.gz
    get heldout3/GSE57945_RAW.tar $GEO/GSE57nnn/GSE57945/suppl/GSE57945_RAW.tar

    # GSE57945 has one count file per sample
    mkdir -p "$D/heldout3/GSE57945"
    tar -xf "$D/heldout3/GSE57945_RAW.tar" -C "$D/heldout3/GSE57945"
}

heldout() {
    # mouse Dchs1 and Fat4 genotypes
    get heldout/GSE272152_Dchs1_counts.txt.gz $GEO/GSE272nnn/GSE272152/suppl/GSE272152_Dchs1_counts.txt.gz
    get heldout/GSE272152_Fat4_counts.txt.gz $GEO/GSE272nnn/GSE272152/suppl/GSE272152_Fat4_counts.txt.gz
}

tcga() {
    # STAR counts of four cancers from the GDC API
    (cd "$EVAL" && $PYTHON analysis/data_prep/tcga_download.py --projects TCGA-KIRC TCGA-LUAD TCGA-THCA TCGA-LIHC --out "$D/tcga")
}

tcga_sim() {
    # TCGA-BRCA normals and their paired tumors, the source of the SPsimSeq datasets
    (cd "$EVAL" && $PYTHON analysis/data_prep/tcga_download.py --projects TCGA-BRCA --max_tumour 0 --out "$D/tcga_sim")
}

seqc() {
    # SEQC counts (GSE47774, BGI site), RefSeq to gene mapping and ERCC Mix 1 and Mix 2 concentrations
    if [ ! -s "$D/seqc/GSE47774_SEQC_ILM_BGI.txt" ]; then
        get seqc/GSE47774_SEQC_ILM_BGI.txt.gz $GEO/GSE47nnn/GSE47774/suppl/GSE47774_SEQC_ILM_BGI.txt.gz
        gunzip "$D/seqc/GSE47774_SEQC_ILM_BGI.txt.gz"
    fi
    get seqc/gene2refseq.gz https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2refseq.gz
    get seqc/cms_095046.txt https://assets.thermofisher.com/TFS-Assets/LSG/manuals/cms_095046.txt

    # inputs of the TaqMan and ERCC runs
    (cd "$EVAL" && PRESET_DATA="$D" $PYTHON analysis/data_prep/prep_seqc_taqman.py)
    (cd "$EVAL" && PRESET_DATA="$D" $PYTHON analysis/data_prep/prep_seqc_ercc.py)
}

for part in ${@:-gtex heldout3 heldout tcga seqc}; do
    case $part in
        gtex | heldout3 | heldout | tcga | tcga_sim | seqc) ;;
        *) echo "unknown part: $part (gtex heldout3 heldout tcga tcga_sim seqc)" >&2; exit 1 ;;
    esac
    echo "== $part"
    $part
done
echo "data in $D"
