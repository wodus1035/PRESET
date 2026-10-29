# Example data

`GSE121212_skin_counts.tsv.gz` — raw read counts (genes × samples) for the 38 healthy-skin and 28 psoriasis-lesional biopsies of
GEO series [GSE121212](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121212) (Tsoi et al., J Invest Dermatol 2019),
taken from the series supplementary file `GSE121212_readcount.txt.gz`. Four rows whose gene symbols had been converted to dates
by a spreadsheet (`1-Mar`, `2-Mar`, each appearing twice) were removed because they cannot be told apart. Nothing else was changed.

`GSE121212_skin_samples.tsv` — sample id, group (`healthy` / `psoriasis`) and patient id parsed from the column names.

This cohort was not used to train PRESET or to choose any of its settings.

`MSigDB_Hallmark_2020.gmt` — the 50 MSigDB Hallmark gene sets (Liberzon et al., Cell Systems 2015), in the version distributed by
Enrichr (`MSigDB_Hallmark_2020`). MSigDB is © Broad Institute, Inc., Massachusetts Institute of Technology and Regents of the University
of California, licensed under the Creative Commons Attribution 4.0 International License (https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp).
Used in Section 10 of the tutorial.
