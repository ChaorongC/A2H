Version: A2H v0.4.0

Environment requirement:
    - python >= 3.7
    - pandas
    - numpy
    - scipy
    - joblib
    - pyarrow

Note:
    The bundled A2H_coef.pkl contains CHALM and CAMDA coefficients in
    columns chalm_coef and camda_coef. The current pickle requires pyarrow
    when loaded by pandas.


==========A2H v0.4.0==========

usage: A2H.py [-h] [-i INPUT] [-o OUTPUT] [--prefix PREFIX] [-c COEF]
              [--cpg-info CPG_INFO_PATH] [-t THREAD] [--ref {hg38,hg19}]
              [--beadchip {450k,EPIC,HM450,850k}] [--cl] [--probe]
              [--verbose VERBOSE] [--chalm] [--camda] [--filter FILTER]
              [--exclude-constant] [--replace REPLACE]

options:
  -h, --help            show this help message and exit
  -i INPUT              Input methylation file with first column as probe ID
                        and first row as sample ID.
  -o OUTPUT             Output directory for A2H transformation.
  --prefix PREFIX       Prefix for A2H transformation output.
  -c COEF               A2H coefficients provided in .pkl format.
  --cpg-info CPG_INFO_PATH
                        CpG loci and their information in training data.
  -t THREAD             Number of parallel workers for transformation.
  --ref {hg38,hg19}     Reference genome version for probe mapping.
  --beadchip {450k,EPIC,HM450,850k}
                        Input beadchip platform.
  --cl                  Output confidence level for each CpG.
  --probe               Use probe ID for output index; otherwise use CpG loci.
  --verbose VERBOSE     Print A2H transformation progress.
  --chalm               Output CHALM.
  --camda               Output CAMDA.
  --filter FILTER       Minimum confidence level to keep transformed values.
                        If assigned, this also enables --cl.
  --exclude-constant    Drop the 501 constant-input CpGs listed in
                        A2H_constant_cpgs.tsv from all outputs.
  --replace REPLACE     If assigned, replace filtered and missing values with
                        this value. Otherwise, filtered values are NaNs.


Bundled Test Use Case

Input:
    test_methylation.tsv

Command:
    python A2H.py \
        -i test_methylation.tsv \
        -o output \
        --prefix A2H_test \
        --ref hg38 \
        --beadchip EPIC \
        --chalm \
        --camda \
        --filter 0.1 \
        --verbose 0

Generated output files:
    output/A2H_test.chalm.tsv
    output/A2H_test.camda.tsv
    output/A2H_test.confidence.tsv
    output/A2H_test.chalm.filtered.tsv
    output/A2H_test.camda.filtered.tsv

Observed test result summary:
    output/A2H_test.chalm.tsv              shape=(10, 93), missing=37
    output/A2H_test.camda.tsv              shape=(10, 93), missing=37
    output/A2H_test.confidence.tsv         shape=(10, 93), missing=37
    output/A2H_test.chalm.filtered.tsv     shape=(10, 93), missing=252
    output/A2H_test.camda.filtered.tsv     shape=(10, 93), missing=252

The first output CpG locus is chr10_100006425 and the first sample is
SRX1648432. CHALM and CAMDA outputs have the same shape, but differ
numerically because they use different coefficient columns.


Additional Examples

1. Run CHALM only:
    python A2H.py \
        -i test_methylation.tsv \
        -o output \
        --prefix A2H_chalm_only \
        --ref hg38 \
        --beadchip EPIC \
        --chalm

2. Run CAMDA only:
    python A2H.py \
        -i test_methylation.tsv \
        -o output \
        --prefix A2H_camda_only \
        --ref hg38 \
        --beadchip EPIC \
        --camda

3. Output probe IDs instead of CpG loci:
    python A2H.py \
        -i test_methylation.tsv \
        -o output \
        --prefix A2H_probe_index \
        --ref hg38 \
        --beadchip EPIC \
        --chalm \
        --probe

4. Filter by confidence level and replace filtered values with -1:
    python A2H.py \
        -i test_methylation.tsv \
        -o output \
        --prefix A2H_filter_replace \
        --ref hg38 \
        --beadchip EPIC \
        --chalm \
        --camda \
        --filter 0.1 \
        --replace -1


Input Format

Replace test_methylation.tsv with your methylation matrix TSV file. The first
row must contain sample IDs, and the first column must contain array probe IDs.
