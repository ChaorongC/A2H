# A2H

Version: `A2H v0.4.0`

A2H transforms array methylation beta values into CHALM and CAMDA outputs using
the bundled probe mapping and coefficient files.

## Requirements

- Python >= 3.7
- `pandas`
- `numpy`
- `scipy`
- `joblib`
- `pyarrow`

The bundled `A2H_coef.pkl` contains CHALM and CAMDA coefficients in
`chalm_coef` and `camda_coef`. The current pickle requires `pyarrow` when
loaded by `pandas`.

## Usage

```text
A2H.py [-h] [-i INPUT] [-o OUTPUT] [--prefix PREFIX] [-c COEF]
       [--cpg-info CPG_INFO_PATH] [-t THREAD] [--ref {hg38,hg19}]
       [--beadchip {450k,EPIC,HM450,850k}] [--cl] [--probe]
       [--verbose VERBOSE] [--chalm] [--camda] [--filter FILTER]
       [--replace REPLACE]
```

| Option | Description |
| --- | --- |
| `-i INPUT` | Input methylation TSV. First row must be sample IDs and first column must be probe IDs. |
| `-o OUTPUT` | Output directory for A2H transformation. |
| `--prefix PREFIX` | Prefix for output files. |
| `-c COEF` | A2H coefficient pickle. Defaults to bundled `A2H_coef.pkl`. |
| `--cpg-info CPG_INFO_PATH` | CpG loci and confidence-level metadata. Defaults to bundled `A2H_cpg_info.pkl`. |
| `-t THREAD` | Number of parallel workers for transformation. |
| `--ref {hg38,hg19}` | Reference genome version for probe mapping. |
| `--beadchip {450k,EPIC,HM450,850k}` | Input beadchip platform. |
| `--cl` | Output confidence level for each CpG. |
| `--probe` | Use probe IDs for output index instead of CpG loci. |
| `--verbose VERBOSE` | Joblib progress verbosity. Use `0` to silence progress. |
| `--chalm` | Output CHALM. |
| `--camda` | Output CAMDA. |
| `--filter FILTER` | Minimum confidence level to keep transformed values. Also enables `--cl`. |
| `--replace REPLACE` | Replace filtered and missing values with this value. By default filtered values are `NaN`. |

## Bundled Test Use Case

Run CHALM, CAMDA, confidence-level calculation, and confidence filtering on the
bundled `test_methylation.tsv`:

```bash
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
```

This command generates:

- `output/A2H_test.chalm.tsv`
- `output/A2H_test.camda.tsv`
- `output/A2H_test.confidence.tsv`
- `output/A2H_test.chalm.filtered.tsv`
- `output/A2H_test.camda.filtered.tsv`

Observed bundled test result summary:

| File | Shape | Missing values |
| --- | --- | --- |
| `output/A2H_test.chalm.tsv` | `(10, 93)` | 37 |
| `output/A2H_test.camda.tsv` | `(10, 93)` | 37 |
| `output/A2H_test.confidence.tsv` | `(10, 93)` | 37 |
| `output/A2H_test.chalm.filtered.tsv` | `(10, 93)` | 252 |
| `output/A2H_test.camda.filtered.tsv` | `(10, 93)` | 252 |

The first output CpG locus is `chr10_100006425`, and the first sample is
`SRX1648432`. CHALM and CAMDA outputs have the same shape, but differ
numerically because they use different coefficient columns.

## Additional Examples

Run CHALM only:

```bash
python A2H.py \
    -i test_methylation.tsv \
    -o output \
    --prefix A2H_chalm_only \
    --ref hg38 \
    --beadchip EPIC \
    --chalm
```

Run CAMDA only:

```bash
python A2H.py \
    -i test_methylation.tsv \
    -o output \
    --prefix A2H_camda_only \
    --ref hg38 \
    --beadchip EPIC \
    --camda
```

Output probe IDs instead of CpG loci:

```bash
python A2H.py \
    -i test_methylation.tsv \
    -o output \
    --prefix A2H_probe_index \
    --ref hg38 \
    --beadchip EPIC \
    --chalm \
    --probe
```

Filter by confidence level and replace filtered values with `-1`:

```bash
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
```

## Input Format

Replace `test_methylation.tsv` with your methylation matrix TSV file. The first
row must contain sample IDs, and the first column must contain array probe IDs.
