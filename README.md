# A2H

Version: `A2H v0.4.0`

A2H transforms DNA methylation array beta values (Illumina HM450 or EPIC) into
CHALM and CAMDA values. For each CpG it applies a degree-2 polynomial fitted on a
reference WGBS methylation atlas, mapping the methylation level to CHALM and,
separately, to CAMDA. Optionally, it reports how well each input beta value is
supported by the reference data (confidence level) and masks poorly supported
values.

## Requirements

- Python >= 3.8
- `numpy` >= 1.17
- `pandas` >= 1.0
- `scipy` >= 1.4
- `joblib` >= 0.14

These minimum versions were tested on Python 3.8 (numpy 1.17.5, pandas 1.0.5,
scipy 1.4.1, joblib 0.14.1). A2H was also tested on Python 3.13 with numpy
2.1/2.4, pandas 2.2/3.0, scipy 1.15/1.17 and joblib 1.4/1.5. The bundled
resource files are plain NumPy, TSV and text files; no pickle or `pyarrow` is
needed.

Install the dependencies, for example with:

```bash
pip install numpy pandas scipy joblib
```

## Quick Start

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

## Usage

```text
A2H.py [-h] [-i INPUT] [-o OUTPUT] [--prefix PREFIX] [-c COEF]
       [--cpg-info CPG_INFO_PATH] [-t THREAD] [--ref {hg38,hg19}]
       [--beadchip {450k,EPIC,HM450,850k}] [--cl] [--probe]
       [--verbose VERBOSE] [--chalm] [--camda] [--filter FILTER]
       [--exclude-constant] [--replace REPLACE]
```

| Option | Description |
| --- | --- |
| `-i INPUT` | Input methylation matrix (TSV). See [Input Format](#input-format). |
| `-o OUTPUT` | Output directory. Must already exist. Default: current directory. |
| `--prefix PREFIX` | Prefix for output file names. Default: `A2H`. |
| `--chalm` | Output CHALM. At least one of `--chalm` and `--camda` is required. |
| `--camda` | Output CAMDA. |
| `--beadchip {450k,EPIC,HM450,850k}` | Input array platform. `450k` and `HM450` are the same, as are `EPIC` and `850k`. Default: `EPIC`. EPIC v2 input is not supported. |
| `--ref {hg38,hg19}` | Genome build of the probe annotation used to map probe IDs to CpG loci. Output locus IDs are hg38 for both choices. Default: `hg38`. |
| `--probe` | Index outputs by input probe ID instead of CpG locus. |
| `--cl` | Also output the confidence level of every input value. |
| `--filter FILTER` | Write additional filtered outputs in which values with confidence level below `FILTER` are set to `NaN`. Implies `--cl`. |
| `--replace REPLACE` | With `--filter`, write `REPLACE` instead of `NaN` for filtered and missing values in the filtered outputs. |
| `--exclude-constant` | Drop the 501 constant-input CpGs (see [Constant-Input CpGs](#constant-input-cpgs)) from all outputs. |
| `-t THREAD` | Number of parallel workers for the CHALM/CAMDA transformation. Default: `-1` (all CPUs). |
| `--verbose VERBOSE` | Joblib progress verbosity. `0` silences progress. Default: `1`. |
| `-c COEF` | Coefficient file. Default: bundled `A2H_coef.npz`. |
| `--cpg-info CPG_INFO_PATH` | CpG information file used for confidence levels. Default: bundled `A2H_cpg_info.npz`. |

## Input Format

A tab-separated matrix with one row per array probe and one column per sample:

- The first column holds probe IDs (for example, `cg03877640`). Its header name
  is not used.
- The remaining header fields are sample IDs.
- Values are beta values between 0 and 1. Write `NA` or leave the cell empty
  for a missing value.

```text
Probe_ID	sample_1	sample_2
cg03877640	0.273	0.130
cg19212550	0.933	NA
```

Probes that are not in the probe map for the chosen `--beadchip` and `--ref`
are dropped silently. `test_methylation.tsv` is a complete example: 10 EPIC
probes by 93 samples.

## Output Files

All outputs are tab-separated matrices with the same columns as the input. The
rows are CpG loci, or input probe IDs with `--probe`.

| File | Written when | Content |
| --- | --- | --- |
| `<prefix>.chalm.tsv` | `--chalm` | A2H-predicted CHALM |
| `<prefix>.camda.tsv` | `--camda` | A2H-predicted CAMDA |
| `<prefix>.confidence.tsv` | `--cl` or `--filter` | Confidence level of each input value, rounded to 4 decimals |
| `<prefix>.chalm.filtered.tsv` | `--chalm` and `--filter` | CHALM with low-confidence values masked |
| `<prefix>.camda.filtered.tsv` | `--camda` and `--filter` | CAMDA with low-confidence values masked |

- A missing input value gives a missing output value.
- Predictions are the raw polynomial values and are not clipped to the range of
  the target measure (for example, some CHALM values in the bundled test output
  exceed 1).
- **Confidence level:** for an input beta value `b` at a CpG,
  `1 - 2 * |0.5 - F(floor(100 * b))|`, where `F` is the CDF of the beta-binomial
  distribution (n = 100) fitted to that CpG's reference methylation levels.
  The value is near 1 when `b` lies in the middle of the reference distribution
  and near 0 in its tails.

## Bundled Resource Files

The locus ID format used throughout is `chr<N>_<position>`: the hg38 chromosome
and the 1-based position of the CpG cytosine on the forward strand (for example,
`chr10_100006425` for `cg03877640`).

### `A2H_coef.npz`

NumPy archive (`numpy.load`) of the per-CpG polynomial coefficients. It covers
1,030,653 CpGs.

| Array | dtype, shape | Content |
| --- | --- | --- |
| `featureID` | string, `(n,)` | CpG locus ID |
| `chalm_coef` | float64, `(n, 3)` | CHALM coefficients `[c2, c1, c0]` |
| `camda_coef` | float64, `(n, 3)` | CAMDA coefficients `[c2, c1, c0]` |

The prediction for beta value `b` is `c2 * b**2 + c1 * b + c0`
(`numpy.polyval(coef, b)`).

### `A2H_cpg_info.npz`

NumPy archive of per-CpG summaries of the reference WGBS methylation levels
used in training. It has the same 1,030,653 CpGs as `A2H_coef.npz`. Each array
has shape `(n,)`.

| Array | dtype | Content |
| --- | --- | --- |
| `featureID` | string | CpG locus ID |
| `min`, `max`, `mean`, `std` | float64 | Minimum, maximum, mean and standard deviation of the reference methylation level |
| `betabinom_a`, `betabinom_b` | float64 | Beta-binomial shape parameters (n = 100) fitted to the reference methylation levels, used for the confidence level |

### `A2H_cpg_probe.hg38.tsv`, `A2H_cpg_probe.hg19.tsv`

TSV maps from array probe IDs to CpG loci. Column `loci.hg38` holds the hg38
locus ID, and each other column holds the probe ID on one platform (empty if
that platform has no probe at the CpG).

- hg38 map: `EPICv2.hg38`, `EPIC+.hg38`, `EPIC.hg38`, `HM450.hg38`
- hg19 map: `EPIC.hg19`, `EPICv2.hg19`, `HM450.hg19`

A2H reads `loci.hg38` and the `EPIC.<ref>` or `HM450.<ref>` column.

### `A2H_constant_cpgs.tsv`

The 501 constant-input CpGs (see below), one row per CpG.

| Column | Content |
| --- | --- |
| `locus` | CpG locus ID |
| `n_reference_samples` | Number of reference samples at this CpG |
| `HM450_probe`, `EPIC_probe`, `EPICv2_probe` | Probe ID on each platform (empty if none) |
| `A2H_CHALM_output`, `A2H_CAMDA_output` | Constant value A2H returns at this CpG |

## Constant-Input CpGs

At 501 CpGs the reference methylation level was identical in every reference
sample (50 to 205 samples per CpG): it was 0 at 500 CpGs and 1 at 1 CpG. No
polynomial can be fitted there, so A2H returns a single constant, estimated by
kernel density, whatever the input beta value. CHALM is 0 at 500 of these CpGs
and 1 at the other, and CAMDA is 0 at all 501.

These CpGs are kept by default. Their confidence level is undefined (`NaN`) at
500 CpGs and below 1e-12 at the other, so `--filter` masks them in the filtered
outputs. Use `--exclude-constant` to remove them from all outputs, including the
confidence output.

## Bundled Test Use Case

The [Quick Start](#quick-start) command generates:

| File | Shape | Missing values |
| --- | --- | --- |
| `output/A2H_test.chalm.tsv` | `(10, 93)` | 37 |
| `output/A2H_test.camda.tsv` | `(10, 93)` | 37 |
| `output/A2H_test.confidence.tsv` | `(10, 93)` | 37 |
| `output/A2H_test.chalm.filtered.tsv` | `(10, 93)` | 252 |
| `output/A2H_test.camda.filtered.tsv` | `(10, 93)` | 252 |

The first output CpG locus is `chr10_100006425`, and the first sample is
`SRX1648432`. CHALM and CAMDA outputs have the same shape but different values,
because they use different coefficients. The reference outputs are included in
`output/`.

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

Index outputs by probe ID instead of CpG locus:

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

Filter by confidence level and write `-1` for filtered values:

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

Drop the constant-input CpGs from all outputs:

```bash
python A2H.py \
    -i test_methylation.tsv \
    -o output \
    --prefix A2H_exclude_constant \
    --ref hg38 \
    --beadchip EPIC \
    --chalm \
    --camda \
    --exclude-constant
```
