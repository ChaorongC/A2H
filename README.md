# A2H

Version: `A2H v0.5.0`

A2H transforms DNA methylation array beta values (Illumina HM450 or EPIC) into
CHALM and CAMDA values. For each CpG it applies a degree-2 polynomial fitted on a
reference WGBS methylation atlas, mapping the methylation level to CHALM and,
separately, to CAMDA. Optionally, it reports how well each input beta value is
supported by the reference data (confidence level) and masks poorly supported
values.

## Requirements

By default A2H reads the bundled pandas pickles `A2H_coef.pkl` and
`A2H_cpg_info.pkl`. Loading them needs:

- Python >= 3.9
- `numpy` >= 1.26
- `pandas` >= 2.3
- `pyarrow` (for `A2H_coef.pkl`; >= 10.0.1 with pandas 2.3, >= 13.0 with
  pandas 3)
- `scipy` and `joblib`

```bash
pip install "numpy>=1.26" "pandas>=2.3" pyarrow scipy joblib
```

`A2H_coef.pkl` stores its locus IDs as a pyarrow-backed pandas string column,
which pandas <= 2.2 and numpy <= 1.25 cannot unpickle. `A2H_cpg_info.pkl` alone
loads with pandas >= 1.4. These limits were found by loading both files under
pandas 1.3.5, 1.4.4, 1.5.3, 2.2.3, 2.3.0, 2.3.3 and 3.0.0 to 3.0.2, numpy 1.25.2,
1.26.4 and 2.x, and pyarrow 10.0.1, 12.0.1, 13.0.0 and later. With pandas 2.3.0,
numpy 1.26.4 and pyarrow 10.0.1, and with pandas 3.0.0 and pyarrow 13.0.0, the
bundled test outputs are reproduced exactly.

### Portable alternative: `.npz` resources

`A2H_coef.npz` and `A2H_cpg_info.npz` hold the same values as the pickles as
plain NumPy arrays. They load without pickle, pandas version constraints or
`pyarrow`. Pass them explicitly:

```bash
python A2H.py -i test_methylation.tsv -o output --prefix A2H_test \
    --chalm --camda --filter 0.1 \
    -c A2H_coef.npz --cpg-info A2H_cpg_info.npz
```

With the `.npz` files, A2H runs on Python >= 3.8 with numpy >= 1.17,
pandas >= 1.0, scipy >= 1.4 and joblib >= 0.14 (tested on Python 3.8 with
numpy 1.17.5, pandas 1.0.5, scipy 1.4.1 and joblib 0.14.1, and on Python 3.13
with pandas 2.2 and 3.0).

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
       [--clip] [--exclude-constant] [--replace REPLACE]
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
| `--clip` | Clip CHALM and CAMDA predictions to [0, 1], in both the raw and the filtered outputs. Off by default. |
| `--exclude-constant` | Drop the 501 constant-input CpGs (see [Constant-Input CpGs](#constant-input-cpgs)) from all outputs. |
| `-t THREAD` | Number of parallel workers for the CHALM/CAMDA transformation and the confidence-level calculation. Default: `-1` (all CPUs). |
| `--verbose VERBOSE` | Joblib progress verbosity. `0` silences progress. Default: `1`. |
| `-c COEF` | Coefficient file, `.pkl` or `.npz` (read by file extension). Default: bundled `A2H_coef.pkl`. |
| `--cpg-info CPG_INFO_PATH` | CpG information file used for confidence levels, `.pkl` or `.npz`. Default: bundled `A2H_cpg_info.pkl`. |

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
- By default, predictions are the raw polynomial values and are not clipped to
  the range of the target measure. In the bundled test output, 57 of 893 CHALM
  values exceed 1, and 23 of 893 CAMDA values fall outside [0, 1] (22 below 0,
  1 above 1). Most of these come from input beta values in the tails of a
  CpG's reference distribution, where the polynomial extrapolates; at
  `--filter 0.1`, 38 of the 57 CHALM values and all 23 CAMDA values are masked.
  Use `--clip` to bound predictions to [0, 1].
- **Confidence level:** for an input beta value `b` at a CpG,
  `1 - 2 * |0.5 - F(floor(100 * b))|`, where `F` is the CDF of the beta-binomial
  distribution (n = 100) fitted to that CpG's reference methylation levels.
  The value is near 1 when `b` lies in the middle of the reference distribution
  and near 0 in its tails.

## Bundled Resource Files

The locus ID format used throughout is `chr<N>_<position>`: the hg38 chromosome
and the 1-based position of the CpG cytosine on the forward strand (for example,
`chr10_100006425` for `cg03877640`).

The code that trains the model and builds these files from the reference WGBS
atlas is in [`reproducibility/`](reproducibility/README.md).

### Model and stored coefficients

For CpG $j$ and sample $s$ with input methylation level (beta value)
$m_{j,s}$, A2H predicts CHALM, and separately CAMDA, as

```math
\hat{y}_{j,s} = a_j m_{j,s}^2 + b_j m_{j,s} + c_j
```

with one set of coefficients $(a_j, b_j, c_j)$ per CpG and target. Each CpG's
coefficients are stored as a length-3 array `[c2, c1, c0] = [a_j, b_j, c_j]`,
highest power first, as `numpy.polyval(coef, m)` expects. At the 501
constant-input CpGs (see [Constant-Input CpGs](#constant-input-cpgs)) the stored
array is `[0, 0, c_j]`, so the prediction is the constant `c_j` for any input.

### `A2H_coef.pkl` (default) and `A2H_coef.npz`

The per-CpG coefficients for 1,030,653 CpGs. Both files hold identical values.
Use the `.npz` file for better compatibility across Python package versions and
easier transport: it loads with NumPy alone, without the pandas and `pyarrow`
versions the pickle needs.

- `A2H_coef.pkl`: a pandas DataFrame (`pandas.read_pickle`) with one row per
  CpG and columns `featureID` (CpG locus ID), `chalm_coef` and `camda_coef`
  (each cell a list `[a_j, b_j, c_j]`). See [Requirements](#requirements) for the
  versions needed to load it.
- `A2H_coef.npz`: a NumPy archive (`numpy.load`) with these arrays:

| Array | dtype, shape | Content |
| --- | --- | --- |
| `featureID` | string, `(n,)` | CpG locus ID |
| `chalm_coef` | float64, `(n, 3)` | CHALM coefficients `[a_j, b_j, c_j]` per CpG |
| `camda_coef` | float64, `(n, 3)` | CAMDA coefficients `[a_j, b_j, c_j]` per CpG |

### `A2H_cpg_info.pkl` (default) and `A2H_cpg_info.npz`

Per-CpG summaries of the reference WGBS methylation levels used in training,
for the same 1,030,653 CpGs. Both files hold identical values. As for the
coefficients, use the `.npz` file for better package compatibility and
transport.

- `A2H_cpg_info.pkl`: a pandas DataFrame indexed by CpG locus ID (index name
  `loci.hg38`), with the columns below except `featureID`.
- `A2H_cpg_info.npz`: a NumPy archive with these arrays, each of shape `(n,)`:

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
polynomial can be fitted there, so A2H returns a single constant `c_j`
(stored coefficients `[0, 0, c_j]`), estimated by kernel density, whatever the
input beta value. CHALM is 0 at 500 of these CpGs
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
