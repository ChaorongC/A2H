# Reproducing the A2H Model Resources

This folder contains the code that trains the A2H model and builds the bundled
resource files of the parent directory:

- `A2H_coef.npz`
- `A2H_cpg_info.npz`
- `A2H_constant_cpgs.tsv`
- `A2H_cpg_probe.hg38.tsv`, `A2H_cpg_probe.hg19.tsv`

The model is trained on the 205 whole-genome bisulfite sequencing (WGBS)
samples of the human methylome atlas (Loyfer et al., *Nature* 613, 355-364,
2023, [doi:10.1038/s41586-022-05580-6](https://doi.org/10.1038/s41586-022-05580-6)).
The raw sequencing data are available under controlled access from the European
Genome-phenome Archive (EGA), as described in that publication.

The code for the supplementary analyses (cross-validation, external
evaluation, sensitivity analyses) is in [`supplement/`](supplement/README.md).

## Pipeline

| Step | Script | Input | Output |
| --- | --- | --- | --- |
| 0 | `00_process_wgbs.sh` | Raw FASTQ files, one sample per run | `<sample_id>_methRatio.txt`, `<sample_id>_CHALM.txt` |
| 1 | `01_merge_matrices.py` | Step 0 outputs of all 205 samples, array manifests | `merged_methratio.pkl`, `merged_CHALM.pkl` |
| 2 | `02_fit_coefficients.py` | Step 1 matrices | `chalm_coef.pkl`, `camda_coef.pkl`, `chalm_fit_info.tsv`, `camda_fit_info.tsv` |
| 3 | `03_cpg_info.py` | Step 1 matrices | `cpg_info.pkl` |
| 4 | `04_probe_maps.py` | `chalm_coef.pkl`, array manifests | `A2H_cpg_probe.hg38.tsv`, `A2H_cpg_probe.hg19.tsv` |
| 5 | `05_build_resources.py` | Steps 2-4 outputs | `A2H_coef.npz`, `A2H_cpg_info.npz`, `A2H_constant_cpgs.tsv` |

Each Python script documents its arguments (`python <script> -h`).

### Step 0: WGBS processing

Run once per sample:

```bash
bash 00_process_wgbs.sh <sample_id> <fastq_dir> <out_dir> hg38.fa <CHALM/src> <samtools-0.1.19> 40
```

1. The two sequencing lanes are concatenated.
2. Reads are trimmed with Trim Galore 0.6.7 (`--paired --2colour 20 --clip_R1 10
   --clip_R2 20 --three_prime_clip_R1 10 --three_prime_clip_R2 10`).
3. Reads are aligned to hg38 with [BASAL](https://github.com/JiejunShi/BASAL)
   v1.8 (`-M C:T -R`, otherwise default parameters).
4. Alignments are filtered with `samtools view -F 268`, then sorted and indexed
   (samtools 1.10).
5. The methylation level and CHALM of each CpG are computed with `CHALM.py trad`
   from [CHALM](https://github.com/JianfengXu93/CHALM) (commit `a5e2c35`), which
   uses samtools 0.1.19. Options: `-x CG -i no-action -m 4 -p -r -u -g`. CpGs
   with fewer than 4 reads are not reported. CHALM additionally uses `-l 1`: a
   read counts as methylated if at least one of its CpGs is methylated.

### Step 1: merge matrices

```bash
python 01_merge_matrices.py \
    --chalm-dir <out_dir>/3.chalm --methratio-dir <out_dir>/3.methratio \
    --manifest-dir <manifests> --out-dir work
```

Merges every sample in `--chalm-dir` (or the sample IDs listed, one per line, in
the optional `--samples` file). Keeps the CpGs targeted by any of the HM450, EPIC, EPIC+ and EPIC v2 arrays,
taken from the hg38 manifests (`<platform>.hg38.manifest.tsv`, with columns
`Probe_ID`, `CpG_chrm`, `CpG_beg`, `CpG_end`). CpGs on chr1-22, X, Y and M are
kept. A CpG is identified as `<chrom>_<position>`, the 1-based position of its
cytosine (`CpG_beg + 1`).

### Step 2: fit the A2H polynomials

```bash
python 02_fit_coefficients.py --chalm work/merged_CHALM.pkl \
    --methratio work/merged_methratio.pkl --out-dir work
```

The targets are CHALM, and CAMDA = CHALM - methylation level. For each CpG and
target, the target is regressed on the methylation level across samples with a
degree-2 polynomial (`numpy.polyfit`), using the samples where both values are
present.

- **CpG filter:** a target is fitted at CpGs with at least 50 samples with a
  value for that target (`--min-obs 50`).
- **Constant input:** if the methylation level is identical in every sample used,
  no polynomial can be fitted. The coefficients are then `[0, 0, c]`, where `c`
  is the argmax of a Gaussian kernel density estimate of the target values
  (bandwidth 0.05), evaluated on 10 evenly spaced points in [0, 1].

### Step 3: CpG information

```bash
python 03_cpg_info.py --chalm work/merged_CHALM.pkl \
    --methratio work/merged_methratio.pkl --out work/cpg_info.pkl
```

For the CpGs fitted for CHALM, computes the minimum, maximum, mean and standard
deviation of the methylation level across samples. It also fits a beta-binomial
distribution (n = 100) to the methylation levels expressed as integer
percentages (`int(100 * level)`), by maximum likelihood with Nelder-Mead,
starting from a = b = mean percentage. A2H uses the fitted `betabinom_a` and
`betabinom_b` to compute confidence levels.

### Step 4: probe maps

```bash
python 04_probe_maps.py --chalm-coef work/chalm_coef.pkl \
    --manifest-dir <manifests> --out-dir work
```

Attaches the probe IDs of each platform to the fitted CpGs:

- **hg38 map:** EPIC v2, EPIC+, EPIC and HM450, from the hg38 manifests.
- **hg19 map:** EPIC, EPIC v2 and HM450, from the hg19 manifests. Positions are
  converted to hg38 with the [`liftover`](https://pypi.org/project/liftover/)
  package; probes that cannot be converted are left out.

### Step 5: build the resource files

```bash
python 05_build_resources.py --fit-dir work --cpg-info work/cpg_info.pkl \
    --probe-map work/A2H_cpg_probe.hg38.tsv --out-dir work
```

- The CpGs are those fitted for CHALM, sorted by locus ID. CAMDA coefficients
  and CpG information are taken for the same CpGs.
- `A2H_constant_cpgs.tsv` lists the CpGs where the CHALM fit used the
  constant-input fallback, with the number of samples used, their probe IDs and
  the constant A2H output.

## Software

Steps 1-5 need Python 3 with `numpy`, `pandas`, `scipy`, `scikit-learn`,
`joblib` and, for step 4, `liftover`:

```bash
pip install numpy pandas scipy scikit-learn joblib liftover
```

Step 0 needs Trim Galore 0.6.7 (with cutadapt), BASAL 1.8, samtools 1.10, samtools
0.1.19 and CHALM. Steps 1-3 load all 205 samples into memory; about 100 GB of
RAM is enough.
