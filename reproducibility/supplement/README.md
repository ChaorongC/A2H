# Code for the supplementary A2H analyses

This folder contains the code that produced the values behind the supplementary
A2H analyses: atlas cross-validation, reference-composition refits, the PGP-UK
and GSE240482 external evaluations, and the sequencing-depth and read-length
sensitivity analysis. The code is the version that was run. Only site-specific
paths were replaced by the relative paths described below.

The figures were drawn from these outputs by co-authors, who applied additional
filters. Their plotting and filtering code is not included, so a plotted value
can differ from the unfiltered output of these scripts. Data are not included:
the atlas WGBS data are under controlled access at the EGA (Loyfer et al.,
*Nature* 613, 355-364, 2023), and the PGP-UK and GEO data are public.

The model itself (coefficients, CpG information, probe maps) is built by the
scripts in the parent folder, [`../`](../README.md).

## Layout

Run everything from this folder. The scripts expect these directories here:

| Path | Content |
| --- | --- |
| `src/`, `script/`, `tests/` | Analysis modules (run as `python -m src.<module>`), SLURM wrappers and unit tests |
| `results/` | Outputs, one folder per analysis (created by the scripts); `results/r2_2_e2e3_20261015/` also holds the manifest and summary helpers of the depth/read-length analysis |
| `collaborator_figure_data_20261017/` | Builders of the tables shared with co-authors, and `make_groups.py` + `GROUPING_SPEC.md` (matched 10-CpG groups) |
| `atlas_cv_original/` | The original repeated 10-fold atlas cross-validation |
| `external_validation/` | PGP-UK and GSE240482 array and WGBS processing; put `raw/` and `metadata/` (sample manifests) here |
| `legacy_model/` | A2H as run (see below); create with `bash make_legacy_model.sh` |
| `atlas/data/` | Merged atlas matrices from [`../01_merge_matrices.py`](../01_merge_matrices.py) |
| `work/` | Large intermediate files (BAMs, per-condition calls, staged matrices) |
| `reference/` | `hg38.fa` and the EPIC hg38 manifests |
| `tools/` | `CHALM/src/CHALM.py` ([CHALM](https://github.com/JianfengXu93/CHALM), commit `a5e2c35`) and `samtools-0.1.19` |
| `data/` | Atlas sample annotation (`atlas_sample_info.csv`) and atlas FASTQ files |

`legacy_model/` holds `A2H.py` and its pickle resources from commit `e66ac58`,
the version these analyses called. The current A2H reads `.npz` resources; the
coefficient values are identical.

## Analyses

| Analysis | Code, roughly in run order |
| --- | --- |
| External cohorts: array and WGBS processing | `external_validation/scripts/paired_validation_v2/`: `submit_array_processing.sh` (`call_array_methylation.R`), `merge_wgbs_fastqs.sh`, `submit_wgbs_processing.sh` (`process_wgbs_sample.sh`), `submit_wgbs_probe_matrices.sh` (`build_wgbs_probe_matrices.py`), `run_a2h.sh`; then `external_validation/scripts/run_wgbs_depth_threshold_outputs.sh` (depth 10) |
| Atlas cross-validation (Supp. Fig. 3, 6) | `atlas_cv_original/`; `src.make_atlas_splits`, `src.prepare_atlas_shards`, `script/run_atlas_cv_split.sub`, `script/summarize_atlas_cv.sub`, `script/site_quality_atlas_cv.sub`, `src.summarize_cv_rmse`, `src.summarize_cv_reliability`, `script/r2_2_oof_reliability.sub`, `src.r2_2_summarize_e4` |
| PGP-UK evaluation (Supp. Fig. 1) | `src.build_authoritative_pgpuk_manifest`, `src.fit_reference_controls`, `src.evaluate_pgpuk`, `src.r3_epic_participant_validation`, `collaborator_figure_data_20261017/build_package.py` (it also copies the co-authors' plotting script `make_figures.py`, which is not included), `script/r3_11_raw_matrices_20261020.sub` |
| Matched 10-CpG groups (Supp. Fig. 2) | `src.run_pgpuk_disjoint_complete10`, `src.summarize_disjoint_group_beta`, `collaborator_figure_data_20261017/raw_matrices_resp_450k_20261026/make_groups.py` (specification: `GROUPING_SPEC.md`) |
| Reference composition: excluding blood/immune or epithelial samples (Supp. Fig. 4a, 5, 6) | `src.make_transfer_manifest`, `src.prepare_transfer_external`, `script/run_transfer_shard.sub`, `script/summarize_transfer.sub`, `script/run_tissue_only_transfer.sub`, `script/summarize_tissue_only_transfer.sub`, `script/r1_6_full205_baseline_20261019.sub`, `src.build_r1_6_holdout_evidence`, `src.package_r1_6_cv_and_holdouts`, `src.build_r1_1_locus_specific_calibration`, `src.append_r1_6_whole_atlas_cv`, `src.append_r1_6_summary_historical_cv`, `src.r1_6_heldout_450k_20261027` |
| Sequencing depth and read length (Supp. Fig. 4b, c) | `src.r2_2_select_read_samples`, `script/r2_2_e2_native_first.sub` and `script/r2_2_e2_native_remaining38.sub` (native calls), `script/r2_2_e2_recovery_dfs3b.sub`, `script/r2_2_native_resort_20261015.sub`, `script/r2_2_native_finish_calls_20261015.sub`, `results/r2_2_e2e3_20261015/build_manifests.py` and `build_window_manifests.py`, `script/r2_2_window_native_20261015.sub`, `script/r2_2_e2_downsample_20261015.sub`, `script/r2_2_e2_window_20261015.sub` (depth), `script/r2_2_e3_cap_20261015.sub`, `script/r2_2_e3_window_20261015.sub` (read length), `results/r2_2_e2e3_20261015/build_summary_shards.sh`, `script/r2_2_e2e3_summarize_shard_20261016.sub`, `combine_summary_shards.py`, `condition_summary.py`, `script/r2_2_e2e3_external_20261016.sub` (39-sample refit, PGP-UK) |
| GSE240482 (prostate tumours) | `src.extend_reference_controls_epic`, `script/gse240482_external_20261017.sub`, `src.epic_controls_rerun`, `collaborator_figure_data_20261017/build_cpg_level_package.py` (`run_cpg_level.sub`) |
| Unfiltered per-individual matrices shared with co-authors | `script/raw_matrices_resp_20261023.sub`, `script/raw_matrices_resp_450k_20261026.sub`, `script/raw_matrices_resp23_450k_20261027.sub` |

Each module documents its arguments (`python -m src.<module> -h`). The SLURM resource lines are those used on the original cluster; adjust the
partition, memory and time to your site.

## Software

- Python 3.13 with numpy, pandas, scipy, scikit-learn, joblib, pyarrow, pysam,
  tqdm, matplotlib (diagnostic plots written alongside some outputs) and pytest (tested with numpy 2.4, pandas 3.0,
  scipy 1.17, scikit-learn 1.8);
- Python 3.10 for `CHALM.py`; samtools 1.10 and 1.15, samtools 0.1.19, BSMAP and
  Trim Galore 0.6.7;
- R with minfi, DMRcate and optparse for array calling.

WGBS of the external cohorts was quantified with a faster CHALM
implementation that is not distributed here (`tools/CHALM_fast/src/CHALM-fast.py`
in `process_wgbs_sample.sh`). Download scripts are not included.

## Tests

The unit tests are synthetic known-answer tests and do not need the data:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/ external_validation/tests/
```

`external_validation/tests/test_wgbs_depth_filtering.py` also needs the sample
manifests in `external_validation/metadata/`.
