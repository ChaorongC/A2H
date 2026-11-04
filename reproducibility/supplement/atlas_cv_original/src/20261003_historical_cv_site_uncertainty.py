#!/usr/bin/env python
"""Derive site-level uncertainty summaries from the historical EPIC CV outputs.

The source CV products are the historical >=4x atlas runs.  Each raw EPIC
table has 100 repeated-fold ``true``/``pred`` column pairs.  This script
summarizes those fold-level, held-out CpG means separately for CHALM and the
independent CAMDA branch.  It does not rerun A2H and does not use the derived
CHALM-to-CAMDA notebook branch.

The reported refit standard deviations and empirical error quantiles describe
variation across the 100 refits.  They are not calibrated prediction
intervals for a new subject.

The CAMDA pickle was written with a newer pandas/Arrow-backed serialization;
run this script with the project environment that can read that pickle (for
example pandas 3.x with pyarrow available).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CV_SPEC = {
    "CHALM": {
        "raw": "20260310_100times_cross_validation_205_nature_atlas_samples_epic_cpg.pkl",
        "mean": "20260310_100times_cross_validation_205_nature_atlas_samples_epic_cpg_mean.pkl",
        "matrix": "20241023_merged_CHALM_all_sample_beadchip_cpg.pkl",
    },
    "CAMDA": {
        "raw": "20260516_100times_cross_validation_205_nature_atlas_samples_epic_cpg_camda.pkl",
        "mean": "20260516_100times_cross_validation_205_nature_atlas_samples_epic_cpg_mean_camda.pkl",
        "matrix": "20260514_merged_CAMDA_all_sample_beadchip_cpg.pkl",
    },
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_pearson(true: np.ndarray, pred: np.ndarray) -> float:
    if len(true) < 2 or np.std(true) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(true, pred)[0, 1])


def summarize_method(
    method: str,
    raw_path: Path,
    mean_path: Path,
    manifest_ids: pd.Series,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = pd.read_pickle(raw_path)
    stored_mean = pd.read_pickle(mean_path)

    expected_ids = manifest_ids.to_numpy(dtype=object)
    observed_ids = raw.index.to_numpy(dtype=object)
    if not np.array_equal(observed_ids, expected_ids):
        mismatch = np.flatnonzero(observed_ids != expected_ids)
        first = int(mismatch[0]) if len(mismatch) else None
        detail = ""
        if first is not None:
            detail = (
                f" first mismatch at row {first}: "
                f"observed={observed_ids[first]!r}, expected={expected_ids[first]!r}"
            )
        raise ValueError(
            f"{method}: raw CV row order/IDs do not exactly match the EPIC manifest;"
            + detail
        )

    if raw.shape[1] != 200 or list(raw.columns).count("true") != 100:
        raise ValueError(f"{method}: expected 100 true/pred fold pairs, got {raw.shape}")

    true = raw.iloc[:, 0::2].to_numpy(dtype=np.float64)
    pred = raw.iloc[:, 1::2].to_numpy(dtype=np.float64)
    true_valid = np.isfinite(true)
    pred_valid = np.isfinite(pred)
    pair_valid = true_valid & pred_valid
    n_valid = pair_valid.sum(axis=1)
    any_valid = n_valid > 0

    if not np.array_equal(true_valid, pred_valid):
        raise ValueError(f"{method}: true/pred missingness differs within a fold")

    def pair_mean(values: np.ndarray) -> np.ndarray:
        numerator = np.where(pair_valid, values, 0.0).sum(axis=1)
        return np.divide(
            numerator,
            n_valid,
            out=np.full(len(n_valid), np.nan),
            where=n_valid > 0,
        )

    mean_true = pair_mean(true)
    mean_pred = pair_mean(pred)
    errors = pred - true
    mean_error = pair_mean(errors)
    mse = pair_mean(errors**2)
    mae = pair_mean(np.abs(errors))

    def sample_sd(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
        squared = np.where(pair_valid, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
        denominator = np.maximum(n_valid - 1, 1)
        result = np.sqrt(squared / denominator)
        result[n_valid < 2] = np.nan
        return result

    refit_prediction_sd = sample_sd(pred, mean_pred)
    refit_error_sd = sample_sd(errors, mean_error)
    heldout_true_mean_sd = sample_sd(true, mean_true)

    valid_rows = np.flatnonzero(any_valid)
    metrics = pd.DataFrame(
        {
            "epic_manifest_row": valid_rows,
            "cpg_id": raw.index.to_numpy(dtype=object)[valid_rows],
            "n_valid_folds": n_valid[valid_rows],
            "mean_true": mean_true[valid_rows],
            "mean_pred": mean_pred[valid_rows],
            "bias_pred_minus_true": mean_error[valid_rows],
            "site_cv_rmse": np.sqrt(mse[valid_rows]),
            "site_cv_mae": mae[valid_rows],
            "refit_prediction_sd": refit_prediction_sd[valid_rows],
            "refit_error_sd": refit_error_sd[valid_rows],
            "heldout_true_mean_sd": heldout_true_mean_sd[valid_rows],
            "error_q025": np.nanquantile(errors[valid_rows], 0.025, axis=1),
            "error_median": np.nanquantile(errors[valid_rows], 0.5, axis=1),
            "error_q975": np.nanquantile(errors[valid_rows], 0.975, axis=1),
        }
    )

    stored_valid = stored_mean.dropna()
    if not stored_valid.index.equals(raw.index[valid_rows]):
        raise ValueError(f"{method}: stored mean output does not preserve valid-row order")
    if not np.allclose(stored_valid["true"].to_numpy(), metrics["mean_true"]):
        raise ValueError(f"{method}: stored true means do not match raw fold means")
    if not np.allclose(stored_valid["pred"].to_numpy(), metrics["mean_pred"]):
        raise ValueError(f"{method}: stored predicted means do not match raw fold means")

    fold_rows = []
    for fold_id in range(true.shape[1]):
        fold_valid = pair_valid[:, fold_id]
        fold_true = true[fold_valid, fold_id]
        fold_pred = pred[fold_valid, fold_id]
        fold_error = fold_pred - fold_true
        fold_rows.append(
            {
                "method": method,
                "fold_id": fold_id,
                "n_epic_rows": int(fold_valid.sum()),
                "rmse": float(np.sqrt(np.mean(fold_error**2))),
                "pearson_r": safe_pearson(fold_true, fold_pred),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)

    aggregate_true = metrics["mean_true"].to_numpy()
    aggregate_pred = metrics["mean_pred"].to_numpy()
    aggregate_error = aggregate_pred - aggregate_true
    summary = {
        "method": method,
        "source_training_depth": ">=4x",
        "source_cv": "RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)",
        "n_samples": 205,
        "n_repeated_folds": 100,
        "n_epic_manifest_rows": int(len(raw)),
        "n_epic_unique_ids_in_input": int(pd.Index(raw.index).nunique()),
        "n_epic_valid_rows": int(len(metrics)),
        "n_epic_valid_unique_ids": int(metrics["cpg_id"].nunique()),
        "n_valid_fold_pairs": int(n_valid[valid_rows].sum()),
        "min_valid_folds_per_site": int(n_valid[valid_rows].min()),
        "max_valid_folds_per_site": int(n_valid[valid_rows].max()),
        "aggregate_mean_mse": float(np.mean(aggregate_error**2)),
        "aggregate_mean_rmse": float(np.sqrt(np.mean(aggregate_error**2))),
        "aggregate_mean_pearson_r": safe_pearson(aggregate_true, aggregate_pred),
        "fold_rmse_mean": float(fold_metrics["rmse"].mean()),
        "fold_rmse_sd": float(fold_metrics["rmse"].std(ddof=1)),
        "fold_rmse_min": float(fold_metrics["rmse"].min()),
        "fold_rmse_max": float(fold_metrics["rmse"].max()),
        "fold_pearson_mean": float(fold_metrics["pearson_r"].mean()),
        "fold_pearson_sd": float(fold_metrics["pearson_r"].std(ddof=1)),
        "fold_pearson_min": float(fold_metrics["pearson_r"].min()),
        "fold_pearson_max": float(fold_metrics["pearson_r"].max()),
        "within_pair_missingness_identical": True,
        "raw_row_order_matches_epic_manifest": True,
        "epic_coordinate_convention": "CpG_chrm + '_' + str(CpG_beg + 1)",
        "site_metric_unit": "one EPIC manifest row; fold-level held-out sample mean",
        "uncertainty_interpretation": "fold-to-fold spread combines refitting and held-out-fold composition; not a calibrated new-subject prediction interval",
    }
    return metrics, summary, fold_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reference/EPIC.hg38.manifest.tsv"),
    )
    args = parser.parse_args()
    result_dir = args.project_root / "atlas_cv_original/result"
    data_dir = args.project_root / "atlas/data"
    result_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, sep="\t", usecols=["CpG_chrm", "CpG_beg"]).dropna()
    manifest_ids = manifest["CpG_chrm"] + "_" + (manifest["CpG_beg"] + 1).astype(int).astype(str)

    all_summaries = []
    all_fold_metrics = []
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": sha256_file(args.manifest),
            "n_rows": int(len(manifest_ids)),
            "n_unique_ids": int(manifest_ids.nunique()),
            "coordinate_convention": "CpG_chrm + '_' + str(CpG_beg + 1)",
        },
        "methods": {},
        "scope": {
            "training_depth": ">=4x historical atlas branch",
            "feature_scope": "EPIC/BeadChip CpG sites only",
            "constant_sites": "retained; no additional site filtering",
            "external_depth": "not applicable; this script summarizes atlas CV only",
        },
    }

    for method, spec in CV_SPEC.items():
        raw_path = result_dir / spec["raw"]
        mean_path = result_dir / spec["mean"]
        matrix_path = data_dir / spec["matrix"]
        metrics, summary, fold_metrics = summarize_method(method, raw_path, mean_path, manifest_ids)
        output_path = result_dir / f"20261003_historical_cv_site_metrics_epic_cpg_{method.lower()}.pkl"
        metrics.to_pickle(output_path, protocol=5)
        all_summaries.append(summary)
        all_fold_metrics.append(fold_metrics)
        provenance["methods"][method] = {
            "raw_cv_path": str(raw_path.resolve()),
            "raw_cv_sha256": sha256_file(raw_path),
            "stored_mean_path": str(mean_path.resolve()),
            "stored_mean_sha256": sha256_file(mean_path),
            "training_matrix_path": str(matrix_path.resolve()),
            "training_matrix_sha256": sha256_file(matrix_path),
            "site_metrics_path": str(output_path.resolve()),
            "summary": summary,
        }

    pd.DataFrame(all_summaries).to_csv(
        result_dir / "20261003_historical_cv_epic_cpg_summary.tsv",
        sep="\t",
        index=False,
    )
    pd.concat(all_fold_metrics, ignore_index=True).to_csv(
        result_dir / "20261003_historical_cv_fold_metrics_epic_cpg.tsv",
        sep="\t",
        index=False,
    )
    with (result_dir / "20261003_historical_cv_site_metrics_epic_cpg_provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
