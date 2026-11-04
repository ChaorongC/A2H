"""Fit reference-only controls for PGP-UK 450K external validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audit_model import ATLAS, MODEL
from src.evaluate_pgpuk import file_hash


TARGET_FILES = {
    "CHALM": "20241023_merged_CHALM_all_sample_beadchip_cpg.pkl",
    "CAMDA": "20260514_merged_CAMDA_all_sample_beadchip_cpg.pkl",
}
METH_FILE = "20241023_merged_methratio_all_sample_beadchip_cpg.pkl"


def fit_chunk(x: np.ndarray, y: np.ndarray, loci: pd.Index, min_train: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if x.shape != y.shape or x.shape[0] != len(loci):
        raise ValueError("Reference predictors and target must align by locus and sample")
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)
    eligible = n >= min_train
    xx = np.where(valid, x, 0.0)
    yy = np.where(valid, y, 0.0)
    denom = np.maximum(n, 1)
    mean_x = xx.sum(axis=1) / denom
    mean_y = yy.sum(axis=1) / denom
    centered_x = np.where(valid, x - mean_x[:, None], 0.0)
    centered_y = np.where(valid, y - mean_y[:, None], 0.0)
    sxx = np.sum(centered_x * centered_x, axis=1)
    sxy = np.sum(centered_x * centered_y, axis=1)
    slope = np.zeros(len(loci), dtype=float)
    linear_fit = eligible & (sxx > 1e-12)
    slope[linear_fit] = sxy[linear_fit] / sxx[linear_fit]
    intercept = mean_y - slope * mean_x
    minimum = np.where(valid, x, np.inf).min(axis=1)
    maximum = np.where(valid, x, -np.inf).max(axis=1)
    span = np.where(n > 0, maximum - minimum, np.nan)
    table = pd.DataFrame({
        "locus": loci,
        "n_train_joint": n,
        "x_span": span,
        "reference_mean": np.where(eligible, mean_y, np.nan),
        "linear_intercept": np.where(eligible, intercept, np.nan),
        "linear_slope": np.where(eligible, slope, np.nan),
        "linear_constant_fallback": eligible & ~linear_fit,
    })

    # Each eligible CpG contributes total weight one, regardless of its valid sample count.
    powers = [np.ones_like(xx), xx, xx * xx, xx * xx * xx, xx * xx * xx * xx]
    moments = [power.sum(axis=1) / denom for power in powers]
    moments[0] = np.ones(len(loci))
    outcomes = [(power * yy).sum(axis=1) / denom for power in powers[:3]]
    matrix = np.array([[moments[i + j][eligible].sum() for j in range(3)] for i in range(3)])
    rhs = np.array([outcome[eligible].sum() for outcome in outcomes])
    return table, matrix, rhs


def fit(atlas_root: Path, probe_map: Path, output: Path, min_train: int, chunk_size: int) -> None:
    mapping = pd.read_csv(probe_map, sep="\t", usecols=["loci.hg38", "HM450.hg38"]).dropna().drop_duplicates()
    if mapping["HM450.hg38"].duplicated().any():
        raise ValueError("One HM450 probe maps to multiple loci")
    loci = pd.Index(mapping["loci.hg38"].drop_duplicates(), name="locus")
    data_root = atlas_root / "data"
    paths = {"probe_map": probe_map, "methylation": data_root / METH_FILE}
    paths.update({target: data_root / name for target, name in TARGET_FILES.items()})
    methylation = pd.read_pickle(paths["methylation"])
    if methylation.index.has_duplicates or methylation.columns.has_duplicates or len(methylation.columns) != 205:
        raise ValueError("Expected unique loci and 205 reference samples")
    if not loci.isin(methylation.index).all():
        raise ValueError("HM450 map contains loci absent from reference methylation")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "atlas_root": str(atlas_root),
        "probe_map": str(probe_map),
        "reference_samples": methylation.columns.tolist(),
        "reference_loci": len(loci),
        "min_joint_training_observations": min_train,
        "universal_scope": "unique hg38 loci mapped to HM450, no PGP-UK outcomes",
        "universal_weight": "one per eligible locus; each valid reference pair weighted 1/n_locus",
        "file_sha256": {name: file_hash(path) for name, path in paths.items()},
    }
    for target in TARGET_FILES:
        response = pd.read_pickle(paths[target])
        if not methylation.index.equals(response.index) or not methylation.columns.equals(response.columns):
            raise ValueError(f"{target} reference matrix is not aligned with methylation")
        tables = []
        gram = np.zeros((3, 3), dtype=float)
        rhs = np.zeros(3, dtype=float)
        for start in range(0, len(loci), chunk_size):
            part = loci[start : start + chunk_size]
            table, block_gram, block_rhs = fit_chunk(
                methylation.loc[part].to_numpy(dtype=float),
                response.loc[part].to_numpy(dtype=float),
                part, min_train,
            )
            tables.append(table)
            gram += block_gram
            rhs += block_rhs
        frame = pd.concat(tables, ignore_index=True)
        if frame.locus.duplicated().any():
            raise ValueError("Duplicate fitted reference locus")
        condition = float(np.linalg.cond(gram))
        if not np.isfinite(condition) or condition > 1e12:
            raise ValueError(f"Universal quadratic design is ill-conditioned for {target}: {condition}")
        coefficients = np.linalg.solve(gram, rhs)
        frame.to_parquet(output / f"{target}_controls.parquet", index=False)
        provenance[target] = {
            "joint_eligible_loci": int(frame.reference_mean.notna().sum()),
            "linear_constant_fallback_loci": int(frame.linear_constant_fallback.sum()),
            "universal_condition_number": condition,
            "universal_coefficients_intercept_linear_quadratic": coefficients.tolist(),
        }
        del response
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ATLAS)
    parser.add_argument("--probe-map", type=Path, default=MODEL / "A2H_cpg_probe.hg38.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()
    if args.min_train < 3 or args.chunk_size < 1:
        parser.error("--min-train must be >=3 and --chunk-size must be positive")
    fit(args.atlas_root, args.probe_map, args.output_dir, args.min_train, args.chunk_size)


if __name__ == "__main__":
    main()
