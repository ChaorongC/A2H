"""Run one sample-resolved atlas CV split over prepared CpG shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.fit_atlas_fold import fit_sites
from src.fit_reference_controls import fit_chunk


TARGETS = ("CHALM", "CAMDA")


def split_members(manifest: pd.DataFrame, branch: str, split_index: int,
                  sample_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    part = manifest.loc[(manifest.branch == branch) & (manifest.split_index == split_index)]
    if part.empty or part.duplicated(["role", "sample_id"]).any():
        raise ValueError("Split is missing or has duplicate role-sample keys")
    if not part.role.isin(["train", "test"]).all():
        raise ValueError("Split has an unexpected role")
    train = part.loc[part.role == "train"].set_index("sample_id")
    test = part.loc[part.role == "test"].set_index("sample_id")
    if set(train.index) | set(test.index) != set(sample_order) or set(train.index) & set(test.index):
        raise ValueError("Split does not partition the reference samples")
    if branch == "donor_grouped" and set(train.donor_id) & set(test.donor_id):
        raise ValueError("Grouped split shares donors")
    if part.repeat.nunique() != 1 or part.fold.nunique() != 1:
        raise ValueError("Split has inconsistent repeat/fold labels")
    return train, test, int(part.repeat.iloc[0]), int(part.fold.iloc[0])


def shard_arrays(frame: pd.DataFrame, target: str, train_ids: list[str],
                 test_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame[[f"m__{s}" for s in train_ids]].to_numpy(dtype=float),
        frame[[f"{target}__{s}" for s in train_ids]].to_numpy(dtype=float),
        frame[[f"m__{s}" for s in test_ids]].to_numpy(dtype=float),
        frame[[f"{target}__{s}" for s in test_ids]].to_numpy(dtype=float),
    )


def universal_coefficients(shards: pd.DataFrame, target: str, train_ids: list[str],
                           min_train: int) -> tuple[np.ndarray, int, float]:
    gram = np.zeros((3, 3), dtype=float)
    rhs = np.zeros(3, dtype=float)
    count = 0
    for path in shards.path:
        columns = ["locus"] + [f"m__{s}" for s in train_ids] + [f"{target}__{s}" for s in train_ids]
        frame = pd.read_parquet(path, columns=columns)
        x, y, _, _ = shard_arrays(frame, target, train_ids, [])
        summary, block_gram, block_rhs = fit_chunk(x, y, pd.Index(frame.locus), min_train)
        gram += block_gram
        rhs += block_rhs
        count += int(summary.reference_mean.notna().sum())
    condition = float(np.linalg.cond(gram))
    if count == 0 or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f"Universal {target} fit has no support or is ill-conditioned: {condition}")
    return np.linalg.solve(gram, rhs), count, condition


def score_shard(frame: pd.DataFrame, target: str, train_ids: list[str],
                test_ids: list[str], coefficients: np.ndarray, min_train: int,
                split_index: int, repeat: int, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.locus.duplicated().any():
        raise ValueError("Duplicate locus in CV shard")
    x, y, x_test, truth = shard_arrays(frame, target, train_ids, test_ids)
    quality, predictions = fit_sites(x, y, x_test, min_train=min_train)
    predictions["universal_quadratic"] = (
        coefficients[0] + coefficients[1] * x_test + coefficients[2] * x_test**2
    )
    quality.insert(0, "locus", frame.locus.to_numpy())
    quality.insert(1, "target", target)
    quality.insert(2, "split_index", split_index)
    quality.insert(3, "repeat", repeat)
    quality.insert(4, "fold", fold)
    rows = {
        "locus": np.repeat(frame.locus.to_numpy(), len(test_ids)),
        "sample_id": np.tile(test_ids, len(frame)),
        "target": target,
        "split_index": split_index,
        "repeat": repeat,
        "fold": fold,
        "observed": truth.ravel(),
        "methylation": x_test.ravel(),
        "fit_status": np.repeat(quality.fit_status.to_numpy(), len(test_ids)),
    }
    rows.update({method: value.ravel() for method, value in predictions.items()})
    observed = pd.DataFrame(rows)
    if observed.duplicated(["locus", "sample_id", "target", "repeat"]).any():
        raise ValueError("Duplicate CV prediction key")
    return quality, observed


def run(shard_dir: Path, split_dir: Path, output_dir: Path,
        branch: str, split_index: int, min_train: int,
        shard_index: int | None = None) -> None:
    if min_train < 3:
        raise ValueError("Minimum joint training count must be >=3")
    shards_path = shard_dir / "shards.tsv"
    shards = pd.read_csv(shards_path, sep="\t")
    if shard_index is not None:
        shards = shards.loc[shards.shard_index == shard_index]
    if shards.empty or shards.shard_index.duplicated().any():
        raise ValueError("Missing or duplicate CpG shard")
    sample_path = split_dir / "atlas_samples.tsv"
    split_path = split_dir / "splits.tsv"
    samples = pd.read_csv(sample_path, sep="\t")
    manifest = pd.read_csv(split_path, sep="\t")
    train, test, repeat, fold = split_members(manifest, branch, split_index, samples.sample_id.tolist())
    train_ids = train.index.tolist()
    test_ids = test.index.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "branch": branch, "split_index": split_index, "repeat": repeat, "fold": fold,
        "min_joint_train": min_train, "train_ids": train_ids, "test_ids": test_ids,
        "shards": shards.shard_index.tolist(),
        "input_sha256": {"shards": file_hash(shards_path), "samples": file_hash(sample_path),
                         "splits": file_hash(split_path)},
        "universal": {}, "outputs": [],
    }
    for target in TARGETS:
        coefficients, count, condition = universal_coefficients(shards, target, train_ids, min_train)
        result["universal"][target] = {"coefficients_intercept_linear_quadratic": coefficients.tolist(),
                                       "eligible_loci": count, "condition_number": condition}
        for row in shards.itertuples(index=False):
            frame = pd.read_parquet(row.path)
            if len(frame) != row.n_loci or frame.locus.iloc[0] != row.first_locus or frame.locus.iloc[-1] != row.last_locus:
                raise ValueError(f"Shard index mismatch: {row.path}")
            quality, observed = score_shard(frame, target, train_ids, test_ids, coefficients,
                                            min_train, split_index, repeat, fold)
            prefix = f"{target}_shard{row.shard_index:04d}"
            quality_path = (output_dir / f"{prefix}_quality.parquet").resolve()
            prediction_path = (output_dir / f"{prefix}_predictions.parquet").resolve()
            quality.to_parquet(quality_path, index=False, compression="zstd")
            observed.to_parquet(prediction_path, index=False, compression="zstd")
            result["outputs"].append({
                "target": target, "shard_index": int(row.shard_index),
                "n_loci": len(quality), "n_prediction_rows": len(observed),
                "fit_status_counts": quality.fit_status.value_counts().to_dict(),
                "quality_path": str(quality_path), "prediction_path": str(prediction_path),
                "quality_sha256": file_hash(quality_path), "prediction_sha256": file_hash(prediction_path),
            })
    (output_dir / "completion.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=["historical_sample", "donor_grouped"], required=True)
    parser.add_argument("--split-index", type=int, required=True)
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    run(args.shard_dir, args.split_dir, args.output_dir,
        args.branch, args.split_index, args.min_train, args.shard_index)


if __name__ == "__main__":
    main()
