"""Verify split artifacts and aggregate sample-resolved atlas CV by repeat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash, row_stats, spearman_rows
from src.run_atlas_cv_split import TARGETS, split_members


METHODS = ("A2H", "reference_mean", "reference_linear", "universal_quadratic", "m", "m2", "m3")


def verify_completion(path: Path, branch: str, split_index: int,
                      shards: pd.DataFrame, manifest: pd.DataFrame,
                      sample_order: list[str], input_hashes: dict[str, str]) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    result = json.loads(path.read_text())
    train, test, repeat, fold = split_members(manifest, branch, split_index, sample_order)
    if (result["branch"] != branch or result["split_index"] != split_index or
            result["repeat"] != repeat or result["fold"] != fold or
            result["train_ids"] != train.index.tolist() or result["test_ids"] != test.index.tolist() or
            result["shards"] != shards.shard_index.tolist() or result["input_sha256"] != input_hashes):
        raise ValueError(f"Split completion manifest mismatch: {path}")
    expected = {(target, int(shard)) for target in TARGETS for shard in shards.shard_index}
    found = {(row["target"], row["shard_index"]) for row in result["outputs"]}
    if found != expected or len(result["outputs"]) != len(expected):
        raise ValueError(f"Incomplete target-shard outputs: {path}")
    for row in result["outputs"]:
        shard = shards.set_index("shard_index").loc[row["shard_index"]]
        if row["n_loci"] != shard.n_loci or row["n_prediction_rows"] != shard.n_loci * len(test):
            raise ValueError(f"Unexpected row count: {path}")
        for kind in ("quality", "prediction"):
            artifact = Path(row[f"{kind}_path"])
            if not artifact.exists() or file_hash(artifact) != row[f"{kind}_sha256"]:
                raise ValueError(f"Missing or changed CV artifact: {artifact}")
    return result


def summarize_repeat(shards: pd.DataFrame, target: str, repeat: int,
                     completions: dict[int, dict], sample_order: list[str],
                     output_dir: Path) -> pd.DataFrame:
    summary_rows = []
    sample_positions = {sample: i for i, sample in enumerate(sample_order)}
    for shard in shards.itertuples(index=False):
        loci = pd.read_parquet(shard.path, columns=["locus"]).locus.to_numpy()
        n_loci = len(loci)
        shape = (n_loci, len(sample_order))
        observed = np.full(shape, np.nan)
        predicted = {method: np.full(shape, np.nan) for method in METHODS}
        fit_status = np.empty(shape, dtype=object)
        filled = np.zeros(len(sample_order), dtype=bool)
        for split_index, completion in completions.items():
            if completion["repeat"] != repeat:
                continue
            test_ids = completion["test_ids"]
            positions = [sample_positions[sample] for sample in test_ids]
            if filled[positions].any():
                raise ValueError(f"Duplicate held-out sample in repeat {repeat}")
            record = next(row for row in completion["outputs"] if row["target"] == target and row["shard_index"] == shard.shard_index)
            frame = pd.read_parquet(record["prediction_path"])
            if (len(frame) != n_loci * len(test_ids) or
                    not np.array_equal(frame.locus.to_numpy(), np.repeat(loci, len(test_ids))) or
                    not np.array_equal(frame.sample_id.to_numpy(), np.tile(test_ids, n_loci)) or
                    not (frame.target == target).all() or
                    not (frame.repeat == repeat).all() or
                    not (frame.split_index == split_index).all()):
                raise ValueError(f"Prediction key or order mismatch: {record['prediction_path']}")
            observed[:, positions] = frame.observed.to_numpy(dtype=float).reshape(n_loci, len(test_ids))
            fit_status[:, positions] = frame.fit_status.to_numpy().reshape(n_loci, len(test_ids))
            for method in METHODS:
                predicted[method][:, positions] = frame[method].to_numpy(dtype=float).reshape(n_loci, len(test_ids))
            filled[positions] = True
        if not filled.all():
            raise ValueError(f"Incomplete held-out sample coverage in repeat {repeat}")
        common = np.isfinite(observed)
        for value in predicted.values():
            common &= np.isfinite(value)
        truth = np.where(common, observed, np.nan)
        status = pd.DataFrame({
            "locus": loci, "target": target, "repeat": repeat,
            "n_heldout_common": common.sum(axis=1),
            "n_heldout_truth": np.isfinite(observed).sum(axis=1),
            "n_constant_fallback_predictions": (fit_status == "constant_kde").sum(axis=1),
            "n_rank_deficient_predictions": (fit_status == "rank_deficient").sum(axis=1),
            "n_insufficient_predictions": (fit_status == "insufficient").sum(axis=1),
        })
        status.to_parquet(output_dir / f"{target}_repeat{repeat:02d}_shard{shard.shard_index:04d}_site_quality.parquet",
                          index=False, compression="zstd")
        for method, value in predicted.items():
            comparison = np.where(common, value, np.nan)
            metrics = row_stats(truth, comparison)
            scores = pd.DataFrame({
                "locus": loci, "target": target, "repeat": repeat, "method": method,
                "n": metrics["n"], "pearson_r": metrics["pearson_r"],
                "spearman_rho": spearman_rows(truth, comparison),
                "rmse": metrics["rmse"], "bias": metrics["bias"],
            })
            scores.to_parquet(output_dir / f"{target}_repeat{repeat:02d}_shard{shard.shard_index:04d}_{method}.parquet",
                              index=False, compression="zstd")
            eligible = scores.n >= 2
            summary_rows.append({
                "target": target, "repeat": repeat, "shard_index": int(shard.shard_index),
                "method": method, "n_loci": n_loci, "n_sites_two_or_more": int(eligible.sum()),
                "n_site_subject_observations": int(scores.n.sum()),
                "mean_site_rmse": float(scores.rmse[eligible].mean()),
                "median_site_spearman": float(scores.spearman_rho[eligible].median()),
            })
    return pd.DataFrame(summary_rows)


def run(shard_dir: Path, split_dir: Path, split_root: Path, output_dir: Path,
        branch: str, split_count: int, verify_only: bool = False) -> None:
    if split_count < 1 or split_count > 100:
        raise ValueError("split_count must be 1-100")
    shards_path = shard_dir / "shards.tsv"
    sample_path = split_dir / "atlas_samples.tsv"
    split_path = split_dir / "splits.tsv"
    shards = pd.read_csv(shards_path, sep="\t")
    samples = pd.read_csv(sample_path, sep="\t").sample_id.tolist()
    manifest = pd.read_csv(split_path, sep="\t")
    hashes = {"shards": file_hash(shards_path), "samples": file_hash(sample_path), "splits": file_hash(split_path)}
    completions = {}
    for split_index in range(split_count):
        path = split_root / f"{branch}_split{split_index:03d}" / "completion.json"
        completions[split_index] = verify_completion(path, branch, split_index, shards, manifest, samples, hashes)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not verify_only:
        if split_count % 10:
            raise ValueError("Aggregation requires complete ten-fold repeats")
        summaries = []
        for repeat in range(split_count // 10):
            for target in TARGETS:
                summaries.append(summarize_repeat(shards, target, repeat, completions, samples, output_dir))
        pd.concat(summaries, ignore_index=True).to_csv(output_dir / "repeat_shard_summary.tsv", sep="\t", index=False)
    result = {"branch": branch, "split_count": split_count, "verified_splits": sorted(completions),
              "n_shards": len(shards), "n_loci": int(shards.n_loci.sum()),
              "aggregation_complete": not verify_only, "input_sha256": hashes}
    (output_dir / "completion.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=["historical_sample", "donor_grouped"], required=True)
    parser.add_argument("--split-count", type=int, default=100)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run(args.shard_dir, args.split_dir, args.split_root, args.output_dir,
        args.branch, args.split_count, args.verify_only)


if __name__ == "__main__":
    main()
