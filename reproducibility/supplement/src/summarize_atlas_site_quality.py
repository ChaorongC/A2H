"""Summarize fold-local fit quality and sample-resolved CV errors by CpG shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.run_atlas_cv_split import TARGETS


def summarize_shard(shard: object, target: str, branch: str, split_root: Path,
                    repeat_root: Path, split_count: int) -> pd.DataFrame:
    loci = pd.read_parquet(shard.path, columns=["locus"]).locus.to_numpy()
    train_count = []
    unique_count = []
    x_variance = []
    status = []
    for split_index in range(split_count):
        path = split_root / f"{branch}_split{split_index:03d}" / f"{target}_shard{shard.shard_index:04d}_quality.parquet"
        frame = pd.read_parquet(path)
        if not np.array_equal(frame.locus.to_numpy(), loci) or not (frame.target == target).all() or not (frame.split_index == split_index).all():
            raise ValueError(f"Fold quality key mismatch: {path}")
        train_count.append(frame.n_train_joint.to_numpy(dtype=int))
        unique_count.append(frame.unique_input_count.to_numpy(dtype=int))
        x_variance.append(frame.train_x_variance.to_numpy(dtype=float))
        status.append(frame.fit_status.to_numpy())
    train_count = np.stack(train_count)
    unique_count = np.stack(unique_count)
    x_variance = np.stack(x_variance)
    status = np.stack(status)
    rmse = []
    spearman = []
    heldout_n = []
    for repeat in range(split_count // 10):
        path = repeat_root / f"{target}_repeat{repeat:02d}_shard{shard.shard_index:04d}_A2H.parquet"
        frame = pd.read_parquet(path)
        if not np.array_equal(frame.locus.to_numpy(), loci) or not (frame.target == target).all() or not (frame.repeat == repeat).all():
            raise ValueError(f"Repeat metric key mismatch: {path}")
        rmse.append(frame.rmse.to_numpy(dtype=float))
        spearman.append(frame.spearman_rho.to_numpy(dtype=float))
        heldout_n.append(frame.n.to_numpy(dtype=int))
    rmse = np.stack(rmse)
    spearman = np.stack(spearman)
    heldout_n = np.stack(heldout_n)
    valid_rho = np.isfinite(spearman)
    return pd.DataFrame({
        "locus": loci,
        "target": target,
        "branch": branch,
        "n_training_folds": split_count,
        "min_train_joint": train_count.min(axis=0),
        "mean_train_joint": train_count.mean(axis=0),
        "max_train_joint": train_count.max(axis=0),
        "min_unique_input": unique_count.min(axis=0),
        "median_train_x_variance": np.median(x_variance, axis=0),
        "n_quadratic_folds": (status == "quadratic").sum(axis=0),
        "n_constant_kde_folds": (status == "constant_kde").sum(axis=0),
        "n_rank_deficient_folds": (status == "rank_deficient").sum(axis=0),
        "n_insufficient_folds": (status == "insufficient").sum(axis=0),
        "n_repeats_with_observed_pairs": (heldout_n > 0).sum(axis=0),
        "n_repeats_with_defined_spearman": valid_rho.sum(axis=0),
        "mean_repeat_rmse": pd.DataFrame(rmse).mean(axis=0).to_numpy(),
        "median_repeat_spearman": pd.DataFrame(spearman).median(axis=0).to_numpy(),
    })


def run(shard_dir: Path, split_root: Path, repeat_root: Path, output_dir: Path,
        branch: str, split_count: int) -> None:
    if split_count < 10 or split_count % 10:
        raise ValueError("Require complete ten-fold repeats")
    completion_path = repeat_root / "completion.json"
    completion = json.loads(completion_path.read_text())
    if completion["branch"] != branch or completion["split_count"] != split_count or not completion["aggregation_complete"]:
        raise ValueError("Atlas repeat aggregation is incomplete")
    shards_path = shard_dir / "shards.tsv"
    shards = pd.read_csv(shards_path, sep="\t")
    if shards.shard_index.duplicated().any() or int(shards.n_loci.sum()) != completion["n_loci"]:
        raise ValueError("CpG shard manifest differs from completed atlas CV")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in shards.itertuples(index=False):
        for target in TARGETS:
            table = summarize_shard(shard, target, branch, split_root, repeat_root, split_count)
            path = (output_dir / f"{target}_shard{shard.shard_index:04d}_site_quality.parquet").resolve()
            table.to_parquet(path, index=False, compression="zstd")
            rows.append({"target": target, "shard_index": shard.shard_index, "n_loci": len(table),
                         "path": str(path), "sha256": file_hash(path),
                         "n_sites_quadratic_all_folds": int((table.n_quadratic_folds == split_count).sum()),
                         "n_sites_insufficient_any_fold": int((table.n_insufficient_folds > 0).sum())})
    pd.DataFrame(rows).to_csv(output_dir / "site_quality_shards.tsv", sep="\t", index=False)
    result = {"branch": branch, "split_count": split_count, "n_loci": int(shards.n_loci.sum()),
              "n_output_shards": len(rows), "source_sha256": {
                  "shards": file_hash(shards_path), "repeat_completion": file_hash(completion_path)}}
    (output_dir / "completion.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=["historical_sample", "donor_grouped"], required=True)
    parser.add_argument("--split-count", type=int, default=100)
    args = parser.parse_args()
    run(args.shard_dir, args.split_root, args.repeat_root, args.output_dir,
        args.branch, args.split_count)


if __name__ == "__main__":
    main()
