"""Verify complete atlas CV shard summaries and aggregate equal-CpG mean RMSE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.summarize_atlas_cv import METHODS
from src.run_atlas_cv_split import TARGETS


def aggregate(frame: pd.DataFrame, shards: pd.DataFrame,
              completion: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeats = completion["split_count"] // 10
    expected = pd.MultiIndex.from_product(
        [TARGETS, range(repeats), shards.shard_index, METHODS],
        names=["target", "repeat", "shard_index", "method"])
    key = ["target", "repeat", "shard_index", "method"]
    if (frame.duplicated(key).any() or len(frame) != len(expected) or
            not frame.set_index(key).index.sort_values().equals(expected.sort_values())):
        raise ValueError("CV summary lacks a complete target-repeat-shard-method grid")
    expected_loci = shards.set_index("shard_index").n_loci
    if (not frame.shard_index.map(expected_loci).eq(frame.n_loci).all() or
            not frame.n_sites_two_or_more.between(0, frame.n_loci).all() or
            not frame.n_site_subject_observations.between(0, frame.n_loci * 205).all()):
        raise ValueError("CV shard counts disagree with source CpG shards")
    if (frame.n_sites_two_or_more.gt(0) & ~np.isfinite(frame.mean_site_rmse)).any():
        raise ValueError("Eligible shard has undefined mean RMSE")
    mask_counts = frame.pivot(index=["target", "repeat", "shard_index"],
                              columns="method", values=["n_sites_two_or_more", "n_site_subject_observations"])
    for metric in ("n_sites_two_or_more", "n_site_subject_observations"):
        values = mask_counts[metric].to_numpy()
        if not (values == values[:, :1]).all():
            raise ValueError("Methods were scored on different CV observation masks")
    rows = []
    for (target, repeat, method), part in frame.groupby(["target", "repeat", "method"], sort=True):
        n_sites = int(part.n_sites_two_or_more.sum())
        if n_sites == 0:
            raise ValueError("No eligible sites in a complete CV repeat")
        rows.append({"target": target, "repeat": repeat, "method": method,
                     "n_loci": int(part.n_loci.sum()), "n_sites_two_or_more": n_sites,
                     "n_site_subject_observations": int(part.n_site_subject_observations.sum()),
                     "mean_site_rmse": float(np.nansum(
                         part.mean_site_rmse.to_numpy() * part.n_sites_two_or_more.to_numpy()) / n_sites)})
    per_repeat = pd.DataFrame(rows)
    wide = per_repeat.pivot(index=["target", "repeat"], columns="method", values="mean_site_rmse")
    differences = []
    for (target, repeat), row in wide.iterrows():
        for method in METHODS:
            if method != "A2H":
                differences.append({"target": target, "repeat": repeat, "control": method,
                                    "mean_site_rmse_A2H_minus_control": row.A2H - row[method],
                                    "n_sites_two_or_more": int(per_repeat.loc[
                                        (per_repeat.target == target) & (per_repeat.repeat == repeat),
                                        "n_sites_two_or_more"].iloc[0])})
    return per_repeat, pd.DataFrame(differences)


def run(summary_dir: Path, shard_dir: Path, output_dir: Path) -> None:
    completion_path = summary_dir / "completion.json"
    table_path = summary_dir / "repeat_shard_summary.tsv"
    shards_path = shard_dir / "shards.tsv"
    completion = json.loads(completion_path.read_text())
    if (not completion["aggregation_complete"] or completion["split_count"] % 10 or
            completion["n_shards"] < 1 or completion["input_sha256"]["shards"] != file_hash(shards_path)):
        raise ValueError("Atlas repeat aggregation is incomplete or uses different CpG shards")
    shards = pd.read_csv(shards_path, sep="\t")
    if (shards.shard_index.duplicated().any() or
            len(shards) != completion["n_shards"] or
            int(shards.n_loci.sum()) != completion["n_loci"]):
        raise ValueError("Source CpG shard manifest changed")
    frame = pd.read_csv(table_path, sep="\t")
    per_repeat, differences = aggregate(frame, shards, completion)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_repeat.to_csv(output_dir / "cv_mean_site_rmse_by_repeat.tsv", sep="\t", index=False)
    differences.to_csv(output_dir / "cv_rmse_difference_by_repeat.tsv", sep="\t", index=False)
    (output_dir / "provenance.json").write_text(json.dumps({
        "branch": completion["branch"], "split_count": completion["split_count"],
        "n_loci": completion["n_loci"],
        "source_sha256": {"completion": file_hash(completion_path),
                          "repeat_shard_summary": file_hash(table_path),
                          "shards": file_hash(shards_path)},
        "aggregation": "eligible-site-weighted mean of shard mean per-site RMSE; common mask across methods",
        "limitation": "Repeated folds reuse the same subjects; this file contains no independent confidence interval or global median rank statistic",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.summary_dir, args.shard_dir, args.output_dir)


if __name__ == "__main__":
    main()
