"""Summarize held-out atlas CV error against fold-local fit reliability."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.run_atlas_cv_split import TARGETS


TRAIN_BINS = [-1, 49, 99, 149, 199, np.inf]
TRAIN_LABELS = ["<50", "50-99", "100-149", "150-199", ">=200"]


def strata(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(frame)
    train = pd.cut(frame.min_train_joint, TRAIN_BINS, labels=TRAIN_LABELS).astype(str).to_numpy()
    fit = np.where(frame.n_quadratic_folds.to_numpy() == frame.n_training_folds.to_numpy(),
                   "quadratic_all_folds", "nonquadratic_any_fold")
    unique = np.where(frame.min_unique_input.to_numpy() <= 1, "<=1",
                      np.where(frame.min_unique_input.to_numpy() == 2, "2", ">=3"))
    if any(len(values) != n for values in (train, fit, unique)):
        raise ValueError("Reliability stratum length differs")
    return {"min_train_joint": train, "fit_status": fit, "min_unique_input": unique}


def summarize(values: dict[str, list[np.ndarray]], n_loci: int) -> dict:
    rmse = np.concatenate(values["rmse"])
    defined = np.concatenate(values["defined"])
    observed = np.concatenate(values["observed"])
    valid = observed & np.isfinite(rmse)
    if len(rmse) != n_loci or np.any(observed & ~np.isfinite(rmse)):
        raise ValueError("Observed CV errors are missing or stratum count differs")
    return {
        "n_loci": n_loci,
        "n_with_heldout_truth": int(valid.sum()),
        "n_with_defined_rank": int(defined.sum()),
        "fraction_undefined_rank_among_observed": float(1 - defined[valid].mean()) if valid.any() else np.nan,
        "median_site_rmse": float(np.median(rmse[valid])) if valid.any() else np.nan,
        "p90_site_rmse": float(np.quantile(rmse[valid], 0.9)) if valid.any() else np.nan,
    }


def run(site_dir: Path, verified_dir: Path, output_dir: Path) -> None:
    manifest_path = site_dir / "site_quality_shards.tsv"
    completion_path = site_dir / "completion.json"
    verified_path = verified_dir / "provenance.json"
    completion = json.loads(completion_path.read_text())
    verified = json.loads(verified_path.read_text())
    if (verified["branch"] != completion["branch"] or
            verified["n_loci"] != completion["n_loci"] or
            verified["split_count"] != completion["split_count"] or
            verified["source_sha256"]["site_completion"] != file_hash(completion_path) or
            verified["source_sha256"]["site_shard_manifest"] != file_hash(manifest_path)):
        raise ValueError("Site-quality inputs differ from the verified branch")
    manifest = pd.read_csv(manifest_path, sep="\t")
    expected = pd.MultiIndex.from_product([TARGETS, range(completion["n_output_shards"] // len(TARGETS))],
                                          names=["target", "shard_index"])
    if (manifest.duplicated(["target", "shard_index"]).any() or
            not manifest.set_index(["target", "shard_index"]).index.sort_values().equals(expected.sort_values())):
        raise ValueError("Incomplete site-quality shard manifest")
    collected = defaultdict(lambda: {"rmse": [], "observed": [], "defined": [], "n_loci": 0})
    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        if file_hash(path) != row.sha256:
            raise ValueError(f"Changed site-quality shard: {path}")
        frame = pd.read_parquet(path)
        if (len(frame) != row.n_loci or not (frame.target == row.target).all() or
                not (frame.branch == completion["branch"]).all()):
            raise ValueError(f"Invalid site-quality shard keys: {path}")
        rmse = frame.mean_repeat_rmse.to_numpy(dtype=float)
        observed = frame.n_repeats_with_observed_pairs.to_numpy() > 0
        defined = frame.n_repeats_with_defined_spearman.to_numpy() > 0
        if np.any(defined & ~observed):
            raise ValueError(f"Defined rank without held-out truth: {path}")
        for dimension, labels in strata(frame).items():
            for label in np.unique(labels):
                selected = labels == label
                bucket = collected[(row.target, dimension, label)]
                bucket["rmse"].append(rmse[selected])
                bucket["observed"].append(observed[selected])
                bucket["defined"].append(defined[selected])
                bucket["n_loci"] += int(selected.sum())
    rows = []
    for (target, dimension, label), values in sorted(collected.items()):
        rows.append({"target": target, "dimension": dimension, "stratum": label,
                     **summarize(values, values["n_loci"])})
    table = pd.DataFrame(rows)
    for target in TARGETS:
        for dimension in ("min_train_joint", "fit_status", "min_unique_input"):
            if table.loc[(table.target == target) & (table.dimension == dimension), "n_loci"].sum() != completion["n_loci"]:
                raise ValueError("Reliability strata do not partition the site population")
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "reliability_by_stratum.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for target in TARGETS:
        subset = table.loc[(table.target == target) & (table.dimension == "min_train_joint")].set_index("stratum").reindex(TRAIN_LABELS)
        subset = subset.loc[subset.n_loci.notna() & subset.n_loci.gt(0)]
        axes[0].plot(subset.index, subset.median_site_rmse, marker="o", label=target)
        axes[1].plot(subset.index, 100 * subset.fraction_undefined_rank_among_observed,
                     marker="o", label=target)
    axes[0].set(ylabel="Median held-out site RMSE", xlabel="Minimum joint training pairs")
    axes[1].set(ylabel="Sites with undefined rank (%)", xlabel="Minimum joint training pairs")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    figure_path = output_dir / "error_vs_training_pairs.pdf"
    png_path = output_dir / "error_vs_training_pairs.png"
    fig.savefig(figure_path)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    (output_dir / "provenance.json").write_text(json.dumps({
        "branch": completion["branch"], "n_loci_per_target": completion["n_loci"],
        "source_sha256": {"site_manifest": file_hash(manifest_path),
                          "site_verification": file_hash(verified_path)},
        "output_sha256": {"strata": file_hash(table_path), "figure": file_hash(figure_path),
                          "png": file_hash(png_path)},
        "interpretation": "Descriptive held-out error and rank eligibility; not calibrated prediction intervals or independent-cohort uncertainty",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--verified-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.site_dir, args.verified_dir, args.output_dir)


if __name__ == "__main__":
    main()
