"""Verify all E4 shards and summarize site-specific sample-resolved OOF accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.r2_2_common import pooled_metrics
from src.r2_2_reference_size import DEFAULT_ROOT, TARGETS


def checked_shards(root: Path, count: int = 52) -> dict[int, dict]:
    complete = {}
    for shard in range(count):
        path = root / "E4" / f"shard{shard:04d}" / "completion.json"
        if not path.exists():
            raise FileNotFoundError(path)
        entry = json.loads(path.read_text())
        if (entry.get("run_status") != "complete" or entry["shard_index"] != shard or
                entry["n_repeats"] != 10 or entry["n_splits"] != 100 or
                entry["n_reference_samples"] != 205):
            raise ValueError(f"Incomplete E4 shard: {path}")
        for target in TARGETS:
            for record in entry["outputs"][target].values():
                artifact = Path(record["path"])
                if not artifact.exists() or file_hash(artifact) != record["sha256"]:
                    raise ValueError(f"E4 output changed or missing: {artifact}")
        complete[shard] = entry
    return complete


def summarize(root: Path) -> None:
    e4 = root / "E4"
    dest = e4 / "summary"
    if dest.exists():
        raise FileExistsError(f"Refusing to overwrite {dest}")
    entries = checked_shards(root)
    mapping = pd.read_csv(root / "E1/epic_mapping.tsv", sep="\t", usecols=["locus"])
    expected = set(mapping.locus)
    if mapping.locus.duplicated().any():
        raise ValueError("E4 mapping has duplicate loci")
    site_tables = []
    repeat_tables = []
    for shard, entry in entries.items():
        for target in TARGETS:
            path = entry["outputs"][target]["per_site_oof"]["path"]
            sites = pd.read_parquet(path)
            if len(sites) != entry["n_epic_loci"] or sites.locus.duplicated().any():
                raise ValueError("E4 per-site shard row count/key mismatch")
            site_tables.append(sites)
            repeats = pd.read_parquet(entry["outputs"][target]["overall_by_repeat_shard"]["path"])
            if len(repeats) != 10 or set(repeats.repeat) != set(range(10)):
                raise ValueError("E4 per-repeat shard coverage incomplete")
            repeat_tables.append(repeats)
    all_sites = pd.concat(site_tables, ignore_index=True)
    if len(all_sites) != len(expected) * 2:
        raise ValueError("E4 did not cover every mapped EPIC locus for both targets")
    for target in TARGETS:
        part = all_sites.loc[all_sites.target == target]
        if len(part) != len(expected) or set(part.locus) != expected or part.locus.duplicated().any():
            raise ValueError(f"E4 locus coverage differs for {target}")
        if (part.n > part.n_truth_measurements).any() or (part.n > 205).any():
            raise ValueError("E4 predictions exceed biological sample support")
        if (part.min_fold_train_joint > part.max_fold_train_joint).any():
            raise ValueError("Invalid training support range")
    dest.mkdir(parents=True)
    all_sites.to_parquet(dest / "E4_per_site_oof_metrics.parquet", index=False, compression="zstd")
    distribution = []
    overall = []
    for target in TARGETS:
        sites = all_sites.loc[all_sites.target == target]
        stats = pooled_metrics(sites)
        overall.append({"target": target, "evaluation": "per-sample averaged OOF over ten repeats",
                        "repeat": -1, "n_epic_sites": len(sites), **stats,
                        "n_constant_observed": int(sites.correlation_reason.eq("constant_observed").sum()),
                        "n_constant_prediction": int(sites.correlation_reason.eq("constant_prediction").sum()),
                        "n_insufficient_pairs_for_r": int(sites.correlation_reason.eq("insufficient_pairs").sum())})
        for column in ("rmse", "mae", "bias_pred_minus_obs", "pearson_r", "observed_variance",
                       "valid_prediction_fraction"):
            values = sites[column].dropna().to_numpy(dtype=float)
            distribution.append({"target": target, "metric": column, "n_defined_sites": len(values),
                                 "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                                 "q75": float(np.quantile(values, 0.75)), "q95": float(np.quantile(values, 0.95)),
                                 "q99": float(np.quantile(values, 0.99))})
    repeats = pd.concat(repeat_tables, ignore_index=True)
    for (target, repeat), part in repeats.groupby(["target", "repeat"]):
        sxy = float(part.centered_sxy.sum())
        syy = float(part.centered_syy.sum())
        spp = float(part.centered_spp.sum())
        n = int(part.n_valid_pairs.sum())
        overall.append({"target": target, "evaluation": "one OOF prediction per biological sample in this repeat",
                        "repeat": repeat, "n_epic_sites": int(part.n_epic_sites.sum()),
                        "n_valid_pairs": n, "n_sites_with_pairs": int(part.n_sites_with_pairs.sum()),
                        "pooled_rmse": float(np.sqrt(part.sse.sum() / n)),
                        "n_centered_sites": int(part.n_centered_sites.sum()),
                        "n_centered_pairs": int(part.n_centered_pairs.sum()),
                        "centered_pearson_r": sxy / np.sqrt(syy * spp) if syy > 0 and spp > 0 else np.nan,
                        "median_site_rmse": np.nan, "median_defined_site_pearson_r": np.nan,
                        "n_defined_site_pearson_r": np.nan,
                        "n_constant_observed": np.nan, "n_constant_prediction": np.nan,
                        "n_insufficient_pairs_for_r": np.nan})
    pd.DataFrame(overall).to_csv(dest / "E4_overall_oof_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(distribution).to_csv(dest / "E4_site_distribution.tsv", sep="\t", index=False)
    bins = [-1, 49, 74, 99, 149, 199, 205]
    labels = ["<50", "50-74", "75-99", "100-149", "150-199", "200-205"]
    all_sites["reference_joint_bin"] = pd.cut(all_sites.n_reference_joint, bins=bins, labels=labels)
    support = []
    for (target, bin_label), part in all_sites.groupby(["target", "reference_joint_bin"], observed=True):
        support.append({"target": target, "reference_joint_bin": str(bin_label),
                        "n_sites": len(part), "n_valid_pairs": int(part.n.sum()),
                        "median_site_rmse": float(part.rmse.median()),
                        "p95_site_rmse": float(part.rmse.quantile(0.95)),
                        "n_site_r_defined": int(part.pearson_r.notna().sum()),
                        "median_defined_site_r": float(part.pearson_r.median())})
    pd.DataFrame(support).to_csv(dest / "E4_by_reference_support.tsv", sep="\t", index=False)
    plot(all_sites, dest)
    (dest / "completion.json").write_text(json.dumps({
        "n_shards": len(entries), "n_epic_unique_loci": len(expected), "n_reference_samples": 205,
        "n_repeats": 10, "unit_of_primary_site_metrics": "unique atlas sample after averaging available OOF predictions across repeats",
        "source_cv_fit_caveat": "reviewer sample-resolved vectorized fit, not literal historical scalar polyfit",
        "shard_completions_sha256": {str(shard): file_hash(e4 / f"shard{shard:04d}/completion.json") for shard in entries},
        "outputs": {path.name: file_hash(path) for path in dest.iterdir() if path.is_file()},
    }, indent=2) + "\n")


def plot(sites: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {"CHALM": "#087f5b", "CAMDA": "#b65d26"}
    for row, target in enumerate(TARGETS):
        part = sites.loc[sites.target == target].sort_values("locus")
        rmse = part.rmse.to_numpy(dtype=float)
        r = part.pearson_r.to_numpy(dtype=float)
        axes[row, 0].hist(rmse[np.isfinite(rmse) & (rmse <= np.nanquantile(rmse, 0.99))],
                          bins=65, color=colors[target], alpha=0.75)
        axes[row, 0].set_title(f"{target}: site OOF RMSE (to 99th pct)")
        axes[row, 0].set_xlabel("RMSE, native scale")
        axes[row, 0].set_ylabel("EPIC loci")
        defined = r[np.isfinite(r)]
        axes[row, 1].hist(defined, bins=60, color=colors[target], alpha=0.75)
        axes[row, 1].set_title(f"{target}: within-site OOF Pearson r")
        axes[row, 1].set_xlabel("Pearson r (n>=3, nonconstant)")
        axes[row, 1].set_ylabel("EPIC loci")
    fig.savefig(dest / "E4_site_reliability.png", dpi=240)
    fig.savefig(dest / "E4_site_reliability.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    summarize(args.root)


if __name__ == "__main__":
    main()
