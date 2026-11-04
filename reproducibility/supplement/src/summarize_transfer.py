"""Verify complete transfer shards and summarize paired external and atlas holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash, row_stats, spearman_rows


TARGETS = ("CHALM", "CAMDA")
EXTERNAL_METHODS = ("A2H", "historical_A2H", "reference_linear", "m")


def load_completions(root: Path, model: str, expected_shards: int) -> list[dict]:
    completions = []
    for shard_index in range(expected_shards):
        path = root / model / f"shard_{shard_index:04d}" / "completion.json"
        if not path.exists():
            raise ValueError(f"Missing transfer completion: {path}")
        record = json.loads(path.read_text())
        if record["model"] != model or record["shard_index"] != shard_index or set(record["targets"]) != set(TARGETS):
            raise ValueError(f"Transfer completion identity mismatch: {path}")
        for target, detail in record["targets"].items():
            for key in ("heldout_site_metrics", "external_site_metrics", "external_predictions"):
                hash_key = "heldout_sha256" if key == "heldout_site_metrics" else f"{key}_sha256"
                if key in detail and file_hash(Path(detail[key])) != detail[hash_key]:
                    raise ValueError(f"Transfer output hash mismatch: {model} {shard_index} {target} {key}")
        completions.append(record)
    input_hashes = [record["source_sha256"] for record in completions]
    for key in ("atlas_shards", "external_shards", "membership", "samples"):
        if len({entry[key] for entry in input_hashes}) != 1:
            raise ValueError(f"Inconsistent {key} across transfer shards")
    return completions


def paired_external_metrics(frame: pd.DataFrame, min_subjects: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"probe_id", "sample_id", "observed", "transfer_A2H", "historical_A2H",
                "transfer_reference_linear", "m"}
    if not required.issubset(frame) or frame.duplicated(["probe_id", "sample_id"]).any():
        raise ValueError("Incomplete or duplicate paired external observations")
    if not np.isfinite(frame[list(required - {"probe_id", "sample_id"})].to_numpy(dtype=float)).all():
        raise ValueError("Paired external predictions contain nonfinite values")
    counts = frame.groupby("probe_id", sort=False).size()
    frame = frame.loc[frame.probe_id.isin(counts.loc[counts >= min_subjects].index)].copy()
    frame = frame.rename(columns={"transfer_A2H": "A2H", "transfer_reference_linear": "reference_linear"})
    subjects = sorted(frame.sample_id.unique())
    probes = pd.Index(frame.probe_id.drop_duplicates())
    index = pd.MultiIndex.from_product([probes, subjects], names=["probe_id", "sample_id"])
    aligned = frame.set_index(["probe_id", "sample_id"]).reindex(index)
    y = aligned.observed.to_numpy(dtype=float).reshape(len(probes), len(subjects))
    rows = []
    for method in EXTERNAL_METHODS:
        pred = aligned[method].to_numpy(dtype=float).reshape(y.shape)
        stats = row_stats(y, pred)
        if (stats["n"] < min_subjects).any():
            raise ValueError(f"Incomplete site in paired {method} external table")
        rows.append(pd.DataFrame({"probe_id": probes, "method": method, "n": stats["n"],
                                  "rmse": stats["rmse"], "pearson_r": stats["pearson_r"],
                                  "spearman_rho": spearman_rows(y, pred)}))
    return frame, pd.concat(rows, ignore_index=True)


def method_differences(site: pd.DataFrame, target: str) -> pd.DataFrame:
    pivot = site.pivot(index="probe_id", columns="method", values="rmse")
    if set(pivot) != set(EXTERNAL_METHODS) or pivot.isna().any().any():
        raise ValueError("Incomplete paired method-site table")
    rows = []
    for control in EXTERNAL_METHODS:
        if control == "A2H":
            continue
        difference = pivot.A2H - pivot[control]
        rows.append({"target": target, "control": control, "n_sites": len(pivot),
                     "mean_site_rmse_A2H_minus_control": float(difference.mean()),
                     "median_site_rmse_A2H_minus_control": float(difference.median()),
                     "fraction_sites_lower_rmse_A2H": float((difference < 0).mean())})
    return pd.DataFrame(rows)


def heldout_summary(site: pd.DataFrame, target: str, model: str, min_subjects: int) -> pd.DataFrame:
    if site.duplicated(["locus", "method"]).any():
        raise ValueError("Duplicate held-out atlas site-method key")
    pivot_n = site.pivot(index="locus", columns="method", values="n")
    eligible = pivot_n.A2H >= min_subjects
    common = pivot_n.loc[eligible, ["A2H", "reference_linear", "m"]]
    if not common.eq(common.A2H, axis=0).all().all():
        raise ValueError("Held-out transfer methods use unequal subject counts")
    paired = site.loc[site.locus.isin(common.index) & site.method.isin(common.columns)]
    pivot_rmse = paired.pivot(index="locus", columns="method", values="rmse")
    rows = []
    for method in ("A2H", "reference_linear", "m"):
        values = pivot_rmse[method]
        rows.append({"model": model, "target": target, "method": method,
                     "n_sites": len(values), "mean_site_rmse": float(values.mean()),
                     "median_site_rmse": float(values.median()),
                     "mean_site_rmse_A2H_minus_method": float((pivot_rmse.A2H - values).mean())})
    return pd.DataFrame(rows)


def summarize(root: Path, model: str, output: Path, expected_shards: int, min_subjects: int,
              pairs_path: Path) -> None:
    if expected_shards < 1 or min_subjects < 2:
        raise ValueError("Invalid shard count or minimum subject count")
    completions = load_completions(root, model, expected_shards)
    output.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(pairs_path, sep="\t")
    pairs.to_csv(output / "pairs.tsv", sep="\t", index=False)
    retention_rows, difference_tables, heldout_tables = [], [], []
    for target in TARGETS:
        details = [record["targets"][target] for record in completions]
        external_frames = [pd.read_parquet(detail["external_predictions"])
                           for detail in details if "external_predictions" in detail]
        external = pd.concat(external_frames, ignore_index=True)
        if not set(external.sample_id).issubset(set(pairs.array_id)):
            raise ValueError("Unexpected external subject in transfer output")
        paired, site = paired_external_metrics(external, min_subjects)
        paired.to_parquet(output / f"{target}_observations.parquet", index=False, compression="zstd")
        site.to_parquet(output / f"{target}_site_metrics.parquet", index=False, compression="zstd")
        difference_tables.append(method_differences(site, target))
        atlas_loci = sum(record["n_loci"] for record in completions)
        input_probes = sum(detail["n_external_input"] for detail in details)
        fitted_probes = sum(detail["n_external_fitted"] for detail in details)
        if paired.probe_id.nunique() != fitted_probes:
            raise ValueError("External fitted-probe count disagrees with paired observations")
        retention_rows.append({"model": model, "target": target, "atlas_loci": atlas_loci,
                               "atlas_joint_eligible": sum(detail["n_joint_eligible"] for detail in details),
                               "external_base_eligible_probes": input_probes,
                               "external_transfer_fitted_probes": fitted_probes,
                               "external_transfer_lost_probes": input_probes - fitted_probes,
                               "external_paired_observations": len(paired)})
        heldout_frames = [pd.read_parquet(detail["heldout_site_metrics"])
                          for detail in details if "heldout_site_metrics" in detail]
        if heldout_frames:
            heldout = pd.concat(heldout_frames, ignore_index=True)
            heldout.to_parquet(output / f"{target}_heldout_site_metrics.parquet", index=False, compression="zstd")
            heldout_tables.append(heldout_summary(heldout, target, model, min_subjects))
    pd.DataFrame(retention_rows).to_csv(output / "retention.tsv", sep="\t", index=False)
    pd.concat(difference_tables, ignore_index=True).to_csv(output / "method_differences.tsv", sep="\t", index=False)
    if heldout_tables:
        pd.concat(heldout_tables, ignore_index=True).to_csv(output / "heldout_summary.tsv", sep="\t", index=False)
    provenance = {"model": model, "expected_shards": expected_shards,
                  "comparison_population": "depth10 full-reference calibrated PGP-UK observations; transfer-fitted sites only",
                  "external_min_subjects": min_subjects,
                  "heldout_min_subjects": min_subjects,
                  "paired_subject_rule": "same probe-subject observations for A2H and controls",
                  "input_completion_sha256": {
                      str(i): file_hash(root / model / f"shard_{i:04d}" / "completion.json")
                      for i in range(expected_shards)},
                  "pairs_sha256": file_hash(pairs_path)}
    (output / "completion.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=52)
    parser.add_argument("--min-subjects", type=int, default=8)
    parser.add_argument("--pairs", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.model, args.output_dir, args.expected_shards, args.min_subjects, args.pairs)


if __name__ == "__main__":
    main()
