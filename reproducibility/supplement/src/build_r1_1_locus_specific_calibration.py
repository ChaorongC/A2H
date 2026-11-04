"""Build the R1-1 A2H coefficient, curve, and matched atlas-CV evidence.

This script consumes the deployed coefficient artifact, the authoritative
locus-to-probe map, the prepared atlas shards, and the completed historical
sample CV summaries. It writes only to a new R1-1 result branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("CHALM", "CAMDA")
METHODS = ("A2H", "universal_quadratic")
COEFFICIENT_NAMES = ("intercept", "linear", "quadratic")
COEFFICIENT_STORAGE_INDEX = {"quadratic": 0, "linear": 1, "intercept": 2}
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
SITE_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def load_coefficients(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    required = {"featureID", "chalm_coef", "camda_coef"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Coefficient artifact is missing columns: {required - set(frame.columns)}")
    if frame.featureID.isna().any() or frame.featureID.duplicated().any():
        raise ValueError("Coefficient featureID keys are missing or duplicated")
    result = pd.DataFrame({"locus": frame.featureID.astype(str)})
    for target, column in (("CHALM", "chalm_coef"), ("CAMDA", "camda_coef")):
        values = []
        for value in frame[column]:
            array = np.asarray(value, dtype=float)
            if array.shape != (3,):
                raise ValueError(f"{column} contains a non-quadratic coefficient triplet")
            values.append(array)
        array = np.vstack(values)
        for name in COEFFICIENT_NAMES:
            result[f"{target}_{name}"] = array[:, COEFFICIENT_STORAGE_INDEX[name]]
    if result.locus.nunique() != len(result):
        raise ValueError("Coefficient loci are not unique")
    return result


def load_probe_map(path: Path, coefficient_loci: set[str]) -> tuple[pd.DataFrame, set[str]]:
    probe = pd.read_csv(path, sep="\t", dtype="string")
    required = {"loci.hg38", "EPIC.hg38"}
    if not required.issubset(probe.columns):
        raise ValueError(f"Probe map is missing columns: {required - set(probe.columns)}")
    probe = probe.loc[probe["loci.hg38"].notna()].copy()
    if probe["loci.hg38"].duplicated().any():
        duplicate_loci = set(probe.loc[probe["loci.hg38"].duplicated(keep=False), "loci.hg38"])
        if not duplicate_loci.issubset(coefficient_loci):
            raise ValueError("Probe-map duplicate locus keys include loci absent from coefficients")
    loci = probe[["loci.hg38"]].drop_duplicates().rename(columns={"loci.hg38": "locus"})
    epic_rows = probe.loc[
        probe["EPIC.hg38"].notna() & probe["EPIC.hg38"].astype(str).ne(""),
        ["loci.hg38", "EPIC.hg38"],
    ].drop_duplicates()
    epic_rows = epic_rows.rename(columns={"loci.hg38": "locus", "EPIC.hg38": "epic_probe_ids"})
    first = epic_rows.drop_duplicates("locus").set_index("locus")["epic_probe_ids"]
    counts = epic_rows.groupby("locus", sort=False).size().rename("n_epic_probe_ids")
    result = loci.merge(first.rename("epic_probe_ids"), left_on="locus", right_index=True, how="left")
    result = result.merge(counts, left_on="locus", right_index=True, how="left")
    result["n_epic_probe_ids"] = result.n_epic_probe_ids.fillna(0).astype(int)
    result["epic_probe_ids"] = result.epic_probe_ids.fillna("").astype(str)
    multiple = set(counts.index[counts.gt(1)])
    for locus in multiple:
        ids = sorted(set(epic_rows.loc[epic_rows.locus == locus, "epic_probe_ids"].astype(str)))
        result.loc[result.locus == locus, "epic_probe_ids"] = ";".join(ids)
    result["is_epic"] = result.n_epic_probe_ids.gt(0)
    if result.locus.duplicated().any():
        raise ValueError("Collapsed probe map still contains duplicate loci")
    epic_loci = set(result.loc[result.is_epic, "locus"]) & coefficient_loci
    return result, epic_loci


def load_shards(shards_path: Path) -> pd.DataFrame:
    shards = pd.read_csv(shards_path, sep="\t")
    required = {"shard_index", "path", "n_loci", "first_locus", "last_locus"}
    if not required.issubset(shards.columns):
        raise ValueError(f"Shard manifest is missing columns: {required - set(shards.columns)}")
    shards.path = shards.path.map(lambda value: Path(value))
    if not all(path.exists() for path in shards.path):
        missing = [str(path) for path in shards.loc[~shards.path.map(Path.exists), "path"]]
        raise FileNotFoundError(missing[:5])
    if shards.shard_index.duplicated().any():
        raise ValueError("Shard manifest contains duplicate shard indices")
    return shards.sort_values("shard_index").reset_index(drop=True)


def load_samples(sample_path: Path) -> list[str]:
    samples = pd.read_csv(sample_path, sep="\t")
    if samples.sample_id.duplicated().any():
        raise ValueError("Atlas sample manifest contains duplicate sample IDs")
    return samples.sample_id.astype(str).tolist()


def coefficient_summary(coefficients: pd.DataFrame, output: Path) -> None:
    rows = []
    for target in TARGETS:
        for universe, mask in (("genomewide_deployed", np.ones(len(coefficients), dtype=bool)),
                               ("EPIC_BeadChip", coefficients.is_epic.to_numpy(dtype=bool))):
            for name in COEFFICIENT_NAMES:
                values = coefficients.loc[mask, f"{target}_{name}"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                quantiles = np.quantile(values, QUANTILES)
                row = {
                    "target": target, "universe": universe, "coefficient": name,
                    "n": len(values), "n_exact_zero": int(np.isclose(values, 0.0, atol=0.0).sum()),
                    "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                    "min": float(values.min()), "max": float(values.max()),
                }
                row.update({f"q{int(q * 100):02d}": float(value) for q, value in zip(QUANTILES, quantiles)})
                rows.append(row)
    pd.DataFrame(rows).to_csv(output / "coefficient_distribution_summary.tsv", sep="\t", index=False)


def write_coefficient_table(coefficients: pd.DataFrame, output: Path) -> None:
    columns = ["locus", "is_epic", "epic_probe_ids", "n_epic_probe_ids"]
    columns += [f"{target}_{name}" for target in TARGETS for name in COEFFICIENT_NAMES]
    coefficients[columns].to_csv(output / "coefficient_table.tsv.gz", sep="\t", index=False, compression="gzip")


def plot_coefficient_distributions(coefficients: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    colors = {"genomewide_deployed": "#374151", "EPIC_BeadChip": "#0f766e"}
    for row_index, target in enumerate(TARGETS):
        for column_index, name in enumerate(COEFFICIENT_NAMES):
            axis = axes[row_index, column_index]
            all_values = coefficients[f"{target}_{name}"].to_numpy(dtype=float)
            finite = all_values[np.isfinite(all_values)]
            lower, upper = np.quantile(finite, [0.01, 0.99])
            bins = np.linspace(lower, upper, 61)
            for universe, mask in (("genomewide_deployed", np.ones(len(coefficients), dtype=bool)),
                                   ("EPIC_BeadChip", coefficients.is_epic.to_numpy(dtype=bool))):
                values = coefficients.loc[mask, f"{target}_{name}"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                axis.hist(values, bins=bins, density=True, histtype="step", linewidth=1.5,
                          color=colors[universe], label=universe.replace("_", " "))
            axis.set_title(f"{target} {name}")
            axis.set_xlabel("coefficient value; plotted range is 1st-99th percentile")
            axis.set_ylabel("density")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=8)
    fig.savefig(output / "coefficient_distributions.png", dpi=300)
    fig.savefig(output / "coefficient_distributions.pdf")
    plt.close(fig)


def compute_truth_ranges(shards: pd.DataFrame, samples: list[str], output: Path) -> pd.DataFrame:
    ranges = []
    for target in TARGETS:
        for shard in shards.itertuples(index=False):
            columns = ["locus"] + [f"{target}__{sample}" for sample in samples]
            frame = pd.read_parquet(shard.path, columns=columns)
            if (len(frame) != int(shard.n_loci) or
                    str(frame.locus.iloc[0]) != str(shard.first_locus) or
                    str(frame.locus.iloc[-1]) != str(shard.last_locus)):
                raise ValueError(f"Atlas shard row structure is invalid: {shard.path}")
            values = frame.iloc[:, 1:].to_numpy(dtype=float)
            finite = np.isfinite(values)
            count = finite.sum(axis=1)
            safe = np.where(finite, values, np.nan)
            minimum = np.nanmin(safe, axis=1)
            maximum = np.nanmax(safe, axis=1)
            minimum[count == 0] = np.nan
            maximum[count == 0] = np.nan
            ranges.append(pd.DataFrame({
                "target": target, "locus": frame.locus.astype(str).to_numpy(),
                "n_truth_finite": count.astype(int), "truth_min": minimum,
                "truth_max": maximum, "truth_range": maximum - minimum,
            }))
    result = pd.concat(ranges, ignore_index=True)
    result.to_csv(output / "atlas_truth_ranges.tsv.gz", sep="\t", index=False, compression="gzip")
    return result


def candidate_rows(shards: pd.DataFrame, samples: list[str], target: str,
                   candidates: set[str]) -> dict[str, dict[str, np.ndarray]]:
    found = {}
    for shard in shards.itertuples(index=False):
        if not any(str(shard.first_locus) <= locus <= str(shard.last_locus) for locus in candidates):
            continue
        columns = ["locus"] + [f"m__{sample}" for sample in samples]
        columns += [f"{target}__{sample}" for sample in samples]
        frame = pd.read_parquet(shard.path, columns=columns,
                                filters=[("locus", "in", sorted(candidates))])
        for row in frame.itertuples(index=False, name=None):
            locus = str(row[0])
            if locus not in candidates:
                continue
            midpoint = len(samples) + 1
            found[locus] = {
                "m": np.asarray(row[1:midpoint], dtype=float),
                "y": np.asarray(row[midpoint:], dtype=float),
            }
    return found


def representative_curves(coefficients: pd.DataFrame, shards: pd.DataFrame,
                           samples: list[str], epic_loci: set[str], output: Path,
                           minimum_pairs: int) -> None:
    selected_rows = []
    value_rows = []
    curve_rows = []
    grid = np.linspace(0.0, 1.0, 201)
    atlas_loci = set()
    for shard in shards.itertuples(index=False):
        atlas_loci.update(pd.read_parquet(shard.path, columns=["locus"]).locus.astype(str))
    for target in TARGETS:
        source = coefficients.loc[coefficients.is_epic & coefficients.locus.isin(atlas_loci)].copy()
        source = source.loc[np.isfinite(source[f"{target}_quadratic"])].copy()
        candidates = set()
        quantile_targets = [("q05", 0.05), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q95", 0.95)]
        for label, quantile in quantile_targets:
            value = float(source[f"{target}_quadratic"].quantile(quantile))
            pool = source.assign(_distance=(source[f"{target}_quadratic"] - value).abs())
            candidates.update(pool.sort_values(["_distance", "locus"]).head(250).locus)
        found = candidate_rows(shards, samples, target, candidates)
        for label, quantile in quantile_targets:
            value = float(source[f"{target}_quadratic"].quantile(quantile))
            pool = source.assign(_distance=(source[f"{target}_quadratic"] - value).abs())
            choice = None
            for row in pool.sort_values(["_distance", "locus"]).itertuples(index=False):
                values = found.get(row.locus)
                if values is None:
                    continue
                pair = np.isfinite(values["m"]) & np.isfinite(values["y"])
                if int(pair.sum()) >= minimum_pairs:
                    choice = (row, values, pair)
                    break
            if choice is None:
                raise ValueError(f"Could not select a representative for {target} {label}")
            row, values, pair = choice
            fitted = (row.__getattribute__(f"{target}_intercept")
                      + row.__getattribute__(f"{target}_linear") * values["m"]
                      + row.__getattribute__(f"{target}_quadratic") * values["m"] ** 2)
            selected_rows.append({
                "target": target, "category": label, "locus": row.locus,
                "epic_probe_ids": row.epic_probe_ids,
                "n_pairs": int(pair.sum()), "m_min": float(np.min(values["m"][pair])),
                "m_max": float(np.max(values["m"][pair])),
                "observed_min": float(np.min(values["y"][pair])),
                "observed_max": float(np.max(values["y"][pair])),
                "intercept": float(row.__getattribute__(f"{target}_intercept")),
                "linear": float(row.__getattribute__(f"{target}_linear")),
                "quadratic": float(row.__getattribute__(f"{target}_quadratic")),
                "quadratic_quantile": quantile,
            })
            for sample, m_value, y_value, fit_value in zip(samples, values["m"], values["y"], fitted):
                if np.isfinite(m_value) and np.isfinite(y_value):
                    value_rows.append({"target": target, "category": label, "locus": row.locus,
                                       "sample_id": sample, "methylation": float(m_value),
                                       "observed": float(y_value), "fitted": float(fit_value),
                                       "residual": float(y_value - fit_value)})
            curve = (row.__getattribute__(f"{target}_intercept")
                     + row.__getattribute__(f"{target}_linear") * grid
                     + row.__getattribute__(f"{target}_quadratic") * grid ** 2)
            curve_rows.extend({"target": target, "category": label, "locus": row.locus,
                               "methylation_grid": float(x), "fitted": float(y)}
                              for x, y in zip(grid, curve))
    pd.DataFrame(selected_rows).to_csv(output / "representative_cpg_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(value_rows).to_csv(output / "representative_atlas_values.tsv", sep="\t", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "representative_fitted_curves.tsv", sep="\t", index=False)
    plot_representative_curves(pd.DataFrame(selected_rows), pd.DataFrame(value_rows),
                               pd.DataFrame(curve_rows), output)


def plot_representative_curves(manifest: pd.DataFrame, values: pd.DataFrame,
                               curves: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for target in TARGETS:
        sites = manifest.loc[manifest.target == target].sort_values("quadratic_quantile")
        fig, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
        for axis, (_, site) in zip(axes.flat, sites.iterrows()):
            point = values.loc[(values.target == target) & (values.locus == site.locus)]
            line = curves.loc[(curves.target == target) & (curves.locus == site.locus)]
            axis.scatter(point.methylation, point.observed, s=10, alpha=0.55, color="#1d4ed8", label="atlas")
            axis.plot(line.methylation_grid, line.fitted, color="#b91c1c", linewidth=2, label="A2H fitted curve")
            axis.set_title(f"{site.locus} ({site.category}); c2={site.quadratic:.3g}")
            axis.set_xlabel("array methylation m")
            axis.set_ylabel(f"atlas {target}")
            axis.set_xlim(0, 1)
            axis.legend(frameon=False, fontsize=8)
        for axis in axes.flat[len(sites):]:
            axis.axis("off")
        fig.savefig(output / f"representative_curves_{target}.png", dpi=300)
        fig.savefig(output / f"representative_curves_{target}.pdf")
        plt.close(fig)


def read_site_metrics(summary_dir: Path, target: str, repeat: int, method: str) -> pd.DataFrame:
    paths = sorted(summary_dir.glob(f"{target}_repeat{repeat:02d}_shard*_{method}.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No site-metric files for {target}, repeat {repeat}, {method}")
    frames = [pd.read_parquet(path, columns=["locus", "n", "pearson_r", "rmse", "bias"])
              for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result.locus = result.locus.astype(str)
    if result.locus.duplicated().any():
        raise ValueError(f"Duplicate site metrics for {target}, repeat {repeat}, {method}")
    return result


def summarize_values(frame: pd.DataFrame, target: str, repeat: int, universe: str,
                     method: str, n_universe: int) -> dict:
    eligible = frame.n >= 2
    correlation = frame.loc[eligible, "pearson_r"].dropna()
    rmse = frame.loc[eligible, "rmse"].dropna()
    nrmse = frame.loc[eligible, "nrmse"].dropna()
    return {
        "target": target, "repeat": repeat, "universe": universe, "method": method,
        "n_loci_universe": n_universe, "n_sites_n_ge_2": int(eligible.sum()),
        "n_sites_corr_defined": len(correlation), "n_sites_nrmse_defined": len(nrmse),
        "mean_pearson_r": float(correlation.mean()), "median_pearson_r": float(correlation.median()),
        "mean_rmse": float(rmse.mean()), "median_rmse": float(rmse.median()),
        "mean_nrmse": float(nrmse.mean()), "median_nrmse": float(nrmse.median()),
        "mean_n_heldout": float(frame.loc[eligible, "n"].mean()),
        "n_zero_truth_range": int((frame.truth_range <= 0).sum()),
    }


def cv_evidence(shards: pd.DataFrame, epic_loci: set[str], truth_ranges: pd.DataFrame,
                summary_dir: Path, output: Path) -> None:
    range_map = truth_ranges.set_index(["target", "locus"])
    atlas_loci = set()
    for shard in shards.itertuples(index=False):
        atlas_loci.update(pd.read_parquet(shard.path, columns=["locus"]).locus.astype(str))
    metric_rows = []
    paired_rows = []
    quantile_rows = []
    source_rows = []
    for target in TARGETS:
        for repeat in range(10):
            method_frames = {}
            for method in METHODS:
                frame = read_site_metrics(summary_dir, target, repeat, method)
                source_rows.extend({"target": target, "repeat": repeat, "method": method,
                                    "path": str(path), "size": path.stat().st_size}
                                   for path in sorted(summary_dir.glob(f"{target}_repeat{repeat:02d}_shard*_{method}.parquet")))
                lookup = range_map.loc[target]
                frame = frame.merge(lookup[["truth_range"]], left_on="locus", right_index=True,
                                    how="left", validate="one_to_one")
                if frame.truth_range.isna().any():
                    raise ValueError(f"Missing truth ranges for {target}, repeat {repeat}, {method}")
                frame["nrmse"] = np.where(frame.truth_range > 0,
                                           frame.rmse / frame.truth_range, np.nan)
                method_frames[method] = frame
            a2h = method_frames["A2H"]
            universal = method_frames["universal_quadratic"]
            aligned = a2h.merge(universal, on="locus", suffixes=("_A2H", "_universal"),
                                validate="one_to_one")
            if not np.array_equal(aligned.n_A2H.to_numpy(), aligned.n_universal.to_numpy()):
                raise ValueError(f"A2H and universal quadratic masks differ at {target}, repeat {repeat}")
            for universe, loci in (("genomewide_atlas", atlas_loci), ("EPIC_BeadChip", epic_loci)):
                for method, frame in method_frames.items():
                    subset = frame.loc[frame.locus.isin(loci)].copy()
                    metric_rows.append(summarize_values(subset, target, repeat, universe, method, len(loci)))
                    eligible = subset.loc[subset.n >= 2]
                    for name in ("pearson_r", "rmse", "nrmse"):
                        values = eligible[name].dropna()
                        for quantile in SITE_QUANTILES:
                            quantile_rows.append({"target": target, "repeat": repeat, "universe": universe,
                                                  "method": method, "metric": name, "quantile": quantile,
                                                  "value": float(values.quantile(quantile))})
                pair = aligned.loc[aligned.locus.isin(loci)].copy()
                eligible = (pair.n_A2H >= 2) & np.isfinite(pair.rmse_A2H) & np.isfinite(pair.rmse_universal)
                eligible_nrmse = eligible & np.isfinite(pair.nrmse_A2H) & np.isfinite(pair.nrmse_universal)
                paired_rows.append({
                    "target": target, "repeat": repeat, "universe": universe,
                    "n_loci_universe": len(loci), "n_sites_common_n_ge_2": int(eligible.sum()),
                    "mean_delta_pearson_A2H_minus_universal": float((pair.loc[eligible, "pearson_r_A2H"] - pair.loc[eligible, "pearson_r_universal"]).mean()),
                    "median_delta_pearson_A2H_minus_universal": float((pair.loc[eligible, "pearson_r_A2H"] - pair.loc[eligible, "pearson_r_universal"]).median()),
                    "mean_delta_rmse_A2H_minus_universal": float((pair.loc[eligible, "rmse_A2H"] - pair.loc[eligible, "rmse_universal"]).mean()),
                    "median_delta_rmse_A2H_minus_universal": float((pair.loc[eligible, "rmse_A2H"] - pair.loc[eligible, "rmse_universal"]).median()),
                    "n_sites_common_nrmse": int(eligible_nrmse.sum()),
                    "mean_delta_nrmse_A2H_minus_universal": float((pair.loc[eligible_nrmse, "nrmse_A2H"] - pair.loc[eligible_nrmse, "nrmse_universal"]).mean()),
                    "median_delta_nrmse_A2H_minus_universal": float((pair.loc[eligible_nrmse, "nrmse_A2H"] - pair.loc[eligible_nrmse, "nrmse_universal"]).median()),
                    "fraction_A2H_lower_rmse": float((pair.loc[eligible, "rmse_A2H"] < pair.loc[eligible, "rmse_universal"]).mean()),
                    "fraction_A2H_lower_nrmse": float((pair.loc[eligible_nrmse, "nrmse_A2H"] < pair.loc[eligible_nrmse, "nrmse_universal"]).mean()),
                })
    metrics = pd.DataFrame(metric_rows)
    paired = pd.DataFrame(paired_rows)
    metrics.to_csv(output / "cv_metrics_by_repeat.tsv", sep="\t", index=False)
    paired.to_csv(output / "cv_matched_A2H_universal_by_repeat.tsv", sep="\t", index=False)
    pd.DataFrame(quantile_rows).to_csv(output / "cv_site_metric_quantiles.tsv", sep="\t", index=False)
    pd.DataFrame(source_rows).drop_duplicates().to_csv(output / "cv_source_site_metric_files.tsv", sep="\t", index=False)
    aggregate = (metrics.groupby(["target", "universe", "method"], as_index=False)
                 .agg(n_repeats=("repeat", "nunique"),
                      mean_of_repeat_mean_pearson_r=("mean_pearson_r", "mean"),
                      sd_of_repeat_mean_pearson_r=("mean_pearson_r", "std"),
                      mean_of_repeat_median_pearson_r=("median_pearson_r", "mean"),
                      mean_of_repeat_mean_rmse=("mean_rmse", "mean"),
                      sd_of_repeat_mean_rmse=("mean_rmse", "std"),
                      mean_of_repeat_median_rmse=("median_rmse", "mean"),
                      mean_of_repeat_mean_nrmse=("mean_nrmse", "mean"),
                      sd_of_repeat_mean_nrmse=("mean_nrmse", "std"),
                      mean_of_repeat_median_nrmse=("median_nrmse", "mean")))
    aggregate.to_csv(output / "cv_metrics_across_repeats.tsv", sep="\t", index=False)


def write_mask_definitions(output: Path, n_coefficients: int, n_atlas: int,
                           n_epic: int) -> None:
    rows = [
        {"analysis": "coefficient_distribution", "universe": "genomewide_deployed",
         "n_loci": n_coefficients, "definition": "All finite deployed A2H coefficient loci."},
        {"analysis": "coefficient_distribution", "universe": "EPIC_BeadChip",
         "n_loci": n_epic, "definition": "Deployed loci with a non-empty EPIC.hg38 probe ID in the authoritative probe map."},
        {"analysis": "atlas_cv", "universe": "genomewide_atlas",
         "n_loci": n_atlas, "definition": "All loci in the prepared historical atlas CV shards."},
        {"analysis": "atlas_cv", "universe": "EPIC_BeadChip",
         "n_loci": n_epic, "definition": "Atlas CV loci intersecting the authoritative EPIC.hg38 locus set."},
        {"analysis": "atlas_cv_common_mask", "universe": "all",
         "n_loci": np.nan, "definition": "Existing site summaries use the same finite held-out observations for A2H and universal quadratic; site eligibility requires n >= 2."},
        {"analysis": "atlas_nrmse", "universe": "all",
         "n_loci": np.nan, "definition": "Per-site held-out RMSE divided by the finite truth range across the 205 atlas samples; zero-range sites are undefined."},
        {"analysis": "atlas_cv_training", "universe": "all",
         "n_loci": np.nan, "definition": "Historical sample-level 100-split, 10-repeat CV; fold-local fits use the existing minimum joint training count of 50."},
    ]
    pd.DataFrame(rows).to_csv(output / "mask_and_metric_definitions.tsv", sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coefficients = load_coefficients(args.coefficients)
    probe, epic_loci = load_probe_map(args.probe_map, set(coefficients.locus))
    coefficients = coefficients.merge(probe, on="locus", how="left", validate="one_to_one")
    coefficients["is_epic"] = coefficients.is_epic.fillna(False).astype(bool)
    if int(coefficients.is_epic.sum()) != len(epic_loci):
        raise ValueError("EPIC locus count changed during coefficient merge")
    shards = load_shards(args.shards)
    samples = load_samples(args.samples)
    split_manifest = pd.read_csv(args.splits, sep="\t")
    if split_manifest.split_index.nunique() != 100:
        raise ValueError("Historical sample split manifest does not contain 100 splits")
    atlas_loci_count = int(shards.n_loci.sum())
    coefficient_summary(coefficients, output)
    write_coefficient_table(coefficients, output)
    plot_coefficient_distributions(coefficients, output)
    truth_ranges = compute_truth_ranges(shards, samples, output)
    representative_curves(coefficients, shards, samples, epic_loci, output, args.minimum_pairs)
    cv_evidence(shards, epic_loci, truth_ranges, args.summary_dir, output)
    write_mask_definitions(output, len(coefficients), atlas_loci_count, len(epic_loci))
    inputs = {
        "deployed_coefficients": str(args.coefficients),
        "probe_map": str(args.probe_map),
        "atlas_shards_manifest": str(args.shards),
        "atlas_samples": str(args.samples),
        "atlas_splits": str(args.splits),
        "historical_cv_site_summary": str(args.summary_dir),
    }
    source_hashes = {
        key: sha256_file(path) for key, path in {
            "deployed_coefficients": args.coefficients,
            "probe_map": args.probe_map,
            "atlas_shards_manifest": args.shards,
            "atlas_samples": args.samples,
            "atlas_splits": args.splits,
        }.items()
    }
    provenance = {
        "analysis": "R1-1 locus-specific A2H calibration audit",
        "status": "complete",
        "training_depth": "CT_count >= 4",
        "model_form": "target = intercept + linear * m + quadratic * m^2",
        "coefficient_storage_order": "[quadratic, linear, intercept], matching np.polyfit/np.polyval",
        "deployed_coefficient_loci": len(coefficients),
        "atlas_cv_loci": atlas_loci_count,
        "epic_beadchip_loci": len(epic_loci),
        "historical_cv_branch": "original/historical_sample",
        "historical_cv_splits": 100,
        "historical_cv_repeats": 10,
        "minimum_joint_training_count": 50,
        "representative_minimum_pairs": args.minimum_pairs,
        "inputs": inputs,
        "source_sha256": source_hashes,
        "interpretation": "CV metrics are equal-locus summaries of existing held-out site metrics. Repeated splits reuse the same atlas subjects and do not provide independent-cohort confidence intervals.",
    }
    output_hashes = {}
    for path in sorted(output.iterdir()):
        if path.name == "provenance.json" or not path.is_file():
            continue
        output_hashes[path.name] = sha256_file(path)
    provenance["output_sha256"] = output_hashes
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, default=json_ready) + "\n")


def main() -> None:
    reviewer = Path(".")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", type=Path,
                        default=Path("legacy_model/A2H_coef.pkl"))
    parser.add_argument("--probe-map", type=Path,
                        default=Path("legacy_model/A2H_cpg_probe.hg38.tsv"))
    parser.add_argument("--shards", type=Path,
                        default=reviewer / "results/atlas_shards_original_full/shards.tsv")
    parser.add_argument("--samples", type=Path,
                        default=reviewer / "results/atlas_splits/atlas_samples.tsv")
    parser.add_argument("--splits", type=Path,
                        default=reviewer / "results/atlas_splits/splits.tsv")
    parser.add_argument("--summary-dir", type=Path,
                        default=reviewer / "results/atlas_cv_full/original/historical_sample_summary")
    parser.add_argument("--output", type=Path,
                        default=reviewer / "results/r1_1_locus_specific_calibration")
    parser.add_argument("--minimum-pairs", type=int, default=50)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
