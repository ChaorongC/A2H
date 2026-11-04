"""Score complete-ten PGP-UK CpGs in disjoint beta-matched groups.

Consumes already-filtered evaluation observations; no support refiltering,
model fitting, imputation, or performance-dependent grouping is performed.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy

from src.evaluate_pgpuk import file_hash, row_stats


TARGETS = ("CHALM", "CAMDA")
METHODS = ("A2H", "m2", "m3")
NUMERIC = ("observed", "A2H", "m", "m2", "m3")
COLUMNS = ["probe_id", "sample_id", *NUMERIC]
COLORS = {"A2H": "#1b6ca8", "m2": "#d95f02", "m3": "#2ca25f"}
LABELS = {"A2H": "A2H", "m2": r"$m^2$", "m3": r"$m^3$"}


def observation_counts(frame: pd.DataFrame, samples: list[str]) -> pd.DataFrame:
    if len(samples) != 10 or len(set(samples)) != 10:
        raise ValueError("Exactly ten unique participant keys are required")
    if not set(COLUMNS).issubset(frame.columns):
        raise ValueError("Missing observation columns")
    if frame[["probe_id", "sample_id"]].isna().any().any():
        raise ValueError("Missing observation keys")
    if frame.duplicated(["probe_id", "sample_id"]).any():
        raise ValueError("Duplicate probe/participant keys")
    if set(frame.sample_id) != set(samples):
        raise ValueError("Observed participant keys do not match the pair manifest")
    valid = np.isfinite(frame[list(NUMERIC)].to_numpy(dtype=np.float64)).all(axis=1)
    present = frame.groupby("probe_id", sort=True).size().rename("n_present")
    finite = pd.Series(valid, index=frame.index).groupby(frame.probe_id).sum()
    result = present.to_frame().assign(n_valid=finite.astype(int))
    result["complete10"] = (result.n_present == 10) & (result.n_valid == 10)
    return result


def aligned_arrays(frame: pd.DataFrame, probes: pd.Index,
                   samples: list[str]) -> dict[str, np.ndarray]:
    subset = frame.loc[frame.probe_id.isin(probes), COLUMNS]
    wide = subset.pivot(index="probe_id", columns="sample_id", values=list(NUMERIC))
    arrays = {name: wide[name].reindex(index=probes, columns=samples).to_numpy(dtype=np.float64)
              for name in NUMERIC}
    if not all(np.isfinite(value).all() and value.shape == (len(probes), 10)
               for value in arrays.values()):
        raise ValueError("Incomplete aligned primary observations")
    if ((arrays["m"] < 0) | (arrays["m"] > 1)).any():
        raise ValueError("Array beta is outside [0, 1]")
    for method, degree in (("m2", 2), ("m3", 3)):
        if not np.allclose(arrays[method], arrays["m"] ** degree, atol=1e-12, rtol=0):
            raise ValueError(f"Saved {method} is not the per-participant beta power")
    return arrays


def make_groups(stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stats.probe_id.duplicated().any():
        raise ValueError("Duplicate CpG statistics")
    if not np.isfinite(stats[["mean_m", "var_m"]].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite grouping statistics")
    ordered = stats.sort_values(["mean_m", "probe_id"], kind="stable").reset_index(drop=True)
    ordered["mean_rank"] = np.arange(len(ordered)) + 1
    ordered["block_id"] = np.arange(len(ordered)) // 100
    ordered = ordered.sort_values(["block_id", "var_m", "mean_m", "probe_id"], kind="stable")
    rank = ordered.groupby("block_id", sort=False).cumcount()
    ordered["group_id"] = ordered.block_id * 10 + rank // 10
    ordered["member_rank"] = rank % 10 + 1
    keep = ordered.groupby("group_id").probe_id.transform("size") == 10
    groups = ordered.loc[keep].sort_values(["group_id", "member_rank"]).reset_index(drop=True)
    remainder = ordered.loc[~keep].copy()
    if len(remainder) > 9 or groups.probe_id.duplicated().any():
        raise ValueError("Invalid remainder or overlapping groups")
    if not groups.groupby("group_id").size().eq(10).all():
        raise ValueError("A retained group does not have ten CpGs")
    if not groups.groupby("group_id").block_id.nunique().eq(1).all():
        raise ValueError("A retained group crosses a mean-sorted block")
    return groups, remainder


def score_arrays(arrays: dict[str, np.ndarray], probes: pd.Index,
                 groups: pd.DataFrame, target: str) -> pd.DataFrame:
    positions = probes.get_indexer(groups.probe_id)
    if (positions < 0).any() or groups.empty:
        raise ValueError("Empty or unaligned groups")
    truth = arrays["observed"][positions].reshape(-1, 100)
    ids = groups.group_id.to_numpy()[::10]
    tables = []
    for method in METHODS:
        predicted = arrays[method][positions].reshape(-1, 100)
        if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
            raise ValueError("Missing data at scoring; no rowwise dropping is allowed")
        metrics = row_stats(truth, predicted)
        if not (metrics["n"] == 100).all():
            raise ValueError("Each primary group must contribute exactly 100 observations")
        tables.append(pd.DataFrame({
            "group_id": ids, "target": target, "method": method,
            "n_cpgs": 10, "n_participants": 10, "n": metrics["n"],
            "pearson_r": metrics["pearson_r"], "rmse": metrics["rmse"],
            "constant_observed": np.ptp(truth, axis=1) == 0,
            "constant_prediction": np.ptp(predicted, axis=1) == 0,
            "correlation_defined": np.isfinite(metrics["pearson_r"]),
        }))
    return pd.concat(tables, ignore_index=True)


def describe(values: np.ndarray) -> dict[str, float | int]:
    valid = values[np.isfinite(values)]
    result = {"n_total": len(values), "n_defined": len(valid), "n_undefined": len(values) - len(valid)}
    if len(valid):
        result.update(mean=float(valid.mean()), sd=float(valid.std(ddof=0)))
        result.update(zip(("min", "p05", "p25", "median", "p75", "p95", "max"),
                          np.quantile(valid, [0, .05, .25, .5, .75, .95, 1]).tolist()))
    else:
        result.update({key: np.nan for key in ("mean", "sd", "min", "p05", "p25", "median", "p75", "p95", "max")})
    return result


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for method in METHODS:
            table = metrics.loc[(metrics.target == target) & (metrics.method == method)]
            for metric in ("pearson_r", "rmse"):
                rows.append({"target": target, "method": method, "metric": metric,
                             **describe(table[metric].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def plots(metrics: pd.DataFrame, quality: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for col, target in enumerate(TARGETS):
        for row, metric, label in ((0, "pearson_r", "Pooled group Pearson correlation"),
                                    (1, "rmse", "Pooled group RMSE")):
            ax = axes[row, col]
            for method in METHODS:
                values = metrics.loc[(metrics.target == target) & (metrics.method == method), metric].to_numpy()
                values = np.sort(values[np.isfinite(values)])
                if len(values):
                    ax.plot(values, np.arange(1, len(values) + 1) / len(values),
                            color=COLORS[method], lw=1.8,
                            label=f"{LABELS[method]}: median {np.median(values):.3f}; n={len(values):,}")
            ax.set(title=target, xlabel=label, ylabel="Cumulative fraction of defined groups", ylim=(0, 1))
            if row == 0:
                ax.set_xlim(-1, 1)
            else:
                ax.set_xlim(left=0)
            ax.legend(fontsize=8, frameon=False, loc="lower right")
            ax.grid(alpha=.25)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"PGP-UK: {quality.shape[0]:,} disjoint ten-CpG groups; ten participants per CpG")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"grouped_distribution_ecdf.{suffix}", dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, column, label in zip(axes, ("mean_m_range", "var_m_range"),
                                  ("Within-group range of CpG mean beta", "Within-group range of CpG beta variance")):
        values = np.sort(quality[column].to_numpy())
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color="#305d65", lw=1.8)
        ax.set(xlabel=label, ylabel="Cumulative fraction of groups", ylim=(0, 1), xlim=(0, None))
        ax.grid(alpha=.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Achieved beta similarity: mean-sorted blocks of 100, variance-sorted groups of ten")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"group_similarity_ecdf.{suffix}", dpi=240)
    plt.close(fig)


def run(input_dir: Path, epic_manifest: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    started = datetime.now(timezone.utc).isoformat()
    inputs = {target: input_dir / f"{target}_observations.parquet" for target in TARGETS}
    inputs.update(pairs=input_dir / "pairs.tsv", source_provenance=input_dir / "provenance.json",
                  epic_manifest=epic_manifest)
    before = {name: file_hash(path) for name, path in inputs.items()}
    source = json.loads(inputs["source_provenance"].read_text())
    if source.get("depth") != 10 or source.get("support_cutoff") != .05:
        raise ValueError("Unexpected inherited source filtering provenance")
    pairs = pd.read_csv(inputs["pairs"], sep="\t")
    if len(pairs) != 10 or pairs.array_id.duplicated().any() or pairs.wgbs_id.duplicated().any():
        raise ValueError("Pair manifest must contain ten unique participant pairs")
    if set(pairs.study) != {"pgpuk"} or set(pairs.tissue) != {"blood"}:
        raise ValueError("Unexpected pair cohort")
    samples = pairs.array_id.tolist()
    manifest = pd.read_csv(epic_manifest, sep="\t", usecols=["probeID", "CpG_chrm", "CpG_beg"])
    mapped = manifest.dropna(subset=["probeID", "CpG_chrm", "CpG_beg"])
    mapped = mapped.loc[mapped.probeID.str.startswith("cg")]
    if mapped.probeID.duplicated().any():
        raise ValueError("Duplicate mapped EPIC probe IDs")
    epic = set(mapped.probeID)
    frames = {target: pd.read_parquet(inputs[target], columns=COLUMNS) for target in TARGETS}
    counts = {target: observation_counts(frame, samples) for target, frame in frames.items()}
    universe = counts["CHALM"].index.union(counts["CAMDA"].index).sort_values()
    audit = pd.DataFrame(index=universe)
    audit.index.name = "probe_id"
    audit["epic_mapped"] = universe.isin(epic)
    for target in TARGETS:
        audit[f"{target}_n_present"] = counts[target].n_present.reindex(universe, fill_value=0)
        audit[f"{target}_n_valid"] = counts[target].n_valid.reindex(universe, fill_value=0)
        audit[f"{target}_complete10"] = counts[target].complete10.reindex(universe, fill_value=False)
    complete = audit.CHALM_complete10 & audit.CAMDA_complete10
    eligible = audit.epic_mapped & complete
    probes = universe[eligible.to_numpy()]
    if len(probes) < 10:
        raise ValueError("Fewer than ten eligible complete-ten EPIC CpGs")
    arrays = {target: aligned_arrays(frames[target], probes, samples) for target in TARGETS}
    if not np.array_equal(arrays["CHALM"]["m"], arrays["CAMDA"]["m"]):
        raise ValueError("CHALM/CAMDA array beta values disagree at aligned keys")
    beta = arrays["CHALM"]["m"]
    stats = pd.DataFrame({"probe_id": probes, "n_participants": 10,
                          "mean_m": beta.mean(axis=1), "var_m": beta.var(axis=1, ddof=0)})
    groups, remainder = make_groups(stats)
    audit["status"] = "incomplete10"
    audit.loc[~audit.epic_mapped, "status"] = "not_mapped_EPIC_CpG"
    audit.loc[eligible, "status"] = "grouping_remainder"
    audit.loc[groups.probe_id, "status"] = "grouped"
    audit = audit.reset_index()
    quality = groups.groupby("group_id", sort=True).agg(
        block_id=("block_id", "first"), n_cpgs=("probe_id", "size"),
        mean_m_min=("mean_m", "min"), mean_m_max=("mean_m", "max"),
        var_m_min=("var_m", "min"), var_m_max=("var_m", "max"),
    ).reset_index()
    quality["mean_m_range"] = quality.mean_m_max - quality.mean_m_min
    quality["var_m_range"] = quality.var_m_max - quality.var_m_min
    quality["n_participants"] = 10
    quality_summary = pd.DataFrame([{"quantity": name, **describe(quality[name].to_numpy())}
                                    for name in ("mean_m_range", "var_m_range")])
    metrics = pd.concat([score_arrays(arrays[target], probes, groups, target) for target in TARGETS], ignore_index=True)
    summary = summarize_metrics(metrics)
    flow = {
        "input_probes_union": len(universe),
        "input_observations_CHALM": len(frames["CHALM"]),
        "input_observations_CAMDA": len(frames["CAMDA"]),
        "input_probes_mapped_EPIC": int(audit.epic_mapped.sum()),
        "input_probes_not_mapped_EPIC": int((~audit.epic_mapped).sum()),
        "common_complete10_before_EPIC": int(complete.sum()),
        "common_complete10_EPIC": len(probes),
        "incomplete10_within_EPIC": int((audit.epic_mapped & ~complete.to_numpy()).sum()),
        "grouped_CpGs": len(groups), "groups": len(quality),
        "remainder_CpGs": len(remainder), "observations_per_group_method": 100,
        "grouped_observations_per_target_method": len(groups) * 10,
    }
    checks = {
        "unique_input_probe_sample_keys": True, "expected_ten_participant_keys": True,
        "all_retained_probes_mapped_EPIC": bool(groups.probe_id.isin(epic).all()),
        "all_retained_probes_complete10_both_targets": True,
        "aligned_CHALM_CAMDA_beta_identical": True, "saved_power_baselines_verified": True,
        "groups_disjoint": not bool(groups.probe_id.duplicated().any()),
        "all_groups_have_ten_CpGs": bool(quality.n_cpgs.eq(10).all()),
        "identical_membership_across_targets_methods": True,
        "all_metric_denominators_100": bool(metrics.n.eq(100).all()),
        "all_RMSE_finite": bool(np.isfinite(metrics.rmse).all()),
        "no_new_support_filter": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Validation failed: {checks}")
    output.mkdir(parents=True, exist_ok=True)
    tables = {"cpg_beta_stats.tsv": stats, "groups.tsv": groups, "group_quality.tsv": quality,
              "group_quality_summary.tsv": quality_summary, "group_metrics.tsv": metrics,
              "summary.tsv": summary, "eligibility.tsv": audit,
              "exclusions.tsv": audit.loc[audit.status != "grouped"],
              "flow.tsv": pd.DataFrame([flow])}
    for name, table in tables.items():
        table.to_csv(output / name, sep="\t", index=False)
    plots(metrics, quality, output)
    result_lines = []
    for target in TARGETS:
        for method in METHODS:
            rows = summary.loc[(summary.target == target) & (summary.method == method)].set_index("metric")
            result_lines.append(f"| {target} | {method} | {rows.loc['pearson_r', 'median']:.6f} | "
                                f"{rows.loc['pearson_r', 'n_defined']:.0f} | {rows.loc['rmse', 'median']:.6f} |")
    report = (
        "# Complete-ten PGP-UK disjoint-group evaluation\n\n"
        f"Generated UTC: {started}\n\n"
        "Ten PGP-UK blood participants; existing atlas-trained predictions and already-filtered "
        "depth-10 WGBS observations. No model refit or additional support filtering.\n\n"
        f"Of {len(universe):,} input evaluation CpGs, {flow['input_probes_mapped_EPIC']:,} "
        f"are mapped EPIC CpGs and {len(probes):,} have complete evaluation data for all ten "
        f"participants in both targets. {len(groups):,} CpGs form {len(quality):,} disjoint groups; "
        f"{len(remainder)} remainder CpGs are ungrouped. Each group/method has 100 observations.\n\n"
        "## Group definition\n\n"
        "Sort by ten-subject mean beta and probe ID; form consecutive blocks of 100; sort each "
        "block by beta variance (ddof=0), mean beta and probe ID; form consecutive tens. "
        "The final partial block is processed; incomplete groups are excluded. Membership is "
        "identical across targets and methods. Powers are of each subject's beta, not its cohort mean.\n\n"
        "## Group-level distribution medians\n\n"
        "| Target | Method | Pearson median | Defined Pearson groups | RMSE median |\n"
        "|---|---|---:|---:|---:|\n" + "\n".join(result_lines) + "\n\n"
        "RMSE is sqrt(mean squared error across the 100 observations), not mean site RMSE. "
        "Undefined correlations remain missing; their group RMSEs remain included. Full "
        "distributions and their denominators are in group_metrics.tsv and summary.tsv.\n\n"
        "## Achieved beta similarity\n\n" + "\n".join(
            f"- {r.quantity}: median {r.median:.8g}, p95 {r.p95:.8g}, maximum {r.max:.8g}."
            for r in quality_summary.itertuples(index=False)
        ) + "\n\n"
        "This rank-based construction does not enforce an absolute matching tolerance. It "
        "matches array-beta distributions, not observed CHALM/CAMDA distributions; pooled "
        "group correlation is not proof of within-CpG subject ordering. No performance-based "
        "selection or regrouping is performed. Raw data, historical >=8-subject analyses and "
        "the locked external prediction branch remain unchanged. Exclusions are documented "
        "per probe in eligibility.tsv and exclusions.tsv. Probes already absent from the source "
        "evaluation tables are outside this grouping audit's denominator.\n"
    )
    (output / "report.md").write_text(report)
    checks["source_files_unchanged"] = all(file_hash(path) == before[name] for name, path in inputs.items())
    if not checks["source_files_unchanged"]:
        raise ValueError("Source files changed during evaluation")
    (output / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    code_paths = {"runner": Path(__file__), "metric_helper": Path(__file__).with_name("evaluate_pgpuk.py")}
    provenance = {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, "-m", "src.run_pgpuk_disjoint_complete10", *sys.argv[1:]]),
        "cwd": str(Path.cwd()), "samples_in_manifest_order": samples,
        "inputs": {name: {"path": str(path.resolve()), "sha256": before[name]} for name, path in inputs.items()},
        "code": {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in code_paths.items()},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "pyarrow": pyarrow.__version__, "matplotlib": matplotlib.__version__},
        "scope": "existing evaluation-eligible, mapped EPIC CpGs, complete for ten participants in both targets",
        "EPIC_key": "probeID; nonmissing CpG_chrm and CpG_beg; cg probe IDs",
        "EPIC_manifest_rows": len(manifest), "mapped_EPIC_CpG_ids": len(epic),
        "inherited_depth": 10, "inherited_support_cutoff": .05, "new_support_filter_applied": False,
        "grouping": {"mean_block_size": 100, "group_size": 10, "variance_ddof": 0,
                     "mean_sort_keys": ["mean_m", "probe_id"],
                     "block_sort_keys": ["var_m", "mean_m", "probe_id"],
                     "final_partial_block": "retain complete tens; exclude remainder",
                     "performance_based_selection": False},
        "metrics": {"correlation": "Pearson over 100 pairs per group", "rmse": "sqrt(mean((observed-predicted)^2))",
                    "undefined_correlation": "preserve missing; retain group RMSE", "dtype": "float64"},
        "flow": flow, "validation": checks,
        "output_sha256": {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(flow, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/pgpuk_depth10_calibrated"))
    parser.add_argument("--epic-manifest", type=Path,
                        default=Path("reference/EPIC.hg38.manifest.gencode.v36.tsv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir, args.epic_manifest, args.output_dir)


if __name__ == "__main__":
    main()
