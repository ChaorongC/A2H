"""Evaluate paired PGP-UK A2H predictions without pooling CpGs for primary metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


DEFAULT_SOURCE = Path("external_validation")


def truth_root(source: Path, min_depth: int) -> Path:
    namespace = "paired_validation_v2" if min_depth == 4 else f"paired_validation_v2_depth{min_depth}"
    return source / "results" / namespace / "wgbs/probe_space"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matrix(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", index_col=0)
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate probes in {path}")
    if not frame.columns.is_unique or set(columns) != set(frame.columns):
        raise ValueError(f"Unexpected sample columns in {path}: {list(frame.columns)}")
    return frame.loc[:, columns]


def map_array_samples(beta: pd.DataFrame, manifest: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    green = manifest.loc[(manifest["study"] == "pgpuk") & (manifest["channel"] == "Grn")]
    mapping = {}
    for row in green.itertuples(index=False):
        name = Path(row.local_path).name.removesuffix("_Grn.idat.gz")
        if name in mapping:
            raise ValueError(f"Duplicate IDAT basename: {name}")
        mapping[name] = row.array_id
    if set(mapping) != set(beta.columns) or set(mapping.values()) != set(columns):
        raise ValueError("IDAT manifest does not map the complete PGP-UK array matrix")
    return beta.rename(columns=mapping).loc[:, columns]


def row_stats(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.isfinite(y) & np.isfinite(pred)
    n = valid.sum(axis=1)
    yy = np.where(valid, y, 0.0)
    pp = np.where(valid, pred, 0.0)
    safe_n = np.maximum(n, 1)
    mean_y = yy.sum(axis=1) / safe_n
    mean_p = pp.sum(axis=1) / safe_n
    yc = np.where(valid, y - mean_y[:, None], 0.0)
    pc = np.where(valid, pred - mean_p[:, None], 0.0)
    covariance = (yc * pc).sum(axis=1)
    scale = np.sqrt((yc * yc).sum(axis=1) * (pc * pc).sum(axis=1))
    y_varies = np.where(valid, y, -np.inf).max(axis=1) > np.where(valid, y, np.inf).min(axis=1)
    p_varies = np.where(valid, pred, -np.inf).max(axis=1) > np.where(valid, pred, np.inf).min(axis=1)
    pearson = np.full(len(n), np.nan)
    np.divide(covariance, scale, out=pearson, where=(n >= 2) & y_varies & p_varies & (scale > 0))
    error = np.where(valid, y - pred, 0.0)
    rmse = np.sqrt((error * error).sum(axis=1) / safe_n)
    bias = error.sum(axis=1) / safe_n
    rmse[n == 0] = np.nan
    bias[n == 0] = np.nan
    return {"n": n, "pearson_r": pearson, "rmse": rmse, "bias": bias}


def spearman_rows(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    valid = np.isfinite(y) & np.isfinite(pred)
    y_rank = rankdata(np.where(valid, y, np.nan), axis=1, nan_policy="omit")
    p_rank = rankdata(np.where(valid, pred, np.nan), axis=1, nan_policy="omit")
    return row_stats(y_rank, p_rank)["pearson_r"]


def scalar_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y) & np.isfinite(pred)
    yy = y[valid]
    pp = pred[valid]
    if len(yy) == 0:
        return {"n": 0, "pearson_r": np.nan, "rmse": np.nan, "nrmse": np.nan}
    rmse = float(np.sqrt(np.mean((yy - pp) ** 2)))
    spread = float(np.ptp(yy))
    correlation = float(np.corrcoef(yy, pp)[0, 1]) if len(yy) >= 2 and np.ptp(yy) > 0 and np.ptp(pp) > 0 else np.nan
    return {
        "n": len(yy),
        "pearson_r": correlation,
        "rmse": rmse,
        "nrmse": rmse / spread if spread > 0 else np.nan,
    }


def pairing_diagnostic(beta: pd.DataFrame, wgbs_m: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    common = beta.index.intersection(wgbs_m.index, sort=False)
    a = beta.loc[common, columns].to_numpy(dtype=float)
    w = wgbs_m.loc[common, columns].to_numpy(dtype=float)
    complete = np.isfinite(a).all(axis=1) & np.isfinite(w).all(axis=1)
    complete &= (np.std(a, axis=1) >= 0.03) & (np.std(w, axis=1) >= 0.03)
    a = a[complete]
    w = w[complete]
    if len(a) < 100:
        raise ValueError("Too few variable complete CpGs for pairing diagnostic")
    a -= a.mean(axis=1, keepdims=True)
    w -= w.mean(axis=1, keepdims=True)
    corr = np.corrcoef(a.T, w.T)[: len(columns), len(columns) :]
    rows, best = linear_sum_assignment(-corr)
    rows_out = []
    for i, array_id in enumerate(columns):
        for j, wgbs_id in enumerate(columns):
            rows_out.append({
                "array_id": array_id,
                "wgbs_id": wgbs_id,
                "centered_methylation_r": corr[i, j],
                "declared_pair": i == j,
                "best_global_assignment": j == best[np.where(rows == i)[0][0]],
                "diagnostic_cpg_count": len(a),
            })
    return pd.DataFrame(rows_out)


def group_sites(beta: np.ndarray, sites: pd.Index,
                reference_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    if reference_stats is None:
        mean = np.nanmean(beta, axis=1)
        variance = np.nanvar(beta, axis=1)
    else:
        if reference_stats.probe_id.duplicated().any():
            raise ValueError("Duplicate probe in reference grouping statistics")
        aligned = reference_stats.set_index("probe_id").reindex(sites)
        mean = aligned.mean_m.to_numpy(dtype=float)
        variance = aligned.var_m.to_numpy(dtype=float)
    valid = np.isfinite(mean) & np.isfinite(variance)
    ordered = pd.DataFrame({"probe_id": sites[valid], "mean_m": mean[valid], "var_m": variance[valid]})
    ordered["mean_bin"] = np.floor(np.clip(ordered["mean_m"], 0, 1) / 0.02).astype(int)
    ordered["variance_bin"] = pd.qcut(ordered["var_m"], q=10, labels=False, duplicates="drop").fillna(0).astype(int)
    ordered = ordered.sort_values(["mean_bin", "variance_bin", "mean_m", "var_m", "probe_id"])
    ordered["rank_in_bin"] = ordered.groupby(["mean_bin", "variance_bin"], observed=True).cumcount()
    ordered["group_key"] = list(zip(ordered.mean_bin, ordered.variance_bin, ordered.rank_in_bin // 10))
    sizes = ordered.groupby("group_key").size()
    ordered = ordered.loc[ordered.group_key.map(sizes) == 10].copy()
    ordered["group_id"] = pd.factorize(ordered.group_key, sort=False)[0]
    return ordered.drop(columns=["rank_in_bin", "group_key"])


def reference_control_predictions(
    beta: np.ndarray, loci: pd.Series, controls: pd.DataFrame, coefficients: list[float]
) -> dict[str, np.ndarray]:
    if len(beta) != len(loci) or len(coefficients) != 3 or controls.locus.duplicated().any():
        raise ValueError("Invalid reference-control mapping or quadratic coefficients")
    aligned = controls.set_index("locus").reindex(loci.to_numpy())
    mean = aligned.reference_mean.to_numpy(dtype=float)
    intercept = aligned.linear_intercept.to_numpy(dtype=float)
    slope = aligned.linear_slope.to_numpy(dtype=float)
    return {
        "reference_mean": np.broadcast_to(mean[:, None], beta.shape),
        "reference_linear": intercept[:, None] + slope[:, None] * beta,
        "universal_quadratic": coefficients[0] + coefficients[1] * beta + coefficients[2] * beta**2,
    }


def paired_site_differences(site: pd.DataFrame) -> pd.DataFrame:
    eligible = site.loc[site.eligible, ["probe_id", "target", "method", "n", "rmse", "spearman_rho"]]
    if eligible.duplicated(["probe_id", "target", "method"]).any():
        raise ValueError("Duplicate target-site-method key")
    main = eligible.loc[eligible.method == "A2H"].set_index(["target", "probe_id"])
    rows = []
    for method in sorted(set(eligible.method) - {"A2H"}):
        control = eligible.loc[eligible.method == method].set_index(["target", "probe_id"])
        if not main.index.equals(control.index) or not np.array_equal(main.n.to_numpy(), control.n.to_numpy()):
            raise ValueError(f"A2H and {method} do not use the same site and subject counts")
        for target in main.index.get_level_values("target").unique():
            a = main.loc[target]
            b = control.loc[target]
            rmse_diff = a.rmse.to_numpy() - b.rmse.to_numpy()
            rho_diff = a.spearman_rho.to_numpy() - b.spearman_rho.to_numpy()
            rows.append({
                "target": target,
                "control": method,
                "n_sites": len(rmse_diff),
                "mean_site_rmse_A2H_minus_control": float(np.mean(rmse_diff)),
                "median_site_rmse_A2H_minus_control": float(np.median(rmse_diff)),
                "fraction_sites_lower_rmse_A2H": float(np.mean(rmse_diff < 0)),
                "n_sites_defined_rho_both": int(np.isfinite(rho_diff).sum()),
                "median_site_spearman_A2H_minus_control": float(np.nanmedian(rho_diff)) if np.isfinite(rho_diff).any() else np.nan,
            })
    return pd.DataFrame(rows)


def observation_table(probes: pd.Index, samples: list[str], observed: np.ndarray,
                      methods: dict[str, np.ndarray], mask: np.ndarray,
                      min_subjects: int) -> pd.DataFrame:
    if not all(value.shape == observed.shape for value in methods.values()) or mask.shape != observed.shape:
        raise ValueError("Unaligned observations and method predictions")
    sites = mask.sum(axis=1) >= min_subjects
    selected = mask[sites].ravel()
    table = pd.DataFrame({
        "probe_id": np.repeat(probes[sites].to_numpy(), len(samples))[selected],
        "sample_id": np.tile(samples, int(sites.sum()))[selected],
        "observed": observed[sites].ravel()[selected],
    })
    for method, prediction in methods.items():
        table[method] = prediction[sites].ravel()[selected]
    if table.duplicated(["probe_id", "sample_id"]).any():
        raise ValueError("Duplicate site-subject observation")
    return table


def score_grouped(y: np.ndarray, pred: np.ndarray, group_rows: np.ndarray, group_ids: np.ndarray) -> pd.DataFrame:
    y_group = y[group_rows].reshape(-1, 100)
    p_group = pred[group_rows].reshape(-1, 100)
    scores = row_stats(y_group, p_group)
    valid_sites = (np.isfinite(y[group_rows]) & np.isfinite(pred[group_rows])).sum(axis=1).reshape(-1, 10)
    keep = (valid_sites >= 8).all(axis=1)
    return pd.DataFrame({
        "group_id": group_ids[::10],
        "n": scores["n"],
        "pearson_r": scores["pearson_r"],
        "rmse": scores["rmse"],
        "all_sites_ge8": keep,
    })


def plot_samples(truth: dict[str, np.ndarray], prediction: dict[str, np.ndarray], columns: list[str], output: Path) -> None:
    fig, axes = plt.subplots(2, len(columns), figsize=(28, 6), sharex="row", sharey="row", constrained_layout=True)
    for row, target in enumerate(("CHALM", "CAMDA")):
        for col, subject in enumerate(columns):
            ax = axes[row, col]
            y = truth[target][:, col]
            p = prediction[target][:, col]
            metrics = scalar_metrics(y, p)
            valid = np.isfinite(y) & np.isfinite(p)
            ax.hexbin(y[valid], p[valid], gridsize=42, mincnt=1, bins="log", cmap="viridis", extent=(0, 1, 0, 1))
            ax.plot([0, 1], [0, 1], color="#c33838", lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(subject, fontsize=9)
            ax.text(0.04, 0.96, f"n={metrics['n']:,}\nr={metrics['pearson_r']:.2f}\nRMSE={metrics['rmse']:.3f}",
                    transform=ax.transAxes, va="top", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
            if col == 0:
                ax.set_ylabel(f"Predicted {target}")
            if row == 1:
                ax.set_xlabel("WGBS observed")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def evaluate(source: Path, output: Path, min_depth: int, support_cutoff: float, min_subjects: int,
             control_root: Path | None = None, save_observations: bool = False,
             grouping_reference: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paired_path = source / "metadata/paired_validation_v2/paired_samples.tsv"
    idat_path = source / "metadata/paired_validation_v2/paired_array_idat_manifest.tsv"
    paired = pd.read_csv(paired_path, sep="\t").query("study == 'pgpuk'").copy()
    if len(paired) != 10 or paired.array_id.duplicated().any() or paired.wgbs_id.duplicated().any():
        raise ValueError("Expected ten unique PGP-UK blood pairs")
    columns = paired.array_id.to_list()
    if set(paired.tissue) != {"blood"}:
        raise ValueError("This implementation expects the ten blood pairs")
    base = source / "results/paired_validation_v2"
    truth_base = truth_root(source, min_depth)
    paths = {
        "pairs": paired_path,
        "idat_manifest": idat_path,
        "beta": base / "array/pgpuk_450K.beta.tsv",
        "confidence": base / "a2h_array/pgpuk_450K.confidence.tsv",
        "pred_CHALM": base / "a2h_array/pgpuk_450K.chalm.tsv",
        "pred_CAMDA": base / "a2h_array/pgpuk_450K.camda.tsv",
        "truth_m": truth_base / "pgpuk_450K.traditional_probe_matrix.tsv",
        "truth_CHALM": truth_base / "pgpuk_450K.chalm_probe_matrix.tsv",
        "truth_CAMDA": truth_base / "pgpuk_450K.camda_probe_matrix.tsv",
    }
    idat = pd.read_csv(idat_path, sep="\t")
    raw_beta = pd.read_csv(paths["beta"], sep="\t", index_col=0)
    beta = map_array_samples(raw_beta, idat, columns)
    if beta.index.has_duplicates:
        raise ValueError("Duplicate array probe IDs")
    frames = {name: load_matrix(path, columns) for name, path in paths.items() if name not in {"pairs", "idat_manifest", "beta"}}
    pairing = pairing_diagnostic(beta, frames["truth_m"], columns)
    pairing.to_csv(output / "pairing_diagnostic.tsv", sep="\t", index=False)
    paired.assign(pairing_status="metadata_pair; methylation_diagnostic_only").to_csv(output / "pairs.tsv", sep="\t", index=False)

    common = beta.index
    for name in ("confidence", "pred_CHALM", "pred_CAMDA", "truth_CHALM", "truth_CAMDA"):
        common = common.intersection(frames[name].index, sort=False)
    if len(common) == 0:
        raise ValueError("No common CpG probes")
    beta_array = beta.loc[common, columns].to_numpy(dtype=float)
    support = frames["confidence"].loc[common, columns].to_numpy(dtype=float)
    reference_controls = {}
    if control_root is not None:
        control_manifest_path = control_root / "provenance.json"
        control_manifest = json.loads(control_manifest_path.read_text())
        probe_map_path = Path(control_manifest["probe_map"])
        if file_hash(probe_map_path) != control_manifest["file_sha256"]["probe_map"]:
            raise ValueError("Reference-control probe map has changed since fitting")
        mapping = pd.read_csv(probe_map_path, sep="\t", usecols=["loci.hg38", "HM450.hg38"]).dropna().drop_duplicates()
        if mapping["HM450.hg38"].duplicated().any():
            raise ValueError("Ambiguous HM450 probe mapping")
        loci = mapping.set_index("HM450.hg38")["loci.hg38"].reindex(common)
        paths["control_manifest"] = control_manifest_path
        paths["control_probe_map"] = probe_map_path
        for target in ("CHALM", "CAMDA"):
            control_path = control_root / f"{target}_controls.parquet"
            paths[f"control_{target}"] = control_path
            controls = pd.read_parquet(control_path)
            coefficients = control_manifest[target]["universal_coefficients_intercept_linear_quadratic"]
            reference_controls[target] = reference_control_predictions(beta_array, loci, controls, coefficients)
    target_arrays = {}
    common_masks = {}
    eligible_for_groups = np.ones(len(common), dtype=bool)
    for target in ("CHALM", "CAMDA"):
        observed = frames[f"truth_{target}"].loc[common, columns].to_numpy(dtype=float)
        predicted = frames[f"pred_{target}"].loc[common, columns].to_numpy(dtype=float)
        mask = np.isfinite(beta_array) & np.isfinite(observed) & np.isfinite(predicted)
        if support_cutoff > 0:
            mask &= np.isfinite(support) & (support >= support_cutoff)
        if control_root is not None:
            for control_pred in reference_controls[target].values():
                mask &= np.isfinite(control_pred)
        target_arrays[target] = (observed, predicted)
        common_masks[target] = mask
        eligible_for_groups &= mask.sum(axis=1) >= min_subjects
    reference_stats = pd.read_parquet(grouping_reference) if grouping_reference is not None else None
    if grouping_reference is not None:
        paths["grouping_reference"] = grouping_reference
    group_map = group_sites(beta_array[eligible_for_groups], common[eligible_for_groups], reference_stats)
    group_map.to_csv(output / "groups.tsv", sep="\t", index=False)
    lookup = pd.Series(np.arange(len(common)), index=common)
    grouped = group_map.sort_values(["group_id", "probe_id"])
    group_rows = lookup.loc[grouped.probe_id].to_numpy(dtype=int)
    group_ids = grouped.group_id.to_numpy(dtype=int)

    site_tables = []
    group_tables = []
    sample_tables = []
    pooled_tables = []
    figure_truth = {}
    figure_pred = {}
    flow = []
    for target in ("CHALM", "CAMDA"):
        y, a2h = target_arrays[target]
        common_mask = common_masks[target]
        yy = np.where(common_mask, y, np.nan)
        methods = {
            "A2H": a2h,
            "m": beta_array,
            "m2": beta_array**2,
            "m3": beta_array**3,
            **reference_controls.get(target, {}),
        }
        if save_observations:
            observation_table(common, columns, y, methods, common_mask, min_subjects).to_parquet(
                output / f"{target}_observations.parquet", index=False
            )
        flow.append({
            "target": target, "depth": min_depth, "support_cutoff": support_cutoff,
            "platform_probes": len(beta), "common_probes": len(common),
            "common_observations": int(common_mask.sum()),
            "sites_ge_min_subjects": int((common_mask.sum(axis=1) >= min_subjects).sum()),
            "complete_ten_sites": int((common_mask.sum(axis=1) == 10).sum()),
            "disjoint_matched_groups": len(grouped) // 10,
            "reference_controls": control_root is not None,
        })
        eligible_sites = common_mask.sum(axis=1) >= min_subjects
        figure_truth[target] = np.where(eligible_sites[:, None], yy, np.nan)
        figure_pred[target] = np.where(eligible_sites[:, None] & common_mask, a2h, np.nan)
        for method, raw_pred in methods.items():
            pred = np.where(common_mask, raw_pred, np.nan)
            scores = row_stats(yy, pred)
            scores["spearman_rho"] = spearman_rows(yy, pred)
            table = pd.DataFrame(scores)
            table.insert(0, "probe_id", common)
            table.insert(1, "target", target)
            table.insert(2, "method", method)
            table["eligible"] = table.n >= min_subjects
            site_tables.append(table)
            group_score = score_grouped(yy, pred, group_rows, group_ids)
            group_score.insert(1, "target", target)
            group_score.insert(2, "method", method)
            group_tables.append(group_score)
            all_sites = table.eligible.to_numpy()
            pooled_tables.append({"target": target, "method": method, "unit": "CpG-sample (descriptive)", **scalar_metrics(yy[all_sites], pred[all_sites])})
            for col, subject in enumerate(columns):
                sample_tables.append({"target": target, "method": method, "sample_id": subject, **scalar_metrics(yy[all_sites, col], pred[all_sites, col])})
    site = pd.concat(site_tables, ignore_index=True)
    grouped_scores = pd.concat(group_tables, ignore_index=True)
    site.to_csv(output / "site_metrics.tsv", sep="\t", index=False)
    grouped_scores.to_csv(output / "group_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(sample_tables).to_csv(output / "sample_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(pooled_tables).to_csv(output / "pooled_descriptive_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(flow).to_csv(output / "flow.tsv", sep="\t", index=False)
    summary = (
        site.loc[site.eligible]
        .groupby(["target", "method"], observed=True)
        .agg(n_sites=("probe_id", "size"), median_spearman=("spearman_rho", "median"),
             median_pearson=("pearson_r", "median"), mean_site_rmse=("rmse", "mean"),
             median_site_rmse=("rmse", "median"))
        .reset_index()
    )
    summary.to_csv(output / "site_summary.tsv", sep="\t", index=False)
    paired_site_differences(site).to_csv(output / "method_differences.tsv", sep="\t", index=False)
    plot_samples(figure_truth, figure_pred, columns, output / f"per_sample_a2h_depth{min_depth}.png")
    provenance = {
        "source_root": str(source), "depth": min_depth, "support_cutoff": support_cutoff,
        "min_subjects": min_subjects,
        "grouping": ("atlas blood-reference m mean width 0.02; variance deciles; consecutive disjoint groups of 10"
                     if grouping_reference is not None else
                     "external array-m mean width 0.02; variance deciles; consecutive disjoint groups of 10"),
        "grouping_reference": str(grouping_reference) if grouping_reference is not None else None,
        "control_root": str(control_root) if control_root is not None else None,
        "saved_observations": save_observations,
        "pairing_note": "metadata mapping only; centered methylation diagnostic does not prove genotype identity",
        "file_sha256": {name: file_hash(path) for name, path in paths.items()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=10, choices=[4, 8, 10])
    parser.add_argument("--support-cutoff", type=float, default=0.05)
    parser.add_argument("--min-subjects", type=int, default=8)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--save-observations", action="store_true")
    parser.add_argument("--grouping-reference", type=Path)
    args = parser.parse_args()
    evaluate(args.source_root, args.output_dir, args.depth, args.support_cutoff, args.min_subjects,
             args.control_root, args.save_observations, args.grouping_reference)


if __name__ == "__main__":
    main()
