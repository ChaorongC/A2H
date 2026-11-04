"""Verify shard outputs and summarize the frozen tissue-only R1-5/R1-6 arms."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.tissue_only_transfer import (DEFAULT_ROOT, METHODS, MODELS, TARGETS,
    members, record_file, utc, verify_file, write_json)

ALL_MODELS = ("full_reference",) + MODELS
KEYS = ["probe_id", "sample_id"]


def reconstruct_sites(frame):
    """Independent pandas reduction, without the fitter's row_stats helper."""
    if frame.duplicated(["locus", "sample_id"]).any():
        raise ValueError("Duplicate held-out key")
    valid = np.isfinite(frame[["observed", *METHODS]]).all(axis=1)
    if not np.array_equal(valid, frame.evaluation_valid):
        raise ValueError("Saved evaluation mask mismatch")
    parts = []
    for method in METHODS:
        errors = frame[["locus"]].copy()
        errors["squared_error"] = np.where(valid, (frame[method] - frame.observed)**2, np.nan)
        result = errors.groupby("locus", sort=False).squared_error.agg(["count", "mean"])
        result = result.rename(columns={"count": "n", "mean": "mse"}).reset_index()
        result["rmse"] = np.sqrt(result.pop("mse"))
        result["method"] = method
        result["eligible"] = result.n >= 8
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def align_external(frames, probe_ids):
    """Use an identical finite probe/participant mask across every model/method."""
    base = None
    input_rows = []
    for model in ALL_MODELS:
        frame = frames[model].loc[frames[model].probe_id.isin(probe_ids)].copy()
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate external key: {model}")
        input_rows.append({"stage": f"input_{model}", "n_probes": frame.probe_id.nunique(), "n_rows": len(frame)})
        frame = frame.set_index(KEYS).sort_index()
        if base is None:
            base = frame[["m", "observed"]].copy()
        else:
            if not base.index.equals(frame.index):
                raise ValueError(f"External key universe differs: {model}")
            np.testing.assert_allclose(base[["m", "observed"]], frame[["m", "observed"]],
                                       atol=0, rtol=0, equal_nan=True)
        for method in ("A2H", "reference_linear"):
            base[f"{model}__{method}"] = frame[method]
    valid = np.isfinite(base).all(axis=1)
    counts = valid.groupby(level="probe_id").sum()
    eligible = counts.index[counts >= 8]
    audit = counts.rename("n_common_observations").reset_index()
    audit["eligible"] = audit.n_common_observations >= 8
    audit["exclusion"] = np.where(audit.eligible, "", "fewer_than_8_common_observations")
    final = base.loc[valid & base.index.get_level_values("probe_id").isin(eligible)].reset_index()
    input_rows += [{"stage": "finite_common_rows", "n_probes": base.loc[valid].index.get_level_values(0).nunique(), "n_rows": int(valid.sum())},
                   {"stage": "eligible_common_rows", "n_probes": len(eligible), "n_rows": len(final)}]
    if final.empty:
        raise ValueError("No common external evaluation observations")
    return final, audit, pd.DataFrame(input_rows)


def external_metrics(common):
    rows = []
    for model in ALL_MODELS:
        for method in METHODS:
            col = "m" if method == "m" else f"{model}__{method}"
            errors = common[KEYS].copy()
            errors["se"] = (common[col] - common.observed)**2
            stats = errors.groupby("probe_id", sort=True).se.agg(["count", "mean"])
            stats = stats.rename(columns={"count": "n", "mean": "mse"}).reset_index()
            stats["rmse"] = np.sqrt(stats.pop("mse"))
            stats["model"], stats["method"] = model, method
            rows.append(stats)
    return pd.concat(rows, ignore_index=True)


def bootstrap(common, samples, site, replicates=1000, seed=2026):
    probes = pd.Index(sorted(common.probe_id.unique()))
    index = pd.MultiIndex.from_product([probes, samples], names=KEYS)
    aligned = common.set_index(KEYS).reindex(index)
    y = aligned.observed.to_numpy().reshape(len(probes), len(samples))
    valid = np.isfinite(y)
    if (valid.sum(axis=1) < 8).any():
        raise ValueError("External bootstrap includes an ineligible site")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(samples), size=(replicates, len(samples)))
    weights = np.eye(len(samples), dtype=float)[draws].sum(axis=1).T
    predictions = {f"{model}__{method}": aligned["m" if method == "m" else f"{model}__{method}"].to_numpy().reshape(y.shape)
                   for model in ALL_MODELS for method in METHODS}
    errors = {name: np.where(valid, (pred-y)**2, 0) for name, pred in predictions.items()}
    for pred in predictions.values():
        if not np.isfinite(pred[valid]).all():
            raise ValueError("Bootstrap predictions mismatch common mask")
    values = {name: np.empty(replicates) for name in errors}
    lost = np.empty(replicates, dtype=int)
    for start in range(0, replicates, 25):
        end = min(start+25, replicates)
        w = weights[:, start:end]
        counts = valid.astype(float) @ w
        usable = counts > 0
        if not usable.any(axis=0).all():
            raise ValueError("Bootstrap draw has no available sites")
        lost[start:end] = (~usable).sum(axis=0)
        for name, error in errors.items():
            per_site = np.where(usable, np.sqrt((error @ w) / np.maximum(counts, 1)), np.nan)
            values[name][start:end] = np.nanmean(per_site, axis=0)
    points = site.groupby(["model", "method"]).rmse.mean()
    summary = []
    for model in ALL_MODELS:
        for method in METHODS:
            name = f"{model}__{method}"
            independent_point = np.sqrt(errors[name].sum(axis=1) / valid.sum(axis=1)).mean()
            np.testing.assert_allclose(independent_point, points.loc[model, method], atol=1e-12, rtol=0)
            delta = values[name] - values[f"full_reference__{method}"]
            lo, hi = np.quantile(values[name], [.025, .975])
            dlo, dhi = np.quantile(delta, [.025, .975])
            summary.append({"model": model, "method": method, "n_sites": len(probes), "n_observations": int(valid.sum()),
                "mean_site_rmse": independent_point, "ci_low": lo, "ci_high": hi,
                "delta_vs_full": independent_point-points.loc["full_reference", method],
                "delta_ci_low": dlo, "delta_ci_high": dhi, "bootstrap_replicates": replicates,
                "draws_with_empty_sites": int((lost>0).sum()), "max_empty_sites": int(lost.max())})
    draws_table = pd.DataFrame(draws, columns=samples)
    # Each column records the index of the resampled participant at that draw position.
    draws_table.columns = [f"draw_position_{i}" for i in range(len(samples))]
    draws_table["empty_sites"] = lost
    return pd.DataFrame(summary), pd.DataFrame(values), draws_table


def figures(external, heldout, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Full atlas\n205 samples", "No blood/immune\n161 samples", "No epithelial\n122 samples"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for col, target in enumerate(TARGETS):
        ax = axes[0, col]
        p = external.loc[(external.target == target) & (external.method == "A2H")].set_index("model").loc[list(ALL_MODELS)]
        x = np.arange(3)
        ax.vlines(x, p.ci_low, p.ci_high, color="#276A88", linewidth=2)
        ax.scatter(x, p.mean_site_rmse, color="#276A88", zorder=3)
        ax.set(xticks=x, xticklabels=labels, ylabel="Mean per-CpG RMSE", title=f"{target}: PGP-UK blood")
        ax.spines[["top", "right"]].set_visible(False)
        ax = axes[1, col]
        q = p.loc[list(MODELS)]
        ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
        ax.vlines(np.arange(2), q.delta_ci_low, q.delta_ci_high, color="#BA563A", linewidth=2)
        ax.scatter(np.arange(2), q.delta_vs_full, color="#BA563A", zorder=3)
        ax.set(xticks=np.arange(2), xticklabels=labels[1:], ylabel="RMSE difference versus full atlas",
               title="Paired participant bootstrap, 95% interval")
        ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(folder/f"R1-5_reference_composition.{ext}", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    colors = ["#276A88", "#BA563A", "#656565"]
    for ax, target in zip(axes, TARGETS):
        for j, method in enumerate(METHODS):
            p = heldout.loc[(heldout.target == target) & (heldout.method == method)].set_index("model").loc[list(MODELS)]
            ax.bar(np.arange(2)+(j-1)*.24, p.mean_site_rmse, .23, label=method, color=colors[j])
        ax.set(xticks=np.arange(2), xticklabels=["Blood/immune\n44 held out", "Epithelial\n83 held out"],
               ylabel="Mean per-locus RMSE", title=f"{target}: atlas tissue holdout")
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(folder/f"R1-6_celltype_holdout.{ext}", dpi=200)
    plt.close(fig)


def run(root):
    prep = json.loads((root/"preparation.json").read_text())
    mapping_validation = root/"pgpuk_mapping_validation"/"completion.json"
    mapping_check = json.loads(mapping_validation.read_text())
    if mapping_check["status"] != "PASS":
        raise ValueError("PGP-UK mapping has not passed validation")
    verify_file(mapping_check["preparation"])
    for info in mapping_check["outputs"].values():
        verify_file(info)
    for section in ("prepared_files", "inputs", "code"):
        for info in prep[section].values():
            verify_file(info)
    output = root/"summary"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    tasks = pd.read_csv(root/"tasks.tsv", sep="\t")
    samples = pd.read_csv(root/"atlas_samples.tsv", sep="\t")
    membership = pd.read_csv(root/"reference_membership.tsv", sep="\t")
    mapping = pd.read_csv(root/"epic_mapping.tsv", sep="\t")
    expected = {int(s): set(p.locus) for s, p in mapping.groupby("shard_index")}
    ext_parts = {(model, target): [] for model in MODELS for target in TARGETS}
    site_parts, checks, completions = [], [], []
    for task in tasks.itertuples(index=False):
        path = root/"fits"/task.model/f"shard_{task.shard_index:04d}"/"completion.json"
        info = json.loads(path.read_text())
        verify_file(info["preparation"])
        train, test = members(membership, samples, task.model)
        if (info["train_ids"] != train or info["test_ids"] != test or info["task_id"] != task.task_id
                or info["model"] != task.model or info["shard_index"] != task.shard_index):
            raise ValueError("Completion identity mismatch")
        for record in info["outputs"].values():
            verify_file(record)
        for target in TARGETS:
            obs = pd.read_parquet(info["outputs"][f"{target}_heldout_predictions"]["path"])
            if set(obs.locus) != expected[task.shard_index] or set(obs.sample_id) != set(test) or len(obs) != len(expected[task.shard_index])*len(test):
                raise ValueError("Incomplete held-out locus/sample product")
            independent = reconstruct_sites(obs).set_index(["locus", "method"]).sort_index()
            saved = pd.read_parquet(info["outputs"][f"{target}_heldout_site_metrics"]["path"]).set_index(["locus", "method"]).sort_index()
            if not independent.index.equals(saved.index):
                raise ValueError("Saved site keys differ")
            np.testing.assert_array_equal(independent.n, saved.n)
            np.testing.assert_array_equal(independent.eligible, saved.eligible)
            np.testing.assert_allclose(independent.rmse, saved.rmse, atol=1e-12, rtol=1e-12, equal_nan=True)
            checks.append({"task_id": task.task_id, "target": target, "n_loci": len(expected[task.shard_index]),
                           "max_abs_rmse_difference": float(np.nanmax(np.abs(independent.rmse-saved.rmse)))})
            saved = saved.reset_index().assign(model=task.model, target=target, n_train=len(train), n_heldout=len(test))
            site_parts.append(saved)
            ext_parts[task.model, target].append(pd.read_parquet(info["outputs"][f"{target}_external_predictions"]["path"]))
        completions.append(record_file(path))
        print(f"Verified task {task.task_id}", flush=True)
    output.mkdir()
    site = pd.concat(site_parts, ignore_index=True)
    site.to_parquet(output/"R1-6_site_metrics.parquet", index=False, compression="zstd")
    status = site.loc[site.method == "A2H"].groupby(["model", "target", "fit_status", "eligible"], dropna=False).size().rename("n_loci").reset_index()
    status.to_csv(output/"R1-6_fit_status_counts.tsv", sep="\t", index=False)
    heldout = site.loc[site.eligible].groupby(["model", "target", "method", "n_train", "n_heldout"]).agg(
        n_sites=("locus", "size"), n_observations=("n", "sum"), mean_site_rmse=("rmse", "mean"),
        median_site_rmse=("rmse", "median")).reset_index()
    heldout.to_csv(output/"R1-6_summary.tsv", sep="\t", index=False)
    pd.DataFrame(checks).to_csv(output/"independent_metric_checks.tsv", sep="\t", index=False)
    pairs = pd.read_csv(prep["inputs"]["pairs"]["path"], sep="\t")
    summaries, flows = [], []
    for target in TARGETS:
        frames = {model: pd.concat(ext_parts[model, target], ignore_index=True) for model in MODELS}
        frames["full_reference"] = pd.read_parquet(prep["inputs"][f"full_{target}"]["path"])
        for frame in frames.values():
            if set(frame.sample_id) != set(pairs.array_id):
                raise ValueError("External participants mismatch")
        common, audit, flow = align_external(frames, set(mapping.probe_id))
        common.to_parquet(output/f"R1-5_{target}_common_observations.parquet", index=False, compression="zstd")
        audit.to_csv(output/f"R1-5_{target}_eligibility.tsv", sep="\t", index=False)
        metrics = external_metrics(common)
        metrics.to_parquet(output/f"R1-5_{target}_site_metrics.parquet", index=False, compression="zstd")
        summary, boot_values, draws = bootstrap(common, pairs.array_id.tolist(), metrics)
        summary["target"] = target
        summaries.append(summary)
        flows.append(flow.assign(target=target))
        boot_values.to_csv(output/f"R1-5_{target}_bootstrap_values.tsv", sep="\t", index=False)
        draws.to_csv(output/f"R1-5_{target}_bootstrap_draws.tsv", sep="\t", index=False)
    external = pd.concat(summaries, ignore_index=True)
    external.to_csv(output/"R1-5_summary.tsv", sep="\t", index=False)
    pd.concat(flows, ignore_index=True).to_csv(output/"R1-5_count_flow.tsv", sep="\t", index=False)
    figures(external, heldout, output)
    write_json(output/"completion.json", {"finished_utc": utc(), "command": sys.argv,
        "preparation": record_file(root/"preparation.json"), "shard_completions": completions,
        "pgpuk_mapping_validation": record_file(mapping_validation),
        "summary_code": record_file(Path(__file__)), "outputs": {p.name: record_file(p) for p in output.iterdir()},
        "verification": "All shard output hashes; all heldout RMSEs independently recomputed using pandas; external means checked using rectangular arrays",
        "bootstrap": {"seed": 2026, "replicates": 1000, "unit": "whole participant", "samples": pairs.array_id.tolist(),
                      "empty_sites": "Omit sites with no resampled observation equally across models; record counts for every draw"},
        "interpretation": "Tissue-only transfer; shared donors permitted. No size-matched controls. Bootstrap describes ten participants, not population-wide certainty."})
    print(external.to_string(index=False), flush=True)
    print(heldout.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    run(parser.parse_args().root.resolve())
