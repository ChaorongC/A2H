"""R2-2 E1: nested atlas-size refits and fixed PGP-UK evaluation.

Run from reviewer_revision_a2h with ``python -m src.r2_2_reference_size``.
The original A2H models, historical CV and external source tables are read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.r2_2_common import fit_compatible, nested_group_draws, pooled_metrics, site_metrics


HERE = Path(__file__).resolve().parents[1]
RESULTS = HERE / "results"
DEFAULT_ROOT = RESULTS / "r2_2_sensitivity_20261011"
SAMPLES = RESULTS / "atlas_splits/atlas_samples.tsv"
SHARDS = RESULTS / "atlas_shards_original_full/shards.tsv"
MAPPING = RESULTS / "tissue_only_r1_5_r1_6_20261008/epic_mapping.tsv"
EXTERNAL = RESULTS / "pgpuk_depth10_calibrated"
TARGETS = ("CHALM", "CAMDA")
SIZES = (75, 100, 150)
SEEDS = tuple(range(42, 62))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def hash_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_hash(path)}


def prepare(root: Path) -> None:
    e1 = root / "E1"
    if e1.exists():
        raise FileExistsError(f"Refusing to overwrite {e1}")
    samples = pd.read_csv(SAMPLES, sep="\t")
    shards = pd.read_csv(SHARDS, sep="\t")
    mapping = pd.read_csv(MAPPING, sep="\t", usecols=["probe_id", "locus", "shard_index", "eligible_mapping"])
    if mapping.probe_id.duplicated().any() or shards.shard_index.duplicated().any():
        raise ValueError("Duplicate probe or shard ID in frozen inputs")
    mapping = mapping.loc[mapping.eligible_mapping].copy()
    ambiguous = mapping.locus.duplicated(keep=False)
    excluded = mapping.loc[ambiguous].sort_values(["locus", "probe_id"])
    mapping = mapping.loc[~ambiguous, ["probe_id", "locus", "shard_index"]].copy()
    if mapping.locus.duplicated().any() or mapping.probe_id.duplicated().any():
        raise ValueError("New EPIC evaluation map is not one-to-one")
    if not set(mapping.shard_index).issubset(shards.shard_index):
        raise ValueError("EPIC mapping references absent atlas shards")
    pairs = pd.read_csv(EXTERNAL / "pairs.tsv", sep="\t")
    pairs = pairs.loc[(pairs.study == "pgpuk") & (pairs.tissue == "blood")]
    sample_ids = pairs.array_id.tolist()
    if len(sample_ids) != 10 or len(set(sample_ids)) != 10 or pairs.wgbs_id.nunique() != 10:
        raise ValueError("Expected ten unique PGP-UK blood pairs")
    draws = nested_group_draws(samples, SEEDS)
    e1.mkdir(parents=True)
    excluded.to_csv(e1 / "excluded_duplicate_loci.tsv", sep="\t", index=False)
    mapping.to_csv(e1 / "epic_mapping.tsv", sep="\t", index=False)
    draws.to_csv(e1 / "draws.tsv", sep="\t", index=False)
    counts = []
    for (replicate, seed), part in draws.groupby(["replicate", "seed"], sort=True):
        for size in (*SIZES, 205):
            selection = part.loc[part["rank"] <= size]
            for group, count in selection.Group.value_counts().items():
                counts.append((replicate, seed, size, group, count, selection.donor_id.nunique()))
    group_counts = pd.DataFrame(counts, columns=["replicate", "seed", "n_reference", "Group",
                                                 "n_group", "n_unique_donors"])
    group_counts.to_csv(e1 / "sample_group_counts.tsv", sep="\t", index=False)
    source = {"samples": hash_record(SAMPLES), "shards": hash_record(SHARDS),
              "mapping": hash_record(MAPPING), "pairs": hash_record(EXTERNAL / "pairs.tsv")}
    flow = {"reference_samples": len(samples), "reference_groups": int(samples.Group.nunique()),
            "reference_donors": int(samples.donor_id.nunique()),
            "epic_probes_one_to_one": len(mapping),
            "duplicate_locus_probe_rows_excluded": len(excluded),
            "duplicate_loci_excluded": int(excluded.locus.nunique()),
            "external_samples": sample_ids, "targets": {}}
    for target in TARGETS:
        path = EXTERNAL / f"{target}_observations.parquet"
        source[f"external_{target}"] = hash_record(path)
        obs = pd.read_parquet(path, columns=["probe_id", "sample_id", "m", "observed", "A2H"])
        if obs.duplicated(["probe_id", "sample_id"]).any() or not set(obs.sample_id).issubset(sample_ids):
            raise ValueError(f"Duplicate or unexpected external keys for {target}")
        obs = obs.merge(mapping, on="probe_id", how="inner", validate="many_to_one")
        if not np.isfinite(obs[["m", "observed", "A2H"]].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite frozen {target} observation/baseline")
        if ((obs.m < 0) | (obs.m > 1) | (obs.observed < 0) | (obs.observed > 1)).any():
            raise ValueError(f"Out-of-domain {target} input/truth")
        wide = obs.pivot(index="probe_id", columns="sample_id", values=["m", "observed", "A2H"])
        ordered = pd.MultiIndex.from_product([["m", "observed", "A2H"], sample_ids])
        wide = wide.reindex(columns=ordered)
        keys = obs[["probe_id", "locus", "shard_index"]].drop_duplicates().set_index("probe_id")
        if keys.index.duplicated().any() or len(keys) != len(wide):
            raise ValueError("External probe/locus keys are not one-to-one")
        out = keys.loc[wide.index].reset_index()
        for field, prefix in (("m", "m"), ("observed", "observed"), ("A2H", "baseline")):
            values = wide[field]
            for sample_id in sample_ids:
                out[f"{prefix}__{sample_id}"] = values[sample_id].to_numpy(dtype=float)
        out = out.sort_values(["shard_index", "locus", "probe_id"]).reset_index(drop=True)
        out_path = e1 / f"external_{target}.parquet"
        out.to_parquet(out_path, index=False, compression="zstd")
        flow["targets"][target] = {"n_pairs": len(obs), "n_probes": len(out),
                                    "n_samples": obs.sample_id.nunique(), "sha256": file_hash(out_path)}
    write_json(e1 / "preparation.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "input": source, "flow": flow, "sample_seed_series": list(SEEDS),
               "code": hash_record(Path(__file__)), "common_code": hash_record(HERE / "src/r2_2_common.py"),
               "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
               "external_eligibility": "Saved depth10 and confidence-filtered observations; no new support filter",
               "fit_rule": ">=50 jointly finite reference pairs; quadratic/constant KDE; scalar polyfit for ill-conditioned sites"})


def verify_preparation(e1: Path) -> dict:
    prep = json.loads((e1 / "preparation.json").read_text())
    for entry in prep["input"].values():
        if file_hash(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Changed frozen input: {entry['path']}")
    for target in TARGETS:
        if file_hash(e1 / f"external_{target}.parquet") != prep["flow"]["targets"][target]["sha256"]:
            raise ValueError(f"Changed frozen external matrix for {target}")
    return prep


def run(root: Path, replicate: int, size: int, pilot_shard: int | None = None) -> None:
    if replicate not in range(20) or size not in SIZES:
        raise ValueError("E1 has 20 replicates at N=75/100/150 only")
    e1 = root / "E1"
    prep = verify_preparation(e1)
    draws = pd.read_csv(e1 / "draws.tsv", sep="\t")
    selected = draws.loc[(draws.replicate == replicate) & (draws["rank"] <= size)]
    train = selected.sample_id.tolist()
    if len(train) != size or len(set(train)) != size or selected.Group.nunique() != 39:
        raise ValueError("Invalid frozen nested training sample subset")
    shards = pd.read_csv(SHARDS, sep="\t")
    if pilot_shard is not None:
        shards = shards.loc[shards.shard_index == pilot_shard]
    if shards.empty:
        raise ValueError("No atlas shards selected")
    mapping = pd.read_csv(e1 / "epic_mapping.tsv", sep="\t")
    out = e1 / ("pilot" if pilot_shard is not None else "runs") / f"N{size}_rep{replicate:02d}"
    if pilot_shard is not None:
        out = out / f"shard{pilot_shard:04d}"
    out.mkdir(parents=True, exist_ok=False)
    output = {}
    for target in TARGETS:
        external = pd.read_parquet(e1 / f"external_{target}.parquet")
        ext_m_cols = [f"m__{sample}" for sample in prep["flow"]["external_samples"]]
        ext_y_cols = [f"observed__{sample}" for sample in prep["flow"]["external_samples"]]
        ext_b_cols = [f"baseline__{sample}" for sample in prep["flow"]["external_samples"]]
        quality_parts, prediction_parts, metric_parts = [], [], []
        for shard in shards.itertuples(index=False):
            local_map = mapping.loc[mapping.shard_index == shard.shard_index]
            if local_map.empty:
                continue
            columns = ["locus"] + [f"m__{sample}" for sample in train] + [f"{target}__{sample}" for sample in train]
            frame = pd.read_parquet(shard.path, columns=columns)
            if len(frame) != shard.n_loci or frame.locus.iloc[0] != shard.first_locus or frame.locus.iloc[-1] != shard.last_locus:
                raise ValueError(f"Atlas shard changed row identity: {shard.path}")
            frame = frame.loc[frame.locus.isin(local_map.locus)].reset_index(drop=True)
            if frame.locus.duplicated().any() or len(frame) != len(local_map):
                raise ValueError("EPIC map and atlas shard differ")
            x = frame[[f"m__{sample}" for sample in train]].to_numpy(dtype=float)
            y = frame[[f"{target}__{sample}" for sample in train]].to_numpy(dtype=float)
            quality = fit_compatible(x, y, min_train=50)
            quality.insert(0, "locus", frame.locus.to_numpy())
            quality.insert(1, "shard_index", shard.shard_index)
            quality_parts.append(quality[["locus", "shard_index", "n_train_joint", "train_x_variance",
                                         "fit_status", "a2h_intercept", "a2h_linear", "a2h_quadratic"]])
            ext = external.loc[external.shard_index == shard.shard_index].copy()
            if ext.empty:
                continue
            if ext.probe_id.duplicated().any() or not set(ext.locus).issubset(set(frame.locus)):
                raise ValueError("External probe/site identity changed")
            coef = quality.set_index("locus").loc[ext.locus]
            m = ext[ext_m_cols].to_numpy(dtype=float)
            pred = (coef.a2h_intercept.to_numpy()[:, None]
                    + coef.a2h_linear.to_numpy()[:, None] * m
                    + coef.a2h_quadratic.to_numpy()[:, None] * m**2)
            predicted = ext[["probe_id", "locus", "shard_index"]].copy()
            for i, sample_id in enumerate(prep["flow"]["external_samples"]):
                predicted[f"pred__{sample_id}"] = pred[:, i]
            prediction_parts.append(predicted)
            obs = ext[ext_y_cols].to_numpy(dtype=float)
            base = ext[ext_b_cols].to_numpy(dtype=float)
            current = site_metrics(obs, pred)
            common = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(base)
            paired_current = site_metrics(np.where(common, obs, np.nan), np.where(common, pred, np.nan))
            paired_baseline = site_metrics(np.where(common, obs, np.nan), np.where(common, base, np.nan))
            metric = ext[["probe_id", "locus", "shard_index"]].copy()
            for prefix, stats in (("all", current), ("common", paired_current), ("baseline_common", paired_baseline)):
                for column in ("n", "rmse", "mae", "bias_pred_minus_obs", "observed_variance",
                               "pearson_r", "correlation_reason", "sse", "sae", "sxy", "syy", "spp"):
                    metric[f"{prefix}_{column}"] = stats[column].to_numpy()
            metric_parts.append(metric)
        quality_all = pd.concat(quality_parts, ignore_index=True)
        predicted_all = pd.concat(prediction_parts, ignore_index=True)
        metric_all = pd.concat(metric_parts, ignore_index=True)
        if quality_all.locus.duplicated().any() or predicted_all.probe_id.duplicated().any() or metric_all.probe_id.duplicated().any():
            raise ValueError("Duplicate E1 fit or external key")
        files = {}
        for name, table in (("quality", quality_all), ("predictions", predicted_all), ("site_metrics", metric_all)):
            path = out / f"{target}_{name}.parquet"
            table.to_parquet(path, index=False, compression="zstd")
            files[name] = hash_record(path) | {"n_rows": len(table)}
        cases = {}
        for prefix in ("all", "common", "baseline_common"):
            renamed = metric_all.rename(columns={f"{prefix}_{col}": col for col in
                                                 ("n", "rmse", "pearson_r", "sse", "sxy", "syy", "spp")})
            cases[prefix] = pooled_metrics(renamed)
        output[target] = {"files": files, "n_fitted_loci": int(quality_all.fit_status.isin(
            ["quadratic", "constant_kde", "quadratic_rank_warning"]).sum()),
            "n_candidate_loci": len(quality_all), "fit_status_counts": quality_all.fit_status.value_counts().to_dict(),
            "cases": cases}
    write_json(out / "completion.json", {"finished_utc": datetime.now(timezone.utc).isoformat(),
        "run_status": "complete", "replicate": replicate, "seed": int(selected.seed.iloc[0]),
        "n_reference": size, "pilot_shard": pilot_shard, "training_ids": train,
        "n_training_donors": int(selected.donor_id.nunique()),
        "preparation_sha256": file_hash(e1 / "preparation.json"),
        "code_sha256": file_hash(Path(__file__)), "common_code_sha256": file_hash(HERE / "src/r2_2_common.py"),
        "command": sys.argv, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "targets": output})
    print(json.dumps({"n": size, "replicate": replicate, "pilot_shard": pilot_shard,
                      "targets": {target: {"n_fitted_loci": entry["n_fitted_loci"],
                                           "cases": entry["cases"]} for target, entry in output.items()}}))


def baseline_check(root: Path, shard_index: int = 0) -> None:
    """Compare the full 205-sample scalar fit with saved production predictions."""
    e1 = root / "E1"
    prep = verify_preparation(e1)
    samples = pd.read_csv(SAMPLES, sep="\t").sample_id.tolist()
    shards = pd.read_csv(SHARDS, sep="\t").set_index("shard_index")
    shard = shards.loc[shard_index]
    checks = []
    for target in TARGETS:
        ext = pd.read_parquet(e1 / f"external_{target}.parquet")
        ext = ext.loc[ext.shard_index == shard_index]
        sample_ids = prep["flow"]["external_samples"]
        if len(ext) < 100:
            raise ValueError("Too few external sites to check original baseline")
        indices = np.linspace(0, len(ext) - 1, 100, dtype=int)
        chosen = ext.iloc[indices]
        columns = ["locus"] + [f"m__{sample}" for sample in samples] + [f"{target}__{sample}" for sample in samples]
        frame = pd.read_parquet(shard.path, columns=columns).set_index("locus").loc[chosen.locus]
        x = frame[[f"m__{sample}" for sample in samples]].to_numpy(dtype=float)
        y = frame[[f"{target}__{sample}" for sample in samples]].to_numpy(dtype=float)
        quality = fit_compatible(x, y)
        m = chosen[[f"m__{sample}" for sample in sample_ids]].to_numpy(dtype=float)
        baseline = chosen[[f"baseline__{sample}" for sample in sample_ids]].to_numpy(dtype=float)
        calc = (quality.a2h_intercept.to_numpy()[:, None] + quality.a2h_linear.to_numpy()[:, None] * m
                + quality.a2h_quadratic.to_numpy()[:, None] * m**2)
        mask = np.isfinite(baseline) & np.isfinite(calc)
        if mask.sum() < 50:
            raise ValueError("Insufficient comparable saved A2H baseline predictions")
        difference = np.abs(calc[mask] - baseline[mask])
        checks.append({"target": target, "n_sites": len(chosen), "n_pairs": int(mask.sum()),
                       "max_absolute_difference": float(difference.max()),
                       "p99_absolute_difference": float(np.quantile(difference, 0.99)),
                       "n_over_2e_minus_7": int((difference > 2e-7).sum()),
                       "n_saved_without_refit": int(np.isfinite(baseline).sum() - mask.sum()),
                       "fit_status_counts": quality.fit_status.value_counts().to_dict()})
    out = e1 / f"baseline_check_shard{shard_index:04d}.json"
    if out.exists():
        raise FileExistsError(out)
    write_json(out, {"condition": "N205 full atlas", "comparisons": checks,
                     "tolerance_from_existing_R1_5_pilot": 2e-7,
                     "source": hash_record(EXTERNAL / "CHALM_observations.parquet")})
    print(json.dumps(checks))
    if any(row["n_over_2e_minus_7"] for row in checks):
        raise ValueError("Full-reference numerical agreement exceeded established pilot tolerance")


def verify_pilots(root: Path, shard_index: int = 0) -> None:
    e1 = root / "E1"
    prep = verify_preparation(e1)
    draws = pd.read_csv(e1 / "draws.tsv", sep="\t")
    sample_order = pd.read_csv(SAMPLES, sep="\t").sample_id.tolist()
    shard = pd.read_csv(SHARDS, sep="\t").set_index("shard_index").loc[shard_index]
    audits = []
    for replicate in (0, 1):
        for size in SIZES:
            result_dir = e1 / "pilot" / f"N{size}_rep{replicate:02d}" / f"shard{shard_index:04d}"
            completed = json.loads((result_dir / "completion.json").read_text())
            expected = draws.loc[(draws.replicate == replicate) & (draws["rank"] <= size)].sample_id.tolist()
            if completed["training_ids"] != expected or len(set(expected)) != size:
                raise ValueError("Pilot sample membership differs from frozen draw")
            if not set(expected).issubset(sample_order):
                raise ValueError("Pilot uses unknown sample")
            for target in TARGETS:
                entry = completed["targets"][target]
                for record in entry["files"].values():
                    path = Path(record["path"])
                    if not path.exists() or file_hash(path) != record["sha256"]:
                        raise ValueError(f"Pilot output is missing or changed: {path}")
                quality = pd.read_parquet(result_dir / f"{target}_quality.parquet")
                pred = pd.read_parquet(result_dir / f"{target}_predictions.parquet")
                ext = pd.read_parquet(e1 / f"external_{target}.parquet")
                ext = ext.loc[ext.shard_index == shard_index].set_index("locus")
                if quality.locus.duplicated().any() or pred.probe_id.duplicated().any():
                    raise ValueError("Duplicate pilot locus or probe")
                indices = set(np.linspace(0, len(quality) - 1, min(100, len(quality)), dtype=int))
                for status in ("constant_kde", "quadratic_rank_warning", "insufficient"):
                    indices.update(quality.index[quality.fit_status == status][:5].tolist())
                chosen = quality.iloc[sorted(indices)]
                frame = pd.read_parquet(shard.path, columns=["locus"] + [f"m__{s}" for s in expected]
                                        + [f"{target}__{s}" for s in expected]).set_index("locus").loc[chosen.locus]
                x = frame[[f"m__{s}" for s in expected]].to_numpy(dtype=float)
                y = frame[[f"{target}__{s}" for s in expected]].to_numpy(dtype=float)
                max_diff = 0.0
                checked = 0
                for index, item in enumerate(chosen.itertuples(index=False)):
                    if item.fit_status == "insufficient":
                        continue
                    valid = np.isfinite(x[index]) & np.isfinite(y[index])
                    if valid.sum() < 50:
                        raise ValueError("A pilot fitted a site with <50 paired observations")
                    if item.fit_status == "constant_kde":
                        from sklearn.neighbors import KernelDensity
                        grid = np.linspace(0, 1, 10)
                        kde = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(y[index, valid, None])
                        coefficients = np.array([0.0, 0.0, grid[kde.score_samples(grid[:, None]).argmax()]])
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            coefficients = np.polyfit(x[index, valid], y[index, valid], 2)
                    observed_coefficients = np.array([item.a2h_quadratic, item.a2h_linear, item.a2h_intercept])
                    fixed_m = np.array([0.1, 0.5, 0.9])
                    expected_pred = np.polyval(coefficients, fixed_m)
                    actual_pred = np.polyval(observed_coefficients, fixed_m)
                    finite = np.isfinite(expected_pred) & np.isfinite(actual_pred)
                    if finite.any():
                        max_diff = max(max_diff, float(np.max(np.abs(expected_pred[finite] - actual_pred[finite]))))
                        checked += 1
                if max_diff > 2e-7:
                    raise ValueError(f"Pilot {target} scalar polyfit mismatch: {max_diff}")
                if not pred.locus.isin(ext.index).all():
                    raise ValueError("Pilot prediction locus not in frozen external inputs")
                audits.append({"replicate": replicate, "n_reference": size, "target": target,
                               "n_sites_checked": checked, "max_scalar_prediction_difference": max_diff,
                               "n_fitted_loci": entry["n_fitted_loci"],
                               "n_candidate_loci": entry["n_candidate_loci"],
                               "n_external_pairs": entry["cases"]["all"]["n_valid_pairs"]})
    path = e1 / f"pilot_validation_edge_shard{shard_index:04d}.tsv"
    if path.exists():
        raise FileExistsError(path)
    pd.DataFrame(audits).to_csv(path, sep="\t", index=False)
    print(pd.DataFrame(audits).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "baseline-check", "verify-pilots"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--size", type=int)
    parser.add_argument("--pilot-shard", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "baseline-check":
        baseline_check(args.root, args.shard_index)
    elif args.action == "verify-pilots":
        verify_pilots(args.root, args.shard_index)
    else:
        if args.replicate is None or args.size is None:
            parser.error("run needs --replicate and --size")
        run(args.root, args.replicate, args.size, args.pilot_shard)


if __name__ == "__main__":
    main()
