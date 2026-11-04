"""Frozen tissue-only atlas fits and EPIC-only R1-5/R1-6 predictions.

Historical donor-excluded pipelines are not modified. All matching uses keys.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash, row_stats
from src.fit_atlas_fold import fit_sites
from src.make_transfer_manifest import class_masks
from src.run_atlas_cv_split import shard_arrays

MODELS = ("exclude_blood_immune", "exclude_epithelial")
TARGETS = ("CHALM", "CAMDA")
METHODS = ("A2H", "reference_linear", "m")
DEFAULT_ROOT = Path("results/tissue_only_r1_5_r1_6_20261008")
EPIC = Path("reference/EPIC.hg38.manifest.gencode.v36.tsv")
PROBE_MAP = Path("legacy_model/A2H_cpg_probe.hg38.tsv")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record_file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def verify_file(record):
    if file_hash(Path(record["path"])) != record["sha256"]:
        raise ValueError(f"File hash mismatch: {record['path']}")


def tissue_membership(samples):
    if samples.sample_id.isna().any() or samples.sample_id.duplicated().any() or samples.Group.isna().any():
        raise ValueError("Sample IDs and cell types must be complete and unique")
    parts, summaries = [], []
    for group, mask in class_masks(samples).items():
        model = f"exclude_{group}"
        part = samples[["sample_id", "donor_id", "Group"]].copy()
        part.insert(0, "model", model)
        part["role"] = np.where(mask, "heldout", "train")
        parts.append(part)
        summaries.append({"model": model, "n_train": int((~mask).sum()),
                          "n_heldout": int(mask.sum()), "n_excluded": 0,
                          "policy": "tissue_only; donors do not determine membership"})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summaries)


def members(manifest, samples, model):
    expected, _ = tissue_membership(samples)
    columns = ["sample_id", "donor_id", "Group", "role"]
    actual = manifest.loc[manifest.model == model, columns].reset_index(drop=True)
    correct = expected.loc[expected.model == model, columns].reset_index(drop=True)
    if not actual.equals(correct):
        raise ValueError(f"Membership is not the frozen tissue-only definition: {model}")
    train = actual.loc[actual.role == "train", "sample_id"].tolist()
    test = actual.loc[actual.role == "heldout", "sample_id"].tolist()
    if set(train) & set(test) or set(train + test) != set(samples.sample_id):
        raise ValueError("Train/test must partition samples")
    return train, test


def epic_mapping(manifest, mapping, allow_conflicts=False):
    epic = manifest.dropna(subset=["probeID", "CpG_chrm", "CpG_beg"])
    epic = epic.loc[epic.probeID.str.startswith("cg")].copy()
    if epic.probeID.duplicated().any():
        raise ValueError("Duplicate EPIC probe ID")
    main = mapping[["loci.hg38", "EPIC.hg38"]].dropna().drop_duplicates().rename(
        columns={"loci.hg38": "locus", "EPIC.hg38": "probe_id"})
    if main.probe_id.duplicated().any():
        raise ValueError("Ambiguous EPIC probe-to-locus mapping")
    main = main.merge(epic.rename(columns={"probeID": "probe_id"}), on="probe_id", validate="one_to_one")
    # Manifest CpG_beg is 0-based; the existing atlas locus key is 1-based.
    expected = main.CpG_chrm + "_" + (main.CpG_beg.astype(np.int64) + 1).astype(str)
    main["manifest_locus"] = expected
    main["manifest_coordinate_agrees"] = main.locus.eq(expected)
    if not allow_conflicts and not main.manifest_coordinate_agrees.all():
        raise ValueError("EPIC probe identity has conflicting hg38 coordinates")
    hm = mapping[["loci.hg38", "HM450.hg38"]].dropna().drop_duplicates().rename(
        columns={"loci.hg38": "locus", "HM450.hg38": "probe_id"})
    if hm.probe_id.duplicated().any():
        raise ValueError("Ambiguous HM450 probe mapping")
    main = main.merge(hm.rename(columns={"locus": "HM450_locus"}), on="probe_id", how="left", validate="one_to_one")
    main["platform_coordinates_agree"] = main.HM450_locus.isna() | main.HM450_locus.eq(main.locus)
    if not allow_conflicts and not main.platform_coordinates_agree.all():
        raise ValueError("HM450/EPIC mapping disagrees for a shared probe")
    main["eligible_mapping"] = main.manifest_coordinate_agrees & main.platform_coordinates_agree
    return main


def prepare(root):
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {root}")
    inputs = {"samples": Path("results/atlas_splits/atlas_samples.tsv"),
              "atlas_shards": Path("results/atlas_shards_original_full/shards.tsv"),
              "external_shards": Path("results/transfer_external_shards/external_shards.tsv"),
              "epic_manifest": EPIC, "probe_map": PROBE_MAP,
              "pairs": Path("results/pgpuk_depth10_calibrated/pairs.tsv")}
    for target in TARGETS:
        inputs[f"external_{target}"] = Path(f"results/pgpuk_depth10_calibrated/{target}_observations.parquet")
        inputs[f"full_{target}"] = Path(f"results/transfer_full/full_reference/summary/{target}_observations.parquet")
    records = {name: record_file(path) for name, path in inputs.items()}
    samples = pd.read_csv(inputs["samples"], sep="\t")
    if len(samples) != 205:
        raise ValueError("Expected 205 atlas samples")
    membership, counts = tissue_membership(samples)
    if counts.n_train.tolist() != [161, 122] or counts.n_heldout.tolist() != [44, 83]:
        raise ValueError("Unexpected class membership counts")
    mapping = epic_mapping(pd.read_csv(EPIC, sep="\t", usecols=["probeID", "CpG_chrm", "CpG_beg"]),
                           pd.read_csv(PROBE_MAP, sep="\t", usecols=["loci.hg38", "EPIC.hg38", "HM450.hg38"]), allow_conflicts=True)
    shards = pd.read_csv(inputs["atlas_shards"], sep="\t")
    if shards.shard_index.duplicated().any() or len(shards) != 52:
        raise ValueError("Expected 52 unique atlas shards")
    positions = []
    for row in shards.itertuples(index=False):
        loci = pd.read_parquet(row.path, columns=["locus"])
        if len(loci) != row.n_loci or loci.locus.iloc[0] != row.first_locus or loci.locus.iloc[-1] != row.last_locus:
            raise ValueError("Atlas shard index mismatch")
        loci["shard_index"] = row.shard_index
        positions.append(loci)
    positions = pd.concat(positions, ignore_index=True)
    if positions.locus.duplicated().any():
        raise ValueError("Duplicate atlas locus")
    joined = mapping.merge(positions, on="locus", how="left", validate="many_to_one")
    available = joined.loc[joined.shard_index.notna() & joined.eligible_mapping].copy()
    available["shard_index"] = available.shard_index.astype(int)
    root.mkdir(parents=True)
    (root / "log").mkdir()
    samples.to_csv(root / "atlas_samples.tsv", sep="\t", index=False)
    membership.to_csv(root / "reference_membership.tsv", sep="\t", index=False)
    counts.to_csv(root / "reference_models.tsv", sep="\t", index=False)
    joined.to_csv(root / "epic_mapping_audit.tsv", sep="\t", index=False)
    joined.loc[~joined.eligible_mapping].to_csv(root / "epic_mapping_conflicts.tsv", sep="\t", index=False)
    available.to_csv(root / "epic_mapping.tsv", sep="\t", index=False)
    shards.to_csv(root / "atlas_shards.tsv", sep="\t", index=False)
    tasks = [{"task_id": i * len(shards) + j, "model": model, "shard_index": int(shard)}
             for i, model in enumerate(MODELS) for j, shard in enumerate(shards.shard_index)]
    pd.DataFrame(tasks).to_csv(root / "tasks.tsv", sep="\t", index=False)
    code = {name: record_file(Path("src") / name) for name in
            ("tissue_only_transfer.py", "fit_atlas_fold.py", "evaluate_pgpuk.py", "make_transfer_manifest.py",
             "run_atlas_cv_split.py")}
    write_json(root / "preparation.json", {
        "created_utc": utc(), "command": sys.argv, "cwd": str(Path.cwd()),
        "inputs": records, "code": code,
        "prepared_files": {p.name: record_file(p) for p in root.glob("*.tsv")},
        "training_depth": 4, "external_depth": 10, "min_train_joint": 50,
        "min_evaluation_observations": 8, "new_support_filter": False,
        "membership_policy": "tissue_only", "size_controls": False,
        "coordinate_policy": "User approved excluding probes with manifest or cross-platform coordinate conflicts; no remapping",
        "coordinate_conflict_probes": int((~mapping.eligible_mapping).sum()),
        "epic_probes_mapped": len(mapping), "epic_probes_in_atlas": len(available),
        "unique_epic_loci_in_atlas": int(available.locus.nunique()),
        "coordinates": "manifest CpG_beg+1 checked against existing 1-based atlas locus; no conversion of source data",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}})
    print(counts.to_string(index=False), flush=True)
    print(f"Prepared {len(available)} probes / {available.locus.nunique()} unique EPIC atlas loci", flush=True)


def predict_coefficients(quality, x):
    return {"A2H": quality.a2h_intercept.to_numpy()[:, None]
            + quality.a2h_linear.to_numpy()[:, None] * x
            + quality.a2h_quadratic.to_numpy()[:, None] * x**2,
            "reference_linear": quality.reference_linear_intercept.to_numpy()[:, None]
            + quality.reference_linear_slope.to_numpy()[:, None] * x,
            "m": x}


def paired_site_metrics(loci, truth, predictions, min_subjects=8):
    valid = np.isfinite(truth)
    for method in METHODS:
        valid &= np.isfinite(predictions[method])
    n = valid.sum(axis=1)
    tables = []
    for method in METHODS:
        stats = row_stats(np.where(valid, truth, np.nan), np.where(valid, predictions[method], np.nan))
        tables.append(pd.DataFrame({"locus": loci, "method": method, "n": n,
                                    "rmse": stats["rmse"], "pearson_r": stats["pearson_r"],
                                    "eligible": n >= min_subjects}))
    return pd.concat(tables, ignore_index=True), valid


def run_shard(root, task_id, pilot=False):
    prep_path = root / "preparation.json"
    prep = json.loads(prep_path.read_text())
    for info in prep["prepared_files"].values():
        verify_file(info)
    for info in prep["code"].values():
        verify_file(info)
    tasks = pd.read_csv(root / "tasks.tsv", sep="\t").set_index("task_id")
    task = tasks.loc[task_id]
    model, shard_index = task.model, int(task.shard_index)
    output = root / "fits" / model / f"shard_{shard_index:04d}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    samples = pd.read_csv(root / "atlas_samples.tsv", sep="\t")
    train, test = members(pd.read_csv(root / "reference_membership.tsv", sep="\t"), samples, model)
    row = pd.read_csv(root / "atlas_shards.tsv", sep="\t").set_index("shard_index").loc[shard_index]
    verify_file({"path": row.path, "sha256": row.sha256})
    frame = pd.read_parquet(row.path)
    mapping = pd.read_csv(root / "epic_mapping.tsv", sep="\t")
    mapping = mapping.loc[mapping.shard_index == shard_index]
    frame = frame.loc[frame.locus.isin(mapping.locus)].reset_index(drop=True)
    if frame.locus.duplicated().any() or set(frame.locus) != set(mapping.locus):
        raise ValueError("EPIC shard identity mismatch")
    loci = pd.Index(frame.locus)
    external_shards = pd.read_csv(prep["inputs"]["external_shards"]["path"], sep="\t")
    verify_file(prep["inputs"]["external_shards"])
    verify_file(prep["inputs"]["pairs"])
    pairs = pd.read_csv(prep["inputs"]["pairs"]["path"], sep="\t")
    ext_samples = pairs.array_id.tolist()
    if (len(pairs) != 10 or len(set(ext_samples)) != 10 or pairs.wgbs_id.nunique() != 10
            or set(pairs.tissue) != {"blood"} or set(pairs.study) != {"pgpuk"}):
        raise ValueError("Expected ten PGP-UK blood pairs")
    output.mkdir(parents=True)
    counts, outputs, checks = [], {}, []
    for target in TARGETS:
        x, y, xt, truth = shard_arrays(frame, target, train, test)
        quality, predictions = fit_sites(x, y, xt, min_train=50, include_coefficients=True)
        quality.insert(0, "locus", loci)
        reproduced = predict_coefficients(quality, xt)
        for method in METHODS:
            np.testing.assert_allclose(reproduced[method], predictions[method], atol=0, rtol=0, equal_nan=True)
        site, valid = paired_site_metrics(loci, truth, predictions)
        site = site.merge(quality[["locus", "fit_status", "n_train_joint"]], on="locus", validate="many_to_one")
        holdout = pd.DataFrame({"locus": np.repeat(loci, len(test)), "sample_id": np.tile(test, len(loci)),
                                "m": xt.ravel(), "observed": truth.ravel(),
                                "A2H": predictions["A2H"].ravel(),
                                "reference_linear": predictions["reference_linear"].ravel(),
                                "evaluation_valid": valid.ravel()})
        tables = {"coefficients": quality, "heldout_predictions": holdout, "heldout_site_metrics": site}
        extrow = external_shards.loc[(external_shards.target == target) & (external_shards.shard_index == shard_index)]
        if len(extrow) != 1:
            raise ValueError("Expected one external shard per target")
        extrow = extrow.iloc[0]
        verify_file({"path": extrow.path, "sha256": extrow.sha256})
        obs = pd.read_parquet(extrow.path)
        obs = obs.loc[obs.probe_id.isin(mapping.probe_id)].copy()
        expected_locus = mapping.set_index("probe_id").locus
        if not obs.probe_id.map(expected_locus).eq(obs.locus).all() or obs.duplicated(["probe_id", "sample_id"]).any():
            raise ValueError("External probe/locus/sample mismatch")
        q = quality.set_index("locus").loc[obs.locus]
        beta = obs.m.to_numpy()[:, None]
        extpred = predict_coefficients(q, beta)
        external = obs[["probe_id", "sample_id", "locus", "m", "observed"]].copy()
        for method in ("A2H", "reference_linear"):
            external[method] = extpred[method][:, 0]
        external["fit_status"] = q.fit_status.to_numpy()
        tables["external_predictions"] = external
        if pilot:
            # Independent scalar fits on predetermined representative rows, not performance-selected.
            indices = np.linspace(0, len(loci)-1, min(100, len(loci)), dtype=int)
            max_difference = 0.0
            for i in indices:
                paired = np.isfinite(x[i]) & np.isfinite(y[i])
                if quality.fit_status.iloc[i] == "quadratic":
                    coef = np.polyfit(x[i, paired], y[i, paired], 2)
                    expected = np.polyval(coef, xt[i])
                    np.testing.assert_allclose(predictions["A2H"][i], expected, atol=2e-7, rtol=2e-7, equal_nan=True)
                    if np.isfinite(expected).any():
                        max_difference = max(max_difference, float(np.nanmax(np.abs(predictions["A2H"][i]-expected))))
            # Reproduce the cached full-reference A2H and linear predictions on this shard.
            all_ids = samples.sample_id.tolist()
            fx, fy, _, _ = shard_arrays(frame, target, all_ids, [])
            fq, _ = fit_sites(fx, fy, np.zeros((len(loci), 1)), include_coefficients=True)
            fq.insert(0, "locus", loci)
            fp = predict_coefficients(fq.set_index("locus").loc[obs.locus], beta)
            verify_file(prep["inputs"][f"full_{target}"])
            full = pd.read_parquet(prep["inputs"][f"full_{target}"]["path"])
            match = obs[["probe_id", "sample_id", "m", "observed"]].merge(full, on=["probe_id", "sample_id"],
                suffixes=("_source", "_cached"), validate="one_to_one", how="left")
            for col in ("m", "observed"):
                np.testing.assert_allclose(match[f"{col}_source"], match[f"{col}_cached"], atol=0, rtol=0)
            for method in ("A2H", "reference_linear"):
                np.testing.assert_allclose(fp[method][:, 0], match[method], atol=2e-10, rtol=2e-10, equal_nan=True)
            checks.append({"target": target, "representative_rows": len(indices),
                           "max_scalar_polyfit_prediction_difference": max_difference,
                           "full_reference_cache_reproduced": True})
        for name, table in tables.items():
            path = output / f"{target}_{name}.parquet"
            table.to_parquet(path, index=False, compression="zstd")
            outputs[f"{target}_{name}"] = record_file(path)
        counts.append({"target": target, "n_loci": len(loci), "n_train": len(train), "n_heldout": len(test),
                       "n_eligible_heldout": int(site.loc[site.method == "A2H", "eligible"].sum()),
                       "n_external_input_probes": int(external.probe_id.nunique()),
                       "fit_status": quality.fit_status.value_counts().to_dict()})
        del holdout, tables, predictions, reproduced, x, y, xt, truth
    write_json(output / "completion.json", {"finished_utc": utc(), "model": model, "shard_index": shard_index,
        "task_id": task_id, "pilot": pilot, "counts": counts, "outputs": outputs, "checks": checks,
        "preparation": record_file(prep_path), "source_shard": {"path": row.path, "sha256": row.sha256},
        "train_ids": train, "test_ids": test, "job_id": os.environ.get("SLURM_JOB_ID"),
        "command": sys.argv, "code": prep["code"]})
    print(json.dumps({"model": model, "shard": shard_index, "pilot": pilot, "counts": counts}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve())
    else:
        if args.task_id is None:
            parser.error("run requires --task-id")
        run_shard(args.root.resolve(), args.task_id, args.pilot)


if __name__ == "__main__":
    main()
