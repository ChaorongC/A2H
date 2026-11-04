"""Package existing depth-4 EPIC whole-atlas CV summaries, without refitting."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

SUBJECT = Path("results/r1_1_locus_specific_calibration")
LEGACY = Path("atlas_cv_original/result")
OUTPUT = Path("reviewer_response_packets/R1-6")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def close(actual, expected):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Summary mismatch: {actual} != {expected}")


def run():
    table_path, provenance_path = OUTPUT/"whole_atlas_cv_summary.tsv", OUTPUT/"whole_atlas_cv_provenance.json"
    if table_path.exists() or provenance_path.exists():
        raise FileExistsError("Refusing to overwrite an existing CV supplement")
    metadata = json.loads((SUBJECT/"provenance.json").read_text())
    inputs = []
    for name in ("provenance.json", "cv_metrics_across_repeats.tsv", "cv_metrics_by_repeat.tsv", "mask_and_metric_definitions.tsv"):
        p = SUBJECT/name
        sha = digest(p)
        if name != "provenance.json" and sha != metadata["output_sha256"][name]:
            raise ValueError(f"Historical subject-CV hash mismatch: {name}")
        inputs.append({"path": str(p), "sha256": sha})
    splits_path = Path(metadata["inputs"]["atlas_splits"])
    if digest(splits_path) != metadata["source_sha256"]["atlas_splits"]:
        raise ValueError("Historical CV split manifest changed")
    inputs.append({"path": str(splits_path), "sha256": digest(splits_path)})
    splits = [r for r in rows(splits_path) if r["branch"] == "historical_sample"]
    ids = sorted({int(r["split_index"]) for r in splits})
    if ids != list(range(100)) or len({r["sample_id"] for r in splits}) != 205:
        raise ValueError("Expected 100 splits of the 205-sample whole atlas")
    train_counts, test_counts = [], []
    for index in ids:
        part = [r for r in splits if int(r["split_index"]) == index]
        train = {r["sample_id"] for r in part if r["role"] == "train"}
        test = {r["sample_id"] for r in part if r["role"] == "test"}
        if train & test or len(part) != 205 or len(train|test) != 205:
            raise ValueError("Invalid whole-atlas train/test partition")
        train_counts.append(len(train)); test_counts.append(len(test))
    if set(train_counts) != {184, 185} or set(test_counts) != {20, 21}:
        raise ValueError("Unexpected training/heldout sample counts")
    aggregate, repeat = rows(SUBJECT/"cv_metrics_across_repeats.tsv"), rows(SUBJECT/"cv_metrics_by_repeat.tsv")
    summary = []
    for target in ("CHALM", "CAMDA"):
        matches = [r for r in aggregate if r["target"] == target and r["universe"] == "EPIC_BeadChip" and r["method"] == "A2H"]
        detail = [r for r in repeat if r["target"] == target and r["universe"] == "EPIC_BeadChip" and r["method"] == "A2H"]
        if len(matches) != 1 or sorted(int(r["repeat"]) for r in detail) != list(range(10)):
            raise ValueError("Missing/duplicate target or repeat in subject-level EPIC CV")
        a = matches[0]
        errors = [float(r["mean_rmse"]) for r in detail]
        correlations = [float(r["mean_pearson_r"]) for r in detail]
        close(a["mean_of_repeat_mean_rmse"], statistics.mean(errors))
        close(a["sd_of_repeat_mean_rmse"], statistics.stdev(errors))
        close(a["mean_of_repeat_mean_pearson_r"], statistics.mean(correlations))
        close(a["sd_of_repeat_mean_pearson_r"], statistics.stdev(correlations))
        summary.append({"endpoint": "subject_level_out_of_fold", "target": target,
            "training_depth": ">=4", "n_atlas_samples": 205, "n_splits": 100, "n_repeats": 10,
            "train_samples_per_split": "184-185", "heldout_samples_per_split": "20-21",
            "rmse_mean": a["mean_of_repeat_mean_rmse"], "rmse_sd": a["sd_of_repeat_mean_rmse"],
            "pearson_mean": a["mean_of_repeat_mean_pearson_r"], "pearson_sd": a["sd_of_repeat_mean_pearson_r"],
            "n_EPIC_candidate_units": 824908,
            "n_eligible_units_min": min(int(r["n_sites_n_ge_2"]) for r in detail),
            "n_eligible_units_max": max(int(r["n_sites_n_ge_2"]) for r in detail),
            "metric_definition": "Mean across ten repeats of equal-locus mean RMSE over individual out-of-fold samples; Pearson is mean within-locus correlation; SD is across repeats (ddof=1)",
            "feature_definition": "Historical deployed EPIC.hg38 locus intersection, not the new coordinate-conflict-filtered holdout mask; at least two valid observations per locus",
            "source": str(SUBJECT/"cv_metrics_across_repeats.tsv")})
    for name in ("20261003_historical_cv_epic_cpg_summary.tsv", "20261003_historical_cv_fold_metrics_epic_cpg.tsv", "20261003_historical_cv_site_metrics_epic_cpg_provenance.json"):
        p = LEGACY/name
        inputs.append({"path": str(p), "sha256": digest(p)})
    historical = rows(LEGACY/"20261003_historical_cv_epic_cpg_summary.tsv")
    folds = rows(LEGACY/"20261003_historical_cv_fold_metrics_epic_cpg.tsv")
    historical_provenance = json.loads((LEGACY/"20261003_historical_cv_site_metrics_epic_cpg_provenance.json").read_text())
    for target in ("CHALM", "CAMDA"):
        matching = [r for r in historical if r["method"] == target]
        detail = [r for r in folds if r["method"] == target]
        if len(matching) != 1 or sorted(int(r["fold_id"]) for r in detail) != list(range(100)):
            raise ValueError("Missing/duplicate historical fold summaries")
        a = matching[0]
        for metric, source_column in (("rmse", "rmse"), ("pearson", "pearson_r")):
            values = [float(r[source_column]) for r in detail]
            close(a[f"fold_{metric}_mean"], statistics.mean(values))
            close(a[f"fold_{metric}_sd"], statistics.stdev(values))
            close(a[f"fold_{metric}_mean"], historical_provenance["methods"][target]["summary"][f"fold_{metric}_mean"])
        summary.append({"endpoint": "historical_heldout_fold_mean", "target": target,
            "training_depth": ">=4", "n_atlas_samples": 205, "n_splits": 100, "n_repeats": 10,
            "train_samples_per_split": "184-185", "heldout_samples_per_split": "20-21",
            "rmse_mean": a["fold_rmse_mean"], "rmse_sd": a["fold_rmse_sd"],
            "pearson_mean": a["fold_pearson_mean"], "pearson_sd": a["fold_pearson_sd"],
            "n_EPIC_candidate_units": a["n_epic_valid_rows"],
            "n_eligible_units_min": min(int(r["n_epic_rows"]) for r in detail),
            "n_eligible_units_max": max(int(r["n_epic_rows"]) for r in detail),
            "metric_definition": "Mean across 100 folds of RMSE/cross-CpG Pearson of heldout-sample mean truth vs mean prediction; SD across folds (ddof=1); not individual-sample prediction error",
            "feature_definition": "Historical EPIC manifest rows, including duplicate probe IDs; 824939 valid rows / 824908 unique probe IDs; not the new holdout mask",
            "source": str(LEGACY/"20261003_historical_cv_fold_metrics_epic_cpg.tsv")})
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(summary)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "action": "Append existing whole-atlas CV summaries only; no new fits or CV jobs",
        "code": {"path": str(Path(__file__)), "sha256": digest(Path(__file__))},
        "inputs": inputs, "outputs": [{"path": str(table_path), "sha256": digest(table_path)}],
        "checks": "Subject-summary hashes; independent means/SDs from ten repeats and 100 folds; historical provenance values; 100 sample-disjoint 205-sample split partitions",
        "interpretation": "The full 205-sample atlas is the CV source, not each fold's training set. Same donors can span train/test. Repeats are not independent cohorts. Different metric and EPIC eligibility definitions prevent a directly matched contrast with tissue holdouts.",
        "source_validation_limit": "Existing CV output summaries were checked; large historical pickles and all raw subject prediction shards were not reread in this documentation-only addition."}
    provenance_path.write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()
