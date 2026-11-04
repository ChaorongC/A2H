"""Combine verified CV and holdout artifacts without modifying original results."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("results/tissue_only_r1_5_r1_6_20261008")
PACKET = Path("reviewer_response_packets/R1-6")
FIELDS = ("endpoint", "reference", "target", "method", "training_depth", "n_atlas_samples",
          "n_splits", "n_repeats", "n_train_min", "n_train_max", "n_heldout_min", "n_heldout_max",
          "candidate_unique_EPIC_loci", "legacy_valid_manifest_rows", "n_eligible_min", "n_eligible_max",
          "n_observations", "rmse", "rmse_sd", "sd_unit", "metric_definition", "feature_definition", "source")


def record(path):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": sha}


def read_table(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_table(path, table, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(table)


def combine(cv, holdouts, cv_source, holdout_source):
    cv_keys = [(r["endpoint"], r["target"]) for r in cv]
    expected_cv = {(e, t) for e in ("subject_level_out_of_fold", "historical_heldout_fold_mean") for t in ("CHALM", "CAMDA")}
    hold_keys = [(r["model"], r["target"], r["method"]) for r in holdouts]
    expected_hold = {(m, t, method) for m in ("exclude_blood_immune", "exclude_epithelial")
        for t in ("CHALM", "CAMDA") for method in ("A2H", "reference_linear", "m")}
    if len(cv_keys)!=4 or set(cv_keys)!=expected_cv or len(hold_keys)!=12 or set(hold_keys)!=expected_hold:
        raise ValueError("Missing or ambiguous CV/holdout summary keys")
    result = []
    for r in cv:
        if r["training_depth"] != ">=4" or r["n_atlas_samples"] != "205" or r["n_splits"] != "100":
            raise ValueError("Unexpected whole-atlas CV definition")
        subject = r["endpoint"] == "subject_level_out_of_fold"
        result.append({"endpoint": r["endpoint"], "reference": "whole_atlas_cv", "target": r["target"],
            "method": "A2H", "training_depth": ">=4", "n_atlas_samples": "205", "n_splits": "100", "n_repeats": "10",
            "n_train_min": "184", "n_train_max": "185", "n_heldout_min": "20", "n_heldout_max": "21",
            "candidate_unique_EPIC_loci": r["n_EPIC_candidate_units"] if subject else "",
            "legacy_valid_manifest_rows": "" if subject else r["n_EPIC_candidate_units"],
            "n_eligible_min": r["n_eligible_units_min"], "n_eligible_max": r["n_eligible_units_max"],
            "n_observations": "", "rmse": r["rmse_mean"], "rmse_sd": r["rmse_sd"],
            "sd_unit": "ten repeated-CV summaries" if subject else "100 heldout-fold means",
            "metric_definition": r["metric_definition"], "feature_definition": r["feature_definition"], "source": str(cv_source)})
    for r in holdouts:
        sizes = ("161", "44") if r["model"] == "exclude_blood_immune" else ("122", "83")
        if (r["n_train"], r["n_heldout"]) != sizes:
            raise ValueError("Unexpected tissue-only training/test partition")
        result.append({"endpoint": "fixed_tissue_holdout", "reference": r["model"], "target": r["target"],
            "method": r["method"], "training_depth": ">=4", "n_atlas_samples": "205", "n_splits": "1", "n_repeats": "1",
            "n_train_min": r["n_train"], "n_train_max": r["n_train"], "n_heldout_min": r["n_heldout"], "n_heldout_max": r["n_heldout"],
            "candidate_unique_EPIC_loci": "824231", "legacy_valid_manifest_rows": "",
            "n_eligible_min": r["n_sites"], "n_eligible_max": r["n_sites"], "n_observations": r["n_observations"],
            "rmse": r["mean_site_rmse"], "rmse_sd": "", "sd_unit": "not estimated for this fixed holdout",
            "metric_definition": "Equal-locus mean RMSE over individual heldout samples; same rows for A2H/linear/raw beta; no repeat SD",
            "feature_definition": "Coordinate-verified EPIC loci; >=8 jointly finite heldout observations; same-donor other tissues retained",
            "source": str(holdout_source)})
    return result


def plot(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    references = ("whole_atlas_cv", "exclude_blood_immune", "exclude_epithelial")
    for ax, target in zip(axes, ("CHALM", "CAMDA")):
        data = [next(r for r in table if r["reference"]==ref and r["target"]==target
                     and r["method"]=="A2H" and r["endpoint"]!="historical_heldout_fold_mean") for ref in references]
        y = [float(r["rmse"]) for r in data]
        ax.bar(range(3), y, color=["#60666A", "#276A88", "#BA563A"], width=.6)
        ax.errorbar(0, y[0], yerr=float(data[0]["rmse_sd"]), fmt="none", capsize=4, color="black", linewidth=1)
        for i, value in enumerate(y):
            ax.text(i, value+.0007, f"{value:.5f}", ha="center", fontsize=10)
        ax.set(xticks=range(3), xticklabels=["Whole-atlas CV\n10 repeats x 10 folds", "Blood/immune\n44 held out", "Epithelial\n83 held out"],
               ylabel="A2H mean per-locus RMSE", title=target, ylim=(0, max(y)*1.18))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("R1-6: whole-atlas CV and tissue-only holdouts", fontsize=14)
    fig.supxlabel("CV whisker: repeat SD; fixed holdouts: point estimates.\nDifferent historical/holdout EPIC masks; descriptive comparison only.", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(path/f"R1-6_atlas_cv_and_holdouts.{extension}", dpi=200)
    plt.close(fig)


def run():
    output = RESULTS/"r1_6_with_whole_atlas_cv"
    if output.exists():
        raise FileExistsError("Refusing to overwrite the combined R1-6 bundle")
    cv_source, holdout_source = PACKET/"whole_atlas_cv_summary.tsv", RESULTS/"summary/R1-6_summary.tsv"
    cv_provenance = json.loads((PACKET/"whole_atlas_cv_provenance.json").read_text())
    hold_provenance = json.loads((RESULTS/"summary/completion.json").read_text())
    if cv_provenance["status"] != "PASS" or json.loads((RESULTS/"summary/independent_validation.json").read_text())["status"] != "PASS":
        raise ValueError("Original CV or holdout validation did not pass")
    expected_cv = next(r for r in cv_provenance["outputs"] if Path(r["path"]) == cv_source)
    if record(cv_source)["sha256"] != expected_cv["sha256"] or record(holdout_source)["sha256"] != hold_provenance["outputs"][holdout_source.name]["sha256"]:
        raise ValueError("CV or holdout source hash changed")
    table = combine(read_table(cv_source), read_table(holdout_source), cv_source, holdout_source)
    output.mkdir()
    write_table(output/"R1-6_combined_summary.tsv", table, FIELDS)
    # Keep both original summaries byte-identical as companion artifacts.
    for source, name in ((cv_source, "R1-6_whole_atlas_cv_summary.tsv"), (holdout_source, "R1-6_tissue_holdout_summary.tsv")):
        (output/name).write_bytes(source.read_bytes())
    plot(table, output)
    text = """# R1-6: whole-atlas CV plus tissue holdouts

This combined bundle appends existing depth>=4 whole-atlas CV to the completed
R1-6 tissue-holdout artifacts. No model was retrained and no mask changed.

| A2H evaluation | CHALM mean per-site RMSE | CAMDA mean per-site RMSE |
|---|---:|---:|
| Whole-atlas subject-level CV | 0.0523124607 | 0.0522806990 |
| Blood/immune holdout | 0.0546579023 | 0.0546529470 |
| Epithelial holdout | 0.0574330929 | 0.0574286125 |

CV uses 205 source samples, ten repeats of ten-fold CV, and 184-185 training /
20-21 test samples per split. Its existing historical EPIC mask (824908
candidate loci) and >=2-valid-observation eligibility remain unchanged.
Fixed tissue holdouts use 161/44 and 122/83 train/test samples, coordinate-
verified EPIC loci and >=8-valid-observation eligibility. Values are therefore
a descriptive overview, not a matched-mask estimate of tissue exclusion.
CV whiskers are descriptive repeat SD; no analogous SD was estimated for
the fixed holdouts. Other tissues from the same donors can remain in training.

R1-6_combined_summary.tsv contains all 16 existing target/method/endpoint rows:
subject-level CV, 12 holdout A2H/linear/raw-beta rows, and two separately labeled
historical fold-mean CV rows. The latter have RMSE 0.0163251935 CHALM and
0.0163083343 CAMDA and do not measure individual-sample prediction error.
They are deliberately not plotted on the subject-level comparison axis.
Unknown/unstated CV observation totals and fixed-holdout SD remain missing.

The companion CV and holdout TSVs are byte-identical copies of verified input
summaries. provenance.json gives their source hashes and validation scope.
The PNG/PDF adds whole-atlas subject-level CV alongside both holdouts.
Existing detailed holdout plots and all raw predictions remain untouched in
../figures/ and ../fits/. The reviewer packet contains the exact CV input
artifact references, including both original CHALM/CAMDA CV outputs.
"""
    (output/"README.md").write_text(text)
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "action": "Append verified existing CV and holdout summaries; no refitting or filtering",
        "code": record(Path(__file__)), "inputs": [record(p) for p in (cv_source, holdout_source,
            PACKET/"whole_atlas_cv_provenance.json", RESULTS/"summary/completion.json", RESULTS/"summary/independent_validation.json")],
        "outputs": [record(p) for p in output.iterdir()], "n_combined_rows": len(table),
        "validation": "Unique complete endpoint/target/method keys; source hashes; correct 161/44 and 122/83 membership; original numeric strings preserved; source TSV copies byte-identical",
        "limitations": "Different EPIC masks and eligibility, no matched-subset CV reanalysis; fold-mean CV separated from subject-level error; no new size controls"}
    (output/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Combined {len(table)} rows and figures in {output}", flush=True)


if __name__ == "__main__":
    run()
