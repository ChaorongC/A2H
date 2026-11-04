"""Append selected historical CV to the R1-6 summary artifact family."""
from datetime import datetime, timezone
import json
from pathlib import Path

from .package_r1_6_cv_and_holdouts import (
    FIELDS, PACKET, RESULTS, combine, read_table, record, write_table,
)


def plot(table, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), layout="constrained")
    cv = [next(r for r in table if r["endpoint"] == "historical_heldout_fold_mean"
               and r["target"] == target) for target in ("CHALM", "CAMDA")]
    ax = axes[0]
    values = [float(r["rmse"]) for r in cv]
    errors = [float(r["rmse_sd"]) for r in cv]
    ax.bar([0, 1], values, yerr=errors, capsize=4, color=["#276A88", "#BA563A"], width=.55)
    for i, value in enumerate(values):
        ax.text(i, value+errors[i]+.0005, f"{value:.5f}", ha="center", fontsize=10)
    ax.set(title="Whole-atlas CV\nHeld-out-fold averages", ylabel="Cross-CpG RMSE of fold averages",
           xticks=[0, 1], xticklabels=["CHALM", "CAMDA"], ylim=(0, .022))
    ax.text(.5, .97, "A2H; mean +/- SD across 100 folds", transform=ax.transAxes,
            ha="center", va="top", fontsize=9)
    methods = ("A2H", "reference_linear", "m")
    for ax, target in zip(axes[1:], ("CHALM", "CAMDA")):
        for j, (method, label, color) in enumerate(zip(methods, ("A2H", "CpG-specific linear", "Raw beta"),
                                                    ("#276A88", "#BA563A", "#777777"))):
            data = [next(r for r in table if r["reference"] == ref and r["target"] == target
                         and r["method"] == method) for ref in ("exclude_blood_immune", "exclude_epithelial")]
            ax.bar([i+(j-1)*.25 for i in (0, 1)], [float(r["rmse"]) for r in data],
                   width=.24, color=color, label=label)
        ax.set(title=f"{target}: tissue holdouts\nIndividual-sample errors", ylabel="Mean per-site RMSE",
               xticks=[0, 1], xticklabels=["Blood/immune\n44 held out", "Epithelial\n83 held out"])
        ax.set_ylim(0, ax.get_ylim()[1]*1.10)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("R1-6: historical whole-atlas CV and tissue-holdout results", fontsize=14)
    fig.supxlabel("Different endpoints, EPIC eligibility and axis scales; do not interpret CV-to-holdout differences as a tissue-exclusion effect.\nCV whiskers show fold SD; fixed holdouts have no repeat SD. Training depth >=4 throughout.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(prefix.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)


def run():
    summary = RESULTS/"summary"
    cv_source, hold_source = PACKET/"whole_atlas_cv_summary.tsv", summary/"R1-6_summary.tsv"
    outputs = [summary/name for name in (
        "R1-6_summary_with_historical_cv.tsv", "R1-6_with_historical_cv.png",
        "R1-6_with_historical_cv.pdf", "R1-6_with_historical_cv.md",
        "R1-6_with_historical_cv_provenance.json")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Refusing to overwrite appended R1-6 summary artifacts")
    cv_meta = json.loads((PACKET/"whole_atlas_cv_provenance.json").read_text())
    hold_meta = json.loads((summary/"completion.json").read_text())
    assert cv_meta["status"] == "PASS"
    assert json.loads((summary/"independent_validation.json").read_text())["status"] == "PASS"
    assert record(cv_source)["sha256"] == next(r["sha256"] for r in cv_meta["outputs"] if r["path"] == str(cv_source))
    assert record(hold_source)["sha256"] == hold_meta["outputs"][hold_source.name]["sha256"]
    inputs = [record(p) for p in (cv_source, hold_source, PACKET/"whole_atlas_cv_provenance.json",
                                 summary/"completion.json", summary/"independent_validation.json")]
    cv, hold = read_table(cv_source), read_table(hold_source)
    table = [r for r in combine(cv, hold, cv_source, hold_source) if r["endpoint"] != "subject_level_out_of_fold"]
    for r in table:
        if r["endpoint"] == "historical_heldout_fold_mean":
            source = next(s for s in cv if s["endpoint"] == r["endpoint"] and s["target"] == r["target"])
            r.update(pearson=source["pearson_mean"], pearson_sd=source["pearson_sd"], median_site_rmse="",
                     pearson_definition="Cross-CpG Pearson of heldout-fold averages; mean and SD across 100 folds")
        else:
            source = next(s for s in hold if (s["model"], s["target"], s["method"]) == (r["reference"], r["target"], r["method"]))
            r.update(pearson="", pearson_sd="", median_site_rmse=source["median_site_rmse"], pearson_definition="not supplied in original holdout summary")
    assert len(table) == 14
    write_table(outputs[0], table, (*FIELDS, "median_site_rmse", "pearson", "pearson_sd", "pearson_definition"))
    plot(table, summary/"R1-6_with_historical_cv")
    outputs[3].write_text("""# R1-6 summary with historical CV

The appended table has two selected historical_heldout_fold_mean CV rows and
all 12 original tissue-holdout rows. It excludes subject_level_out_of_fold.
All saved numerical strings are preserved; no model or metric is recomputed.

- R1-6_summary_with_historical_cv.tsv: combined results with explicit endpoints.
- R1-6_with_historical_cv.png/.pdf: CV and holdouts in separate labeled panels.
- R1-6_with_historical_cv_provenance.json: source/code/output SHA-256 hashes.

CV uses the depth>=4 205-sample atlas, 10 repeats x 10 folds, 184-185 train /
20-21 test samples per split. For each CpG/fold, average observed and predicted
values across held-out samples first; then score across EPIC CpGs. Summarize
100 fold metrics with mean and SD (ddof=1). CHALM/CAMDA RMSE is
0.016325193502873554 / 0.01630833431921187. These are errors of fold averages,
not individual-sample errors or within-CpG subject ranking. Historical EPIC
manifest rows include duplicate IDs (824933-824939 valid rows per fold).

Tissue holdouts retain individual-sample mean per-site RMSE, coordinate-
verified EPIC loci and >=8 valid held-out observations. Training/test sizes
are 161/44 and 122/83; same-donor other tissues remain eligible. Missing SD
and Pearson remain blank. CV and tissue-holdout endpoints, eligibility and
plot scales differ; their numerical differences are not tissue-exclusion
effects. Fold SD is descriptive, not an independent-cohort confidence interval.

Original R1-6_summary.tsv, site metrics, fit-status counts and plots remain
unchanged. The original completion.json verifies the original outputs; this
appendix has its own provenance. Historical subject-level CV remains archived
in ../r1_6_with_whole_atlas_cv/, outside this selected presentation.
""")
    for entry in inputs:
        assert record(entry["path"])["sha256"] == entry["sha256"]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
                  "selected_cv_endpoint": "historical_heldout_fold_mean", "n_rows": len(table),
                  "action": "Append existing summaries without changing scientific definitions or original outputs",
                  "inputs": inputs, "code": [record(Path(__file__)), record(Path(__file__).with_name("package_r1_6_cv_and_holdouts.py"))],
                  "outputs": [record(p) for p in outputs[:-1]],
                  "limitations": "Different CV/holdout endpoints and masks; no matched-effect claim; no new fitting or filtering"}
    outputs[-1].write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Saved {len(table)} rows and PNG/PDF directly under {summary}", flush=True)


if __name__ == "__main__":
    run()
