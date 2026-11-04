"""Build EPIC-window production manifests for the E2/E3 conditions still to run (2026-10-15).

A condition is left out when its full-genome run already finished (run_metadata.json)
or is still running (its output directory is non-empty).  Native directories: the
re-sorted namespace for samples in recovery_resort_20261015, the original otherwise.
Usage (from R/): python results/r2_2_e2e3_20261015/build_window_manifests.py
"""

import csv
from pathlib import Path

B = Path("results/r2_2_e2e3_20261015")
ROOT = Path("work/r2_2_e2e3_20261015")
OLD = Path("work/r2_2_recovery_20261012")
RESORT = Path("results/r2_2_sensitivity_20261011/recovery_resort_20261015/resort_manifest.tsv")


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def pending(out_dir: Path) -> bool:
    return not out_dir.exists() or not any(out_dir.iterdir())


def write(path, header, records):
    with path.open("x", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(records)
    print(path, len(records))


resorted = {row["sample_id"]: row["out_dir"] for row in rows(RESORT)}
e2 = rows(B / "manifests/e2_manifest.tsv")
e3 = [row for row in rows(B / "manifests/e3_manifest.tsv") if row["native_dir"] != "NA"]  # atlas rows only
samples = sorted({row["sample_id"] for row in e2})
native = {s: resorted.get(s, str(OLD / f"native_{s}")) for s in samples}

write(B / "manifests/window_native_manifest.tsv", ["task", "sample_id", "native_dir", "window_dir"],
      [[i, s, native[s], ROOT / "window_native" / s] for i, s in enumerate(samples)])
e2_out = [(r, ROOT / "e2" / r["sample_id"] / f"seed{r['seed']}_depth{r['target_depth']}") for r in e2]
write(B / "manifests/e2_window_manifest.tsv",
      ["task", "sample_id", "seed", "target_depth", "native_dir", "window_dir", "out_dir", "e2_manifest_task"],
      [[i, r["sample_id"], r["seed"], r["target_depth"], native[r["sample_id"]],
        ROOT / "window_native" / r["sample_id"], out, r["task"]]
       for i, (r, out) in enumerate((r, out) for r, out in e2_out if pending(out))])
write(B / "manifests/e3_window_manifest.tsv",
      ["task", "sample_id", "condition", "native_dir", "window_dir", "out_dir", "e3_manifest_task"],
      [[i, r["sample_id"], r["condition"], native[r["sample_id"]], ROOT / "window_native" / r["sample_id"],
        r["out_dir"], r["task"]]
       for i, r in enumerate(r for r in e3 if pending(Path(r["out_dir"])))])
