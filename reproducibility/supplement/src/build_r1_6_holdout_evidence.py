"""Create a read-only evidence branch for the completed R1-6 atlas holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_checked(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build(source_root: Path, manifest_dir: Path, output: Path) -> None:
    models = ("exclude_blood_immune", "exclude_epithelial")
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    summaries = []
    for model in models:
        source = source_root / model / "summary"
        destination = output / model
        for name in (
            "heldout_summary.tsv", "retention.tsv", "method_differences.tsv",
            "completion.json",
        ):
            src = source / name
            dst = destination / name
            copy_checked(src, dst)
            copied.append({"path": str(dst.relative_to(output)), "sha256": sha256(dst)})
        table = pd.read_csv(destination / "heldout_summary.tsv", sep="\t")
        summaries.append(table)

    for name in ("reference_membership.tsv", "reference_models.tsv", "provenance.json"):
        src = manifest_dir / name
        dst = output / "transfer_manifest" / name
        copy_checked(src, dst)
        copied.append({"path": str(dst.relative_to(output)), "sha256": sha256(dst)})

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "r1_6_heldout_summary.tsv", sep="\t", index=False)
    copied.append({"path": "r1_6_heldout_summary.tsv", "sha256": sha256(output / "r1_6_heldout_summary.tsv")})
    provenance = {
        "analysis": "R1-6 donor-safe atlas functional-cell-group holdout",
        "source_root": str(source_root.resolve()),
        "source_models": list(models),
        "groups": {
            "blood_immune": {"heldout_samples": 44, "heldout_donors": 13},
            "epithelial": {"heldout_samples": 83, "heldout_donors": 65},
        },
        "training_threshold": "historical atlas branch; CT_count >= 4",
        "targets": ["CHALM", "CAMDA"],
        "methods": ["A2H", "reference_linear", "m"],
        "evaluation": "equal-CpG mean of held-out per-site RMSE after requiring >=8 held-out observations per site",
        "donor_rule": "all held-out donors excluded from training",
        "external_validation_not_used": True,
        "copied_files": copied,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_root, args.manifest_dir, args.output)


if __name__ == "__main__":
    main()
