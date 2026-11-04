"""Freeze historical sample CV and donor-safe grouped CV assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, RepeatedKFold

from src.audit_model import ATLAS
from src.evaluate_pgpuk import file_hash


def load_samples(metadata_path: Path, sample_order: list[str]) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
    metadata["sample_id"] = metadata["Sample name"].str.extract(r"(Z[0-9A-Z]+)$")[0]
    if metadata.sample_id.isna().any() or metadata.sample_id.duplicated().any():
        raise ValueError("Sample IDs in atlas metadata are missing or duplicated")
    if len(sample_order) != 205 or len(set(sample_order)) != 205 or set(metadata.sample_id) != set(sample_order):
        raise ValueError("Atlas metadata does not match the 205-sample matrix order")
    if metadata[["Source", "PatientID", "Group"]].isna().any().any():
        raise ValueError("Missing source, donor, or tissue-group metadata")
    metadata["donor_id"] = metadata.Source.astype(str) + ":" + metadata.PatientID.astype(str)
    return metadata.set_index("sample_id").loc[sample_order].reset_index()[
        ["sample_id", "donor_id", "Source", "PatientID", "Group", "Refined group", "Source Tissue"]
    ]


def make_splits(samples: pd.DataFrame, branch: str, n_splits: int = 10,
                n_repeats: int = 10, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    if branch not in {"historical_sample", "donor_grouped"}:
        raise ValueError("Unknown CV branch")
    groups = samples.donor_id.to_numpy()
    if branch == "historical_sample":
        split_iter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(samples)
    else:
        split_iter = (
            pair for repeat in range(n_repeats)
            for pair in GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat).split(samples, groups=groups)
        )
    rows = []
    audit = []
    for split_index, (train, test) in enumerate(split_iter):
        repeat, fold = divmod(split_index, n_splits)
        train_donors = set(groups[train])
        test_donors = set(groups[test])
        shared = train_donors & test_donors
        if branch == "donor_grouped" and shared:
            raise ValueError("Grouped CV leaked donors")
        audit.append({"branch": branch, "split_index": split_index, "repeat": repeat, "fold": fold,
                      "n_train_samples": len(train), "n_test_samples": len(test),
                      "shared_donors": len(shared), "n_test_donors": len(test_donors)})
        for role, indices in (("train", train), ("test", test)):
            for index in indices:
                rows.append({"branch": branch, "split_index": split_index,
                             "repeat": repeat, "fold": fold, "role": role,
                             "sample_id": samples.sample_id.iloc[index], "donor_id": groups[index]})
    manifest = pd.DataFrame(rows)
    for repeat, part in manifest.loc[manifest.role == "test"].groupby("repeat"):
        if len(part) != len(samples) or part.sample_id.duplicated().any() or part.split_index.nunique() != n_splits:
            raise ValueError(f"Incomplete out-of-fold coverage in repeat {repeat}")
    if len(audit) != n_splits * n_repeats:
        raise ValueError("Unexpected number of CV splits")
    return manifest, pd.DataFrame(audit)


def write(atlas_root: Path, control_root: Path, output: Path) -> None:
    metadata_path = atlas_root / "nature_methylation_atlas_sample_info.csv"
    controls_path = control_root / "provenance.json"
    controls = json.loads(controls_path.read_text())
    samples = load_samples(metadata_path, controls["reference_samples"])
    output.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output / "atlas_samples.tsv", sep="\t", index=False)
    branches = []
    audits = []
    for branch in ("historical_sample", "donor_grouped"):
        manifest, audit = make_splits(samples, branch)
        branches.append(manifest)
        audits.append(audit)
    pd.concat(branches, ignore_index=True).to_csv(output / "splits.tsv", sep="\t", index=False)
    pd.concat(audits, ignore_index=True).to_csv(output / "split_audit.tsv", sep="\t", index=False)
    details = {
        "metadata_sha256": file_hash(metadata_path),
        "control_manifest_sha256": file_hash(controls_path),
        "sample_order_source": "reference_controls_original/provenance.json; original 205-sample methylation matrix",
        "historical_sample": "sklearn RepeatedKFold(10 splits, 10 repeats, random_state=0)",
        "donor_grouped": "sklearn GroupKFold(10 splits, shuffle=True, random_state=repeat), donor=Source:PatientID",
        "n_samples": len(samples), "n_donors": int(samples.donor_id.nunique()),
    }
    (output / "provenance.json").write_text(json.dumps(details, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ATLAS)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write(args.atlas_root, args.control_root, args.output_dir)


if __name__ == "__main__":
    main()
