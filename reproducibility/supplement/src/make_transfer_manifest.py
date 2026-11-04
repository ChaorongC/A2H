"""Freeze donor-safe atlas cell-class holdouts and reference-size controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash


def class_masks(samples: pd.DataFrame) -> dict[str, pd.Series]:
    group = samples["Group"].astype(str)
    return {
        "blood_immune": group.str.startswith("Blood-"),
        "epithelial": group.str.contains(r"(?:^|-)Ep(?:-|$)", regex=True) |
                      group.isin(["Pancreas-Duct", "Epid-Kerat"]),
    }


def donor_subset(samples: pd.DataFrame, target_size: int, seed: int) -> set[str]:
    donors = samples.groupby("donor_id", sort=False).size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(donors))
    # One retained path per reachable total is enough to find a whole-donor subset.
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for position in order:
        count = int(donors.iloc[position])
        additions = {
            total + count: path + (int(position),)
            for total, path in list(reachable.items())
            if total + count <= target_size and total + count not in reachable
        }
        reachable.update(additions)
    if target_size not in reachable:
        raise ValueError(f"No donor-complete subset of {target_size} reference samples")
    selected = {str(donors.index[position]) for position in reachable[target_size]}
    if int(samples.donor_id.isin(selected).sum()) != target_size:
        raise ValueError("Donor subset size check failed")
    return selected


def build(samples: pd.DataFrame, n_controls: int = 10, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"sample_id", "donor_id", "Group"}
    if not required.issubset(samples) or samples.sample_id.duplicated().any() or samples.donor_id.isna().any():
        raise ValueError("Atlas samples need unique IDs and donor/group metadata")
    masks = class_masks(samples)
    rows = []
    summary = []

    def add(model: str, train: pd.Series, holdout: pd.Series, role_note: str) -> None:
        if (train & holdout).any() or not train.any():
            raise ValueError(f"Invalid transfer membership for {model}")
        for row, is_train, is_test in zip(samples.itertuples(index=False), train, holdout):
            role = "train" if is_train else "heldout" if is_test else "excluded"
            rows.append({"model": model, "sample_id": row.sample_id,
                         "donor_id": row.donor_id, "group": row.Group, "role": role})
        summary.append({"model": model, "n_train": int(train.sum()),
                        "n_train_donors": int(samples.loc[train, "donor_id"].nunique()),
                        "n_heldout": int(holdout.sum()),
                        "n_heldout_donors": int(samples.loc[holdout, "donor_id"].nunique()),
                        "n_excluded": int((~train & ~holdout).sum()),
                        "role_note": role_note})

    all_samples = pd.Series(True, index=samples.index)
    none = pd.Series(False, index=samples.index)
    add("full_reference", all_samples, none, "External prediction only; no atlas holdout")
    for name, heldout in masks.items():
        donors = set(samples.loc[heldout, "donor_id"])
        train = ~samples.donor_id.isin(donors)
        add(f"exclude_{name}", train, heldout,
            "All samples from held-out donors excluded from training; related non-target cells remain excluded")
    add("blood_only", masks["blood_immune"], none,
        "Unavailable under >=50 jointly valid training pairs because only 44 samples")
    target_size = int((~samples.donor_id.isin(samples.loc[masks["blood_immune"], "donor_id"])).sum())
    seen: set[frozenset[str]] = set()
    for replicate in range(n_controls):
        donor_set = frozenset(donor_subset(samples, target_size, seed + replicate))
        if donor_set in seen:
            raise ValueError("Repeated size-control donor subset")
        seen.add(donor_set)
        add(f"size_control_{replicate:02d}", samples.donor_id.isin(donor_set), none,
            f"Donor-complete full-reference subset matched to {target_size} training samples")
    manifest = pd.DataFrame(rows)
    summary_table = pd.DataFrame(summary)
    for model, part in manifest.groupby("model"):
        train_donors = set(part.loc[part.role == "train", "donor_id"])
        test_donors = set(part.loc[part.role == "heldout", "donor_id"])
        if train_donors & test_donors:
            raise ValueError(f"Donor leakage in {model}")
    return manifest, summary_table


def write(sample_path: Path, output_dir: Path, n_controls: int, seed: int) -> None:
    samples = pd.read_csv(sample_path, sep="\t")
    manifest, summary = build(samples, n_controls, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_dir / "reference_membership.tsv", sep="\t", index=False)
    summary.to_csv(output_dir / "reference_models.tsv", sep="\t", index=False)
    provenance = {
        "sample_manifest_sha256": file_hash(sample_path),
        "seed": seed, "n_size_controls": n_controls,
        "blood_rule": "Group starts Blood-",
        "epithelial_rule": "Group has Ep as a hyphen-delimited token or is Pancreas-Duct/Epid-Kerat",
        "donor_rule": "Source:PatientID; every held-out donor entirely excluded from training",
        "size_controls": "ten deterministic whole-donor subsets matching the nonblood training sample count",
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-controls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.n_controls < 1:
        parser.error("--n-controls must be positive")
    write(args.sample_manifest, args.output_dir, args.n_controls, args.seed)


if __name__ == "__main__":
    main()
