#!/usr/bin/env python
"""Copy existing GSE240482 outputs from the old prefix to the final cohort name."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT = Path("external_validation")
OLD_PREFIX = "epicv2_eval_EPIC"
NEW_PREFIX = "gse240482_prostate_epicv1_EPIC"


COPY_GROUPS = {
    Path("results/paired_validation_v2/array"): [
        "beta.tsv",
        "sample_qc.tsv",
    ],
    Path("results/paired_validation_v2/a2h_array"): [
        "chalm.tsv",
        "camda.tsv",
        "confidence.tsv",
        "chalm.filtered_005.tsv",
        "camda.filtered_005.tsv",
        "chalm.filtered_05.tsv",
        "camda.filtered_05.tsv",
    ],
    Path("results/paired_validation_v2/wgbs/probe_space"): [
        "traditional_probe_matrix.tsv",
        "chalm_probe_matrix.tsv",
        "camda_probe_matrix.tsv",
    ],
    Path("results/paired_validation_v2/wgbs/a2h_from_traditional"): [
        "chalm.tsv",
        "camda.tsv",
        "confidence.tsv",
        "chalm.filtered_005.tsv",
        "camda.filtered_005.tsv",
        "chalm.filtered_05.tsv",
        "camda.filtered_05.tsv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--old-prefix", default=OLD_PREFIX)
    parser.add_argument("--new-prefix", default=NEW_PREFIX)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing alias files. Source files are never modified.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print copy operations without writing files.",
    )
    return parser.parse_args()


def copy_one(src: Path, dest: Path, *, overwrite: bool, dry_run: bool) -> str:
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    if dest.exists() and not overwrite:
        return f"kept_existing\t{dest}"
    if dry_run:
        return f"copy\t{src}\t{dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return f"copied\t{src}\t{dest}"


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    for rel_dir, suffixes in COPY_GROUPS.items():
        for suffix in suffixes:
            src = root / rel_dir / f"{args.old_prefix}.{suffix}"
            dest = root / rel_dir / f"{args.new_prefix}.{suffix}"
            print(copy_one(src, dest, overwrite=args.overwrite, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
