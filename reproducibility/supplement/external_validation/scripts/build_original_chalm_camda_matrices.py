#!/usr/bin/env python
"""Build original methRatio, CHALM, and CAMDA matrices for EPIC-covered hg38 loci."""

from __future__ import annotations

import argparse
from pathlib import Path

from chalm_array_from_methratio import (
    DEFAULT_CHALM_ARRAY_ROOT,
    build_original_camda_matrix,
    build_original_ratio_matrix,
    discover_sample_files,
)


DEFAULT_METHRATIO_DIR = Path("results/traditional")
DEFAULT_CHALM_DIR = Path("results/chalm")
DEFAULT_OUTPUT_DIR = Path("results/chalm_array/full")


def resolve_probe_map_path(chalm_array_root: Path, reference: str) -> Path:
    candidates = [
        chalm_array_root / f"CHALM-array_cpg_probe.{reference}.tsv",
        chalm_array_root / f"A2H_cpg_probe.{reference}.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing probe map. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build original locus-indexed methRatio/CHALM/CAMDA matrices restricted to EPIC-covered hg38 loci."
    )
    parser.add_argument(
        "--methratio-dir",
        default=str(DEFAULT_METHRATIO_DIR),
        help="Directory containing *_methRatio.txt files.",
    )
    parser.add_argument(
        "--chalm-dir",
        default=str(DEFAULT_CHALM_DIR),
        help="Directory containing *_CHALM.txt files.",
    )
    parser.add_argument(
        "--chalm-array-root",
        default=str(DEFAULT_CHALM_ARRAY_ROOT),
        help="Path to the CHALM-array repo root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the original comparison matrices.",
    )
    parser.add_argument(
        "--prefix",
        default="wgbs_epic_hg38",
        help="Prefix for output matrix names.",
    )
    parser.add_argument(
        "--ref",
        default="hg38",
        choices=["hg38", "hg19"],
    )
    parser.add_argument(
        "--beadchip",
        default="EPIC",
        choices=["450k", "EPIC", "HM450", "850k"],
    )
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        help="Sample ID to include. Repeat to restrict the matrix to a subset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    methratio_dir = Path(args.methratio_dir)
    chalm_dir = Path(args.chalm_dir)
    chalm_array_root = Path(args.chalm_array_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_map_path = resolve_probe_map_path(chalm_array_root, args.ref)
    methratio_files = discover_sample_files(methratio_dir, sample_ids=args.samples)
    chalm_files = {
        sample_id: chalm_dir / f"{sample_id}_CHALM.txt"
        for sample_id in methratio_files
    }

    original_methratio = build_original_ratio_matrix(
        methratio_files,
        probe_map_path=probe_map_path,
        reference=args.ref,
        beadchip=args.beadchip,
    )
    original_chalm = build_original_ratio_matrix(
        chalm_files,
        probe_map_path=probe_map_path,
        reference=args.ref,
        beadchip=args.beadchip,
    )
    original_camda = build_original_camda_matrix(original_methratio, original_chalm)

    outputs = {
        "original_methratio": original_methratio,
        "original_chalm": original_chalm,
        "original_camda": original_camda,
    }
    for suffix, frame in outputs.items():
        path = output_dir / f"{args.prefix}.{suffix}.tsv"
        frame.to_csv(path, sep="\t", index=False)
        print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
