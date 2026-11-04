#!/usr/bin/env python
"""Filter existing WGBS methRatio and CHALM outputs by reported CT_count."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def filter_ratio_file(input_path: Path, output_path: Path, min_depth: int) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    retained_rows = 0
    min_retained_ct_count: float | None = None

    with input_path.open(newline="") as source, output_path.open("w", newline="") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or "CT_count" not in reader.fieldnames:
            raise ValueError(f"Missing CT_count column in {input_path}")
        writer = csv.DictWriter(
            destination,
            fieldnames=reader.fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            input_rows += 1
            ct_count = float(row["CT_count"])
            if ct_count < min_depth:
                continue
            writer.writerow(row)
            retained_rows += 1
            if min_retained_ct_count is None or ct_count < min_retained_ct_count:
                min_retained_ct_count = ct_count

    retained_fraction = retained_rows / input_rows if input_rows else 0.0
    return {
        "input_rows": input_rows,
        "retained_rows": retained_rows,
        "retained_fraction": retained_fraction,
        "min_retained_ct_count": min_retained_ct_count,
    }


def discover_samples(traditional_dir: Path, sample_ids: list[str] | None) -> list[str]:
    if sample_ids:
        return list(dict.fromkeys(sample_ids))
    return sorted(
        path.name.removesuffix("_methRatio.txt")
        for path in traditional_dir.glob("*_methRatio.txt")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-traditional-dir", required=True, type=Path)
    parser.add_argument("--input-chalm-dir", required=True, type=Path)
    parser.add_argument("--output-traditional-dir", required=True, type=Path)
    parser.add_argument("--output-chalm-dir", required=True, type=Path)
    parser.add_argument("--min-depth", required=True, type=int)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_depth < 1:
        raise ValueError("--min-depth must be at least 1")

    samples = discover_samples(args.input_traditional_dir, args.sample_ids)
    if not samples:
        raise ValueError("No WGBS samples were selected for filtering")

    summary_rows = []
    for sample_id in samples:
        files = [
            (
                "methRatio",
                args.input_traditional_dir / f"{sample_id}_methRatio.txt",
                args.output_traditional_dir / f"{sample_id}_methRatio.txt",
            ),
            (
                "CHALM",
                args.input_chalm_dir / f"{sample_id}_CHALM.txt",
                args.output_chalm_dir / f"{sample_id}_CHALM.txt",
            ),
        ]
        for file_type, input_path, output_path in files:
            if not input_path.exists():
                raise FileNotFoundError(f"Missing WGBS input: {input_path}")
            metrics = filter_ratio_file(input_path, output_path, args.min_depth)
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "file_type": file_type,
                    "min_depth": args.min_depth,
                    "input_rows": metrics["input_rows"],
                    "retained_rows": metrics["retained_rows"],
                    "retained_fraction": metrics["retained_fraction"],
                    "min_retained_ct_count": metrics["min_retained_ct_count"],
                    "input_path": input_path,
                    "output_path": output_path,
                }
            )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as summary_handle:
        writer = csv.DictWriter(
            summary_handle,
            fieldnames=list(summary_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
