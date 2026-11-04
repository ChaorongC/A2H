#!/usr/bin/env python
"""Build an EPIC-style methylation matrix from WGBS methRatio files and run CHALM-array."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CHALM_ARRAY_ROOT = Path("legacy_model")
DEFAULT_METHRATIO_DIR = Path("results/traditional")
DEFAULT_OUTPUT_DIR = Path("results/chalm_array")


def resolve_probe_column(reference: str, beadchip: str) -> str:
    beadchip_upper = beadchip.upper()
    if beadchip_upper in {"EPIC", "850K"}:
        return f"EPIC.{reference}"
    if beadchip_upper in {"HM450", "450K"}:
        return f"HM450.{reference}"
    raise ValueError(f"Unsupported beadchip value: {beadchip}")


def discover_sample_files(methratio_dir: Path, sample_ids: list[str] | None = None) -> dict[str, Path]:
    if sample_ids:
        return {
            sample_id: methratio_dir / f"{sample_id}_methRatio.txt"
            for sample_id in sample_ids
        }

    sample_paths = sorted(methratio_dir.glob("*_methRatio.txt"))
    return {
        path.name.removesuffix("_methRatio.txt"): path
        for path in sample_paths
    }


def load_probe_map(probe_map_path: Path, reference: str, beadchip: str) -> pd.DataFrame:
    probe_column = resolve_probe_column(reference, beadchip)
    probe_map = (
        pd.read_table(probe_map_path, usecols=["loci.hg38", probe_column])
        .dropna()
        .drop_duplicates()
        .rename(columns={probe_column: "Probe_ID"})
    )
    return probe_map


def build_probe_matrix(
    sample_to_path: dict[str, Path],
    probe_map_path: Path,
    reference: str = "hg38",
    beadchip: str = "EPIC",
) -> pd.DataFrame:
    probe_map = load_probe_map(probe_map_path, reference=reference, beadchip=beadchip)

    sample_series = []
    for sample_id, methratio_path in sample_to_path.items():
        methratio = pd.read_csv(
            methratio_path,
            sep="\t",
            usecols=["chr", "pos", "context", "ratio"],
        )
        methratio = methratio[methratio["context"] == "CG"].copy()
        methratio["loci.hg38"] = (
            methratio["chr"].astype(str) + "_" + methratio["pos"].astype(str)
        )
        matched = (
            methratio[["loci.hg38", "ratio"]]
            .merge(probe_map, on="loci.hg38", how="inner")
            .drop_duplicates(subset=["Probe_ID"])
            .set_index("Probe_ID")["ratio"]
            .rename(sample_id)
        )
        sample_series.append(matched)

    matrix = pd.concat(sample_series, axis=1, sort=True).reset_index()
    matrix = matrix.rename(columns={"index": "Probe_ID"})
    matrix = matrix.loc[matrix.iloc[:, 1:].notna().any(axis=1)].copy()
    matrix = matrix.sort_values("Probe_ID").reset_index(drop=True)
    return matrix


def build_original_ratio_matrix(
    sample_to_path: dict[str, Path],
    probe_map_path: Path,
    reference: str = "hg38",
    beadchip: str = "EPIC",
) -> pd.DataFrame:
    probe_map = load_probe_map(probe_map_path, reference=reference, beadchip=beadchip)
    epic_loci = set(probe_map["loci.hg38"])

    sample_series = []
    for sample_id, ratio_path in sample_to_path.items():
        ratio_table = pd.read_csv(
            ratio_path,
            sep="\t",
            usecols=["chr", "pos", "context", "ratio"],
        )
        ratio_table = ratio_table[ratio_table["context"] == "CG"].copy()
        ratio_table["Probe_ID"] = (
            ratio_table["chr"].astype(str) + "_" + ratio_table["pos"].astype(str)
        )
        matched = (
            ratio_table[ratio_table["Probe_ID"].isin(epic_loci)][["Probe_ID", "ratio"]]
            .drop_duplicates(subset=["Probe_ID"])
            .set_index("Probe_ID")["ratio"]
            .rename(sample_id)
        )
        sample_series.append(matched)

    matrix = pd.concat(sample_series, axis=1, sort=True).reset_index()
    matrix = matrix.rename(columns={"index": "Probe_ID"})
    matrix = matrix.loc[matrix.iloc[:, 1:].notna().any(axis=1)].copy()
    matrix = matrix.sort_values("Probe_ID").reset_index(drop=True)
    return matrix


def build_original_camda_matrix(
    methratio_matrix: pd.DataFrame,
    chalm_matrix: pd.DataFrame,
) -> pd.DataFrame:
    meth = methratio_matrix.set_index("Probe_ID")
    chalm = chalm_matrix.set_index("Probe_ID")
    common_columns = list(meth.columns)
    camda = (chalm[common_columns] - meth[common_columns]).reset_index()
    return camda


def filter_matrix_by_confidence(
    matrix_df: pd.DataFrame,
    confidence_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    matrix = matrix_df.set_index("Probe_ID")
    confidence = confidence_df.set_index("Probe_ID")
    shared_columns = list(matrix.columns)
    filtered = matrix.where(confidence[shared_columns] >= threshold)
    return filtered.reset_index()


def build_chalm_array_command(
    python_executable: str,
    chalm_array_root: Path,
    matrix_path: Path,
    output_dir: Path,
    prefix: str,
    reference: str,
    beadchip: str,
    threads: int,
    camda: bool,
    filter_threshold: float | None,
) -> list[str]:
    command = [
        python_executable,
        str(chalm_array_root / "CHALM_array.py"),
        "-i",
        str(matrix_path),
        "-o",
        str(output_dir),
        "--prefix",
        prefix,
        "--ref",
        reference,
        "--beadchip",
        beadchip,
        "-t",
        str(threads),
    ]
    if camda:
        command.append("--camda")
    if filter_threshold is not None:
        command.extend(["--filter", str(filter_threshold)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WGBS methRatio outputs into a CHALM-array EPIC matrix and optionally run CHALM-array."
    )
    parser.add_argument(
        "--methratio-dir",
        default=str(DEFAULT_METHRATIO_DIR),
        help="Directory containing *_methRatio.txt files.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        help="Sample ID to include. Repeat to restrict the matrix to a subset.",
    )
    parser.add_argument(
        "--chalm-array-root",
        default=str(DEFAULT_CHALM_ARRAY_ROOT),
        help="Path to the CHALM-array repo root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for the adapter matrix and CHALM-array outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="wgbs_epic_hg38",
        help="Prefix for the adapter matrix and CHALM-array outputs.",
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
        "--threads",
        default=-1,
        type=int,
        help="Thread count passed through to CHALM-array.",
    )
    parser.add_argument(
        "--filter",
        dest="filter_threshold",
        default=None,
        type=float,
        help="Minimum confidence threshold passed to CHALM-array.",
    )
    parser.add_argument(
        "--camda",
        action="store_true",
        help="Request CAMDA outputs from CHALM-array.",
    )
    parser.add_argument(
        "--matrix-only",
        action="store_true",
        help="Build the EPIC probe matrix but do not run CHALM-array.",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable used to run CHALM-array.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methratio_dir = Path(args.methratio_dir)
    chalm_array_root = Path(args.chalm_array_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_map_path = chalm_array_root / f"CHALM-array_cpg_probe.{args.ref}.tsv"
    sample_to_path = discover_sample_files(methratio_dir, sample_ids=args.samples)
    missing = [str(path) for path in sample_to_path.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing methRatio inputs: {missing}")

    matrix = build_probe_matrix(
        sample_to_path,
        probe_map_path=probe_map_path,
        reference=args.ref,
        beadchip=args.beadchip,
    )
    matrix_path = output_dir / f"{args.prefix}.matrix.tsv"
    matrix.to_csv(matrix_path, sep="\t", index=False)
    print(f"Wrote CHALM-array input matrix to {matrix_path}")

    if args.matrix_only:
        return 0

    command = build_chalm_array_command(
        python_executable=args.python_executable,
        chalm_array_root=chalm_array_root,
        matrix_path=matrix_path,
        output_dir=output_dir,
        prefix=args.prefix,
        reference=args.ref,
        beadchip=args.beadchip,
        threads=args.threads,
        camda=args.camda,
        filter_threshold=args.filter_threshold,
    )
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
