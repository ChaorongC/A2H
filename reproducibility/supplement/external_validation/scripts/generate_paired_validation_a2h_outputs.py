#!/usr/bin/env python
"""Generate persisted A2H outputs for paired-validation array and WGBS matrices."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("external_validation")
DEFAULT_A2H_ROOT = Path("legacy_model")
DEFAULT_PYTHON = Path("python")


NAMESPACE_CONFIG = {
    "paired_validation": {
        "sample_manifest": Path("metadata/paired_samples.tsv"),
        "sample_study_col": "study",
        "sample_platform_col": "array_platform",
        "array_manifest": Path("metadata/paired_array_idat_manifest.tsv"),
        "array_manifest_study_col": "study",
        "array_manifest_platform_col": "array_platform",
        "array_manifest_id_col": "array_gsm",
        "array_dir": Path("results/paired_validation/array"),
        "a2h_array_dir": Path("results/paired_validation/a2h_array"),
        "wgbs_input_dir": Path("results/paired_validation/wgbs/probe_space"),
        "a2h_wgbs_dir": Path("results/paired_validation/wgbs/a2h_from_traditional"),
    },
    "paired_validation_v2": {
        "sample_manifest": Path("metadata/paired_validation_v2/paired_samples.tsv"),
        "sample_study_col": "study",
        "sample_platform_col": "platform",
        "array_manifest": Path("metadata/paired_validation_v2/paired_array_idat_manifest.tsv"),
        "array_manifest_study_col": "study",
        "array_manifest_platform_col": "platform",
        "array_manifest_id_col": "array_id",
        "array_dir": Path("results/paired_validation_v2/array"),
        "a2h_array_dir": Path("results/paired_validation_v2/a2h_array"),
        "wgbs_input_dir": Path("results/paired_validation_v2/wgbs/probe_space"),
        "a2h_wgbs_dir": Path("results/paired_validation_v2/wgbs/a2h_from_traditional"),
    },
}


def detect_threads() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK")
    if value:
        return max(int(value), 1)
    cpu_count = os.cpu_count() or 1
    return min(cpu_count, 8)


def infer_platform_beadchip(platform: str) -> str:
    if platform.upper() == "450K":
        return "450k"
    if platform.upper() == "EPIC":
        return "EPIC"
    raise ValueError(f"Unsupported platform: {platform}")


def resolve_probe_map_path(a2h_root: Path, reference: str) -> Path:
    candidates = [
        a2h_root / f"A2H_cpg_probe.{reference}.tsv",
        a2h_root / f"CHALM-array_cpg_probe.{reference}.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing probe map. Checked: " + ", ".join(str(candidate) for candidate in candidates)
    )


def infer_array_beta_column(local_path: str) -> str:
    name = Path(local_path).name
    for suffix in ("_Grn.idat.gz", "_Red.idat.gz", "_Grn.idat", "_Red.idat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise ValueError(f"Unsupported IDAT filename: {local_path}")


def normalize_a2h_output(
    path: Path,
    rename_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    first_col = df.columns[0]
    if first_col != "Probe_ID":
        df = df.rename(columns={first_col: "Probe_ID"})
    if rename_columns:
        df = df.rename(columns=rename_columns)
    return df


def write_normalized_output(
    src: Path,
    dest: Path,
    rename_columns: dict[str, str] | None = None,
) -> None:
    df = normalize_a2h_output(src, rename_columns=rename_columns)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, sep="\t", index=False)


def load_groups(root: Path, namespace: str, study: str | None, platform: str | None) -> list[tuple[str, str]]:
    cfg = NAMESPACE_CONFIG[namespace]
    paired = pd.read_table(root / cfg["sample_manifest"])
    groups = (
        paired[[cfg["sample_study_col"], cfg["sample_platform_col"]]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    normalized = [(study_name, plat) for study_name, plat in groups]
    if study or platform:
        if not study or not platform:
            raise ValueError("--study and --platform must be provided together")
        normalized = [(study_name, plat) for study_name, plat in normalized if study_name == study and plat == platform]
    if not normalized:
        raise ValueError("No paired-validation groups matched the requested filter")
    return normalized


def build_array_rename_map(root: Path, namespace: str, study: str, platform: str) -> dict[str, str]:
    cfg = NAMESPACE_CONFIG[namespace]
    manifest = pd.read_table(root / cfg["array_manifest"])
    study_col = cfg["array_manifest_study_col"]
    platform_col = cfg["array_manifest_platform_col"]
    id_col = cfg["array_manifest_id_col"]
    filtered = manifest[(manifest[study_col] == study) & (manifest[platform_col] == platform)].copy()
    filtered["array_beta_column"] = filtered["local_path"].map(infer_array_beta_column)
    filtered = filtered.drop_duplicates(subset=["array_beta_column"])
    return dict(zip(filtered["array_beta_column"], filtered[id_col], strict=True))


def run_a2h_once(
    *,
    python_bin: Path,
    a2h_root: Path,
    input_path: Path,
    prefix: str,
    platform: str,
    output_dir: Path,
    threads: int,
    filter_threshold: float,
) -> None:
    resolve_probe_map_path(a2h_root, "hg38")
    cmd = [
        str(python_bin),
        str(a2h_root / "A2H.py"),
        "-i",
        str(input_path),
        "-o",
        str(output_dir),
        "--prefix",
        prefix,
        "--ref",
        "hg38",
        "--beadchip",
        infer_platform_beadchip(platform),
        "-t",
        str(threads),
        "--probe",
        "--chalm",
        "--camda",
        "--filter",
        str(filter_threshold),
        "--cpg-info",
        str(a2h_root / "A2H_cpg_info.pkl"),
        "-c",
        str(a2h_root / "A2H_coef.pkl"),
        "--verbose",
        "0",
    ]
    subprocess.run(cmd, check=True)


def generate_outputs_for_input(
    *,
    python_bin: Path,
    a2h_root: Path,
    input_path: Path,
    output_dir: Path,
    prefix: str,
    platform: str,
    threads: int,
    rename_columns: dict[str, str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{prefix}.a2h.", dir=str(output_dir)) as tmpdir:
        tmpdir_path = Path(tmpdir)

        run_a2h_once(
            python_bin=python_bin,
            a2h_root=a2h_root,
            input_path=input_path,
            prefix=prefix,
            platform=platform,
            output_dir=tmpdir_path,
            threads=threads,
            filter_threshold=0.05,
        )
        write_normalized_output(
            tmpdir_path / f"{prefix}.chalm.tsv",
            output_dir / f"{prefix}.chalm.tsv",
            rename_columns=rename_columns,
        )
        write_normalized_output(
            tmpdir_path / f"{prefix}.camda.tsv",
            output_dir / f"{prefix}.camda.tsv",
            rename_columns=rename_columns,
        )
        write_normalized_output(
            tmpdir_path / f"{prefix}.confidence.tsv",
            output_dir / f"{prefix}.confidence.tsv",
            rename_columns=rename_columns,
        )
        write_normalized_output(
            tmpdir_path / f"{prefix}.chalm.filtered.tsv",
            output_dir / f"{prefix}.chalm.filtered_005.tsv",
            rename_columns=rename_columns,
        )
        write_normalized_output(
            tmpdir_path / f"{prefix}.camda.filtered.tsv",
            output_dir / f"{prefix}.camda.filtered_005.tsv",
            rename_columns=rename_columns,
        )

        run_a2h_once(
            python_bin=python_bin,
            a2h_root=a2h_root,
            input_path=input_path,
            prefix=prefix,
            platform=platform,
            output_dir=tmpdir_path,
            threads=threads,
            filter_threshold=0.5,
        )
        write_normalized_output(
            tmpdir_path / f"{prefix}.chalm.filtered.tsv",
            output_dir / f"{prefix}.chalm.filtered_05.tsv",
            rename_columns=rename_columns,
        )
        write_normalized_output(
            tmpdir_path / f"{prefix}.camda.filtered.tsv",
            output_dir / f"{prefix}.camda.filtered_05.tsv",
            rename_columns=rename_columns,
        )

    shutil.rmtree(output_dir / f"{prefix}.a2h.", ignore_errors=True)


def process_group(
    *,
    root: Path,
    namespace: str,
    study: str,
    platform: str,
    mode: str,
    python_bin: Path,
    a2h_root: Path,
    threads: int,
    wgbs_input_dir: Path | None = None,
    a2h_wgbs_dir: Path | None = None,
    input_prefix: str | None = None,
    output_prefix: str | None = None,
) -> None:
    cfg = NAMESPACE_CONFIG[namespace]
    prefix = input_prefix or f"{study}_{platform}"
    output_name = output_prefix or prefix

    if mode in {"array", "both"}:
        array_input = root / cfg["array_dir"] / f"{prefix}.beta.tsv"
        if not array_input.exists():
            raise FileNotFoundError(f"Missing array beta matrix: {array_input}")
        rename_map = build_array_rename_map(root, namespace, study, platform)
        generate_outputs_for_input(
            python_bin=python_bin,
            a2h_root=a2h_root,
            input_path=array_input,
            output_dir=root / cfg["a2h_array_dir"],
            prefix=output_name,
            platform=platform,
            threads=threads,
            rename_columns=rename_map,
        )

    if mode in {"wgbs", "both"}:
        input_dir = wgbs_input_dir if wgbs_input_dir is not None else cfg["wgbs_input_dir"]
        output_dir = a2h_wgbs_dir if a2h_wgbs_dir is not None else cfg["a2h_wgbs_dir"]
        wgbs_input = root / input_dir / f"{prefix}.traditional_probe_matrix.tsv"
        if not wgbs_input.exists():
            raise FileNotFoundError(f"Missing WGBS traditional probe matrix: {wgbs_input}")
        generate_outputs_for_input(
            python_bin=python_bin,
            a2h_root=a2h_root,
            input_path=wgbs_input,
            output_dir=root / output_dir,
            prefix=output_name,
            platform=platform,
            threads=threads,
            rename_columns=None,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--namespace",
        required=True,
        choices=sorted(NAMESPACE_CONFIG),
    )
    parser.add_argument("--mode", choices=["array", "wgbs", "both"], default="both")
    parser.add_argument("--study")
    parser.add_argument("--platform", choices=["450K", "EPIC"])
    parser.add_argument("--a2h-root", default=str(DEFAULT_A2H_ROOT))
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    parser.add_argument("--threads", type=int, default=detect_threads())
    parser.add_argument(
        "--wgbs-input-dir",
        type=Path,
        help="Override the WGBS probe-space matrix directory for --mode wgbs or both.",
    )
    parser.add_argument(
        "--a2h-wgbs-dir",
        type=Path,
        help="Override the A2H output directory generated from WGBS traditional beta.",
    )
    parser.add_argument(
        "--input-prefix",
        help="Override input matrix prefix. Defaults to <study>_<platform>.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Override A2H output file prefix. Defaults to the input prefix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    python_bin = Path(args.python_bin)
    a2h_root = Path(args.a2h_root)
    groups = load_groups(root, args.namespace, args.study, args.platform)
    for study, platform in groups:
        process_group(
            root=root,
            namespace=args.namespace,
            study=study,
            platform=platform,
            mode=args.mode,
            python_bin=python_bin,
            a2h_root=a2h_root,
            threads=args.threads,
            wgbs_input_dir=args.wgbs_input_dir,
            a2h_wgbs_dir=args.a2h_wgbs_dir,
            input_prefix=args.input_prefix,
            output_prefix=args.output_prefix,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
