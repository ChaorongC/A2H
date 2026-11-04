#!/usr/bin/env python
"""Build paired-validation-v2 WGBS probe-space matrices in array sample columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("external_validation")
DEFAULT_CHALM_ARRAY_ROOT = Path("legacy_model")


def beadchip_probe_column(platform: str, reference: str) -> str:
    platform_upper = platform.upper()
    if platform_upper in {"450K", "HM450"}:
        return f"HM450.{reference}"
    if platform_upper in {"EPIC", "850K"}:
        return f"EPIC.{reference}"
    raise ValueError(f"Unsupported platform: {platform}")


def load_probe_map(path: Path, platform: str, reference: str) -> pd.DataFrame:
    probe_col = beadchip_probe_column(platform, reference)
    return (
        pd.read_table(path, usecols=["loci.hg38", probe_col])
        .dropna()
        .drop_duplicates()
        .rename(columns={probe_col: "Probe_ID"})
    )


def resolve_probe_map_path(root: Path, reference: str) -> Path:
    candidates = [
        root / f"CHALM-array_cpg_probe.{reference}.tsv",
        root / f"A2H_cpg_probe.{reference}.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Missing probe map: " + ", ".join(str(candidate) for candidate in candidates))


def load_ratio_series(path: Path, probe_map: pd.DataFrame, sample_name: str) -> pd.Series:
    table = pd.read_csv(path, sep="\t", usecols=["chr", "pos", "context", "ratio"])
    table = table[table["context"] == "CG"].copy()
    table["loci.hg38"] = table["chr"].astype(str) + "_" + table["pos"].astype(str)
    matched = (
        table[["loci.hg38", "ratio"]]
        .merge(probe_map, on="loci.hg38", how="inner")
        .drop_duplicates(subset=["Probe_ID"])
        .set_index("Probe_ID")["ratio"]
        .rename(sample_name)
    )
    return matched


def split_wgbs_ids(wgbs_id: str) -> list[str]:
    return [item for item in wgbs_id.split(";") if item]


def unique_wgbs_ids(wgbs_ids: list[str]) -> list[str]:
    return list(dict.fromkeys(wgbs_ids))


def wgbs_ids_for_row(row: object, paired_samples: pd.DataFrame) -> list[str]:
    row_wgbs_ids = split_wgbs_ids(row.wgbs_id)
    if getattr(row, "pairing_level") != "cell_type":
        return row_wgbs_ids
    matched = paired_samples[
        (paired_samples["pairing_level"] == "cell_type")
        & (paired_samples["subject_or_cell"] == row.subject_or_cell)
    ]
    pooled_ids: list[str] = []
    for wgbs_id in matched["wgbs_id"]:
        pooled_ids.extend(split_wgbs_ids(wgbs_id))
    return unique_wgbs_ids(pooled_ids)


def build_array_series(
    wgbs_ids: list[str],
    array_id: str,
    probe_map: pd.DataFrame,
    input_dir: Path,
    suffix: str,
) -> pd.Series:
    series = []
    for wgbs_id in wgbs_ids:
        input_path = input_dir / f"{wgbs_id}_{suffix}.txt"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing WGBS input: {input_path}")
        series.append(load_ratio_series(input_path, probe_map, wgbs_id))
    if len(series) == 1:
        return series[0].rename(array_id)
    pooled = pd.concat(series, axis=1, sort=True).mean(axis=1, skipna=True)
    return pooled.rename(array_id)


def build_matrix(
    paired_samples: pd.DataFrame,
    probe_map: pd.DataFrame,
    input_dir: Path,
    suffix: str,
) -> pd.DataFrame:
    series = []
    for row in paired_samples.itertuples(index=False):
        wgbs_ids = wgbs_ids_for_row(row, paired_samples)
        if not wgbs_ids:
            raise ValueError(f"No WGBS IDs listed for array {row.array_id}")
        series.append(build_array_series(wgbs_ids, row.array_id, probe_map, input_dir, suffix))
    matrix = pd.concat(series, axis=1, sort=True).reset_index()
    matrix = matrix.rename(columns={"index": "Probe_ID"})
    matrix = matrix.loc[matrix.iloc[:, 1:].notna().any(axis=1)].copy()
    return matrix.sort_values("Probe_ID").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--paired-samples", default="metadata/paired_validation_v2/paired_samples.tsv")
    parser.add_argument("--chalm-array-root", default=str(DEFAULT_CHALM_ARRAY_ROOT))
    parser.add_argument("--study", required=True)
    parser.add_argument("--platform", required=True, choices=["450K", "EPIC", "HM450", "850K"])
    parser.add_argument("--ref", default="hg38", choices=["hg38"])
    parser.add_argument("--output-dir", default="results/paired_validation_v2/wgbs/probe_space")
    parser.add_argument(
        "--traditional-dir",
        default="results/paired_validation_v2/wgbs/traditional",
    )
    parser.add_argument("--chalm-dir", default="results/paired_validation_v2/wgbs/chalm")
    parser.add_argument(
        "--output-prefix",
        help="Override output matrix prefix. Defaults to <study>_<platform>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    paired = pd.read_table(root / args.paired_samples)
    paired = paired[(paired["study"] == args.study) & (paired["platform"] == args.platform)]
    if paired.empty:
        raise ValueError(f"No paired samples matched {args.study} {args.platform}")

    probe_map_path = resolve_probe_map_path(Path(args.chalm_array_root), args.ref)
    probe_map = load_probe_map(probe_map_path, args.platform, args.ref)
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    trad_matrix = build_matrix(
        paired,
        probe_map,
        root / args.traditional_dir,
        "methRatio",
    )
    chalm_matrix = build_matrix(
        paired,
        probe_map,
        root / args.chalm_dir,
        "CHALM",
    )
    prefix = args.output_prefix or f"{args.study}_{args.platform}"
    trad_path = output_dir / f"{prefix}.traditional_probe_matrix.tsv"
    chalm_path = output_dir / f"{prefix}.chalm_probe_matrix.tsv"
    camda_path = output_dir / f"{prefix}.camda_probe_matrix.tsv"
    camda_matrix = chalm_matrix.copy()
    camda_matrix.iloc[:, 1:] = chalm_matrix.iloc[:, 1:] - trad_matrix.iloc[:, 1:]
    trad_matrix.to_csv(trad_path, sep="\t", index=False)
    chalm_matrix.to_csv(chalm_path, sep="\t", index=False)
    camda_matrix.to_csv(camda_path, sep="\t", index=False)
    print(f"Wrote {trad_path}")
    print(f"Wrote {chalm_path}")
    print(f"Wrote {camda_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
