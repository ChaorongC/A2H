"""Summarize E2/E3/C perturbed calls against the same sample's native calls.

For each completed condition directory (``run_metadata.json`` present) the
methRatio and CHALM tables are restricted to the audited EPIC universe and
written as a compact per-condition EPIC table (``locus, m, chalm, ct``) for the
option-A refit.  Paired deltas are computed on loci callable (CT >= 4) in BOTH
the condition and the native run; callability loss is reported separately.
CAMDA = CHALM - m on the same keys.  Missing values stay missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.r2_2_e2_downsample import epic_loci


def epic_calls(directory: Path, universe: pd.Index, files=("methRatio.txt", "CHALM.txt")) -> pd.DataFrame:
    """Return EPIC-locus m, CHALM and CT_count for one caller output directory."""

    parts = []
    for name, column in zip(files, ("m", "chalm")):
        table = pd.read_csv(directory / name, sep="\t", usecols=["chr", "pos", "ratio", "CT_count"],
                            dtype={"chr": "category", "pos": np.int64, "ratio": np.float64,
                                   "CT_count": np.int64})
        table.index = table["chr"].astype(str) + "_" + table["pos"].astype(str)
        table = table[table.index.isin(universe)]
        if table.index.duplicated().any():
            raise ValueError(f"Duplicate EPIC loci in {directory / name}")
        parts.append(table[["ratio", "CT_count"]].rename(columns={"ratio": column, "CT_count": f"ct_{column}"}))
    frame = parts[0].join(parts[1], how="outer")
    both = frame["ct_m"].notna() & frame["ct_chalm"].notna()
    if not (frame.loc[both, "ct_m"] == frame.loc[both, "ct_chalm"]).all():
        raise ValueError(f"methRatio and CHALM CT_count disagree in {directory}")
    frame["ct"] = frame["ct_m"].fillna(frame["ct_chalm"])
    frame.index.name = "locus"
    return frame[["m", "chalm", "ct"]]


def censored_median(ct: pd.Series, n_universe: int) -> float:
    """Median depth over the universe; absent loci are below 4 reads."""

    values = ct.dropna().to_numpy()
    n_absent = n_universe - len(values)
    if 2 * n_absent >= n_universe:
        return float("nan")
    return float(np.median(np.concatenate([values, np.zeros(n_absent)])))


def paired_metrics(native: pd.DataFrame, condition: pd.DataFrame, n_universe: int) -> list[dict]:
    rows = []
    for target in ("m", "chalm", "camda"):
        nat = native["chalm"] - native["m"] if target == "camda" else native[target]
        cond = condition["chalm"] - condition["m"] if target == "camda" else condition[target]
        nat, cond = nat.dropna(), cond.dropna()
        common = nat.index.intersection(cond.index)
        delta = (cond.loc[common] - nat.loc[common]).to_numpy()
        absolute = np.abs(delta)
        rows.append({
            "target": target,
            "n_universe": n_universe,
            "n_native_callable": len(nat),
            "n_condition_callable": len(cond),
            "n_matched": len(common),
            "native_retention": len(common) / len(nat) if len(nat) else np.nan,
            "n_condition_only": len(cond.index.difference(nat.index)),
            "mean_signed_delta": float(delta.mean()) if len(delta) else np.nan,
            "median_signed_delta": float(np.median(delta)) if len(delta) else np.nan,
            "mae": float(absolute.mean()) if len(delta) else np.nan,
            "rmse": float(np.sqrt((delta ** 2).mean())) if len(delta) else np.nan,
            "abs_delta_q50": float(np.quantile(absolute, 0.5)) if len(delta) else np.nan,
            "abs_delta_q90": float(np.quantile(absolute, 0.9)) if len(delta) else np.nan,
            "abs_delta_q99": float(np.quantile(absolute, 0.99)) if len(delta) else np.nan,
            "pearson_r_secondary": float(np.corrcoef(cond.loc[common], nat.loc[common])[0, 1])
            if len(common) > 2 else np.nan,
        })
    return rows


def condition_dirs(root: Path) -> list[tuple[str, str, str, Path]]:
    """Return (experiment, sample, condition, path) for completed condition runs."""

    found = []
    for experiment in ("e2", "e3", "pgpuk_c"):
        for meta in sorted((root / experiment).glob("*/*/run_metadata.json")):
            status = json.loads(meta.read_text()).get("status", "complete")
            if status == "complete":
                found.append((experiment, meta.parent.parent.name, meta.parent.name, meta.parent))
    return found


def native_dir_for(experiment: str, sample: str, native_root: Path, pgpuk_root: Path) -> tuple[Path, tuple[str, str]]:
    if experiment == "pgpuk_c":
        return pgpuk_root, (f"traditional/{sample}_methRatio.txt", f"chalm/{sample}_CHALM.txt")
    return native_root / f"native_{sample}", ("methRatio.txt", "CHALM.txt")


def run(args: argparse.Namespace) -> None:
    universe = epic_loci(args.epic_mapping)
    epic_dir = args.root / "epic"
    epic_dir.mkdir(exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    natives: dict[tuple[str, str], pd.DataFrame] = {}
    records, depth_rows = [], []
    for experiment, sample, condition, path in condition_dirs(args.root):
        key = (experiment, sample)
        if key not in natives:
            native_path, files = native_dir_for(experiment, sample, args.native_root, args.pgpuk_wgbs_root)
            natives[key] = epic_calls(native_path, universe, files)
            native_out = epic_dir / f"{experiment}__{sample}__native.parquet"
            if not native_out.exists():
                natives[key].reset_index().to_parquet(native_out, index=False)
        out = epic_dir / f"{experiment}__{sample}__{condition}.parquet"
        frame = epic_calls(path, universe)
        if not out.exists():
            frame.reset_index().to_parquet(out, index=False)
        for row in paired_metrics(natives[key], frame, len(universe)):
            records.append({"experiment": experiment, "sample_id": sample, "condition": condition, **row})
        depth_rows.append({"experiment": experiment, "sample_id": sample, "condition": condition,
                           "native_median_ct": censored_median(natives[key]["ct"], len(universe)),
                           "condition_median_ct": censored_median(frame["ct"], len(universe)),
                           "native_mean_ct_present": float(natives[key]["ct"].mean()),
                           "condition_mean_ct_present": float(frame["ct"].mean())})
    pd.DataFrame(records).to_csv(args.output_dir / "paired_epic_deltas.tsv", sep="\t", index=False)
    pd.DataFrame(depth_rows).to_csv(args.output_dir / "realized_epic_depth.tsv", sep="\t", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="E2/E3/C output root on DFS3b")
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--pgpuk-wgbs-root", type=Path, required=True,
                        help="PGP-UK results/paired_validation_v2/wgbs directory")
    parser.add_argument("--epic-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
