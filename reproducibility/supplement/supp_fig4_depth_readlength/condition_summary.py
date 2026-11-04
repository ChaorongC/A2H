"""Across-sample summary of summary_20261016/paired_epic_deltas.tsv (descriptive).

Unit = sample (39 atlas representatives; 10 PGP-UK participants for C). E2 seeds are first
averaged within sample. Primary per the pre-fixed criteria (R2-2_PLAN_amendment_20261014.md):
mean signed shift vs native and callability loss (1 - native_retention); RMSE vs native is
reported, not a pass/fail measure. 95% intervals: percentile bootstrap over samples.
Run from reproducibility/supplement/: python supp_fig4_depth_readlength/condition_summary.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("results/r2_2_e2e3_20261015/summary_20261016")
N_BOOT, SEED = 2000, 20261016


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    deltas = pd.read_csv(HERE / "paired_epic_deltas.tsv", sep="\t")
    depth = pd.read_csv(HERE / "realized_epic_depth.tsv", sep="\t")
    deltas = deltas.merge(depth, on=["experiment", "sample_id", "condition"])
    deltas["level"] = deltas["condition"].str.replace(r"^seed\d+_", "", regex=True)
    deltas["callability_loss"] = 1 - deltas["native_retention"]
    metrics = ["mean_signed_delta", "callability_loss", "rmse", "mae", "condition_median_ct", "native_median_ct"]
    per_sample = deltas.groupby(["experiment", "level", "target", "sample_id"])[metrics].mean().reset_index()
    rng = np.random.default_rng(SEED)
    rows = []
    for (experiment, level, target), g in per_sample.groupby(["experiment", "level", "target"]):
        shift = g["mean_signed_delta"].to_numpy()
        low, high = boot_ci(shift, rng)
        rows.append({"experiment": experiment, "level": level, "target": target, "n_samples": len(g),
                     "mean_shift": shift.mean(), "mean_shift_ci95_low": low, "mean_shift_ci95_high": high,
                     "shift_min": shift.min(), "shift_max": shift.max(),
                     "callability_loss_median": g["callability_loss"].median(),
                     "callability_loss_max": g["callability_loss"].max(),
                     "rmse_mean": g["rmse"].mean(), "mae_mean": g["mae"].mean(),
                     "condition_median_ct_median": g["condition_median_ct"].median(),
                     "native_median_ct_median": g["native_median_ct"].median()})
    out = pd.DataFrame(rows)
    order = {"depth20": 0, "depth10": 1, "depth5": 2, "raw125": 3, "raw100": 4, "raw75": 5}
    out = out.sort_values(["experiment", "target", "level"], key=lambda s: s.map(order) if s.name == "level" else s)
    out.to_csv(HERE / "condition_summary.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
