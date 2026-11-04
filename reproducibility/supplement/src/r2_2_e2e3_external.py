"""R2-2 external arms for E2/E3 on the locked PGP-UK set (E1 external tables).

``refit`` (option A): fit A2H on the 39 E2/E3 representatives from one
condition's calls (native, an E2 seed/depth, or an E3 raw length) with a
lowered ``min_train`` (sensitivity arm only, not production), predict the
fixed PGP-UK array betas and keep the per-site errors and predictions.

``compare``: paired difference in mean per-CpG RMSE between a condition fit
and the native-39 fit on CpGs predicted by both, with a bootstrap over the 10
PGP-UK participants.  The deployed 205-sample A2H on the same CpGs is shown
for scale.

``pgpuk-c`` (option C): the deployed A2H predictions scored against PGP-UK
WGBS truth recomputed from read-length-capped calls (same depth >= 10 rule),
versus the locked native truth, on participant-CpG pairs valid in both.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.r2_2_common import fit_compatible, site_metrics


TARGETS = ("CHALM", "CAMDA")
FITTED = ("quadratic", "constant_kde", "quadratic_rank_warning")


def external_table(external_dir: Path, target: str) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_parquet(external_dir / f"external_{target}.parquet")
    samples = [c[len("m__"):] for c in table.columns if c.startswith("m__")]
    return table, samples


def reference_matrices(epic_dir: Path, experiment: str, condition: str, samples: list[str],
                       loci: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Return x = m and y = CHALM (site x sample) for the reference samples."""

    xs, ys = [], []
    for sample in samples:
        frame = pd.read_parquet(epic_dir / f"{experiment}__{sample}__{condition}.parquet").set_index("locus")
        frame = frame.reindex(loci)
        xs.append(frame["m"].to_numpy(dtype=float))
        ys.append(frame["chalm"].to_numpy(dtype=float))
    return np.column_stack(xs), np.column_stack(ys)


def predict(quality: pd.DataFrame, m: np.ndarray) -> np.ndarray:
    fitted = quality["fit_status"].isin(FITTED).to_numpy()[:, None]
    pred = (quality["a2h_intercept"].to_numpy()[:, None] + quality["a2h_linear"].to_numpy()[:, None] * m
            + quality["a2h_quadratic"].to_numpy()[:, None] * m ** 2)
    return np.where(fitted, pred, np.nan)


def refit(args: argparse.Namespace) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    samples = pd.read_csv(args.samples, sep="\t")["sample_id"].tolist()
    summary = {"experiment": args.experiment, "condition": args.condition, "n_reference": len(samples),
               "min_train": args.min_train, "reference_samples": samples, "targets": {},
               "note": "sensitivity refit on E2/E3 representatives; not the production A2H"}
    for target in TARGETS:
        ext, ext_samples = external_table(args.external_dir, target)
        loci = pd.Index(ext["locus"])
        x, chalm = reference_matrices(args.epic_dir, args.experiment, args.condition, samples, loci)
        y = chalm - x if target == "CAMDA" else chalm
        quality = fit_compatible(x, y, min_train=args.min_train)
        pred = predict(quality, ext[[f"m__{s}" for s in ext_samples]].to_numpy(dtype=float))
        observed = ext[[f"observed__{s}" for s in ext_samples]].to_numpy(dtype=float)
        metrics = site_metrics(observed, pred)
        table = ext[["probe_id", "locus"]].copy()
        table["fit_status"] = quality["fit_status"].to_numpy()
        table["n_train_joint"] = quality["n_train_joint"].to_numpy()
        for i, sample in enumerate(ext_samples):
            table[f"pred__{sample}"] = pred[:, i]
        table["rmse"] = metrics["rmse"].to_numpy()
        table["n_external"] = metrics["n"].to_numpy()
        table.to_parquet(out / f"{target}_predictions.parquet", index=False)
        valid = np.isfinite(table["rmse"])
        summary["targets"][target] = {
            "n_external_probes": len(table), "n_predicted_probes": int(valid.sum()),
            "fit_status_counts": table["fit_status"].value_counts().to_dict(),
            "mean_site_rmse_all_predicted": float(table.loc[valid, "rmse"].mean()),
        }
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["code_sha256"] = file_hash(Path(__file__))
    (out / "refit.json").write_text(json.dumps(summary, indent=2) + "\n")


def mean_site_rmse(observed: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(pred)
    n = valid.sum(axis=1)
    sse = np.where(valid, (pred - observed) ** 2, 0.0).sum(axis=1)
    keep = n > 0
    return float(np.sqrt(sse[keep] / n[keep]).mean())


def paired_bootstrap(observed: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_boot: int, seed: int) -> dict[str, float]:
    """Mean per-site RMSE of A and B and A - B; resample participants (columns)."""

    common = np.isfinite(observed) & np.isfinite(pred_a) & np.isfinite(pred_b)
    obs = np.where(common, observed, np.nan)
    a, b = np.where(common, pred_a, np.nan), np.where(common, pred_b, np.nan)
    estimate = mean_site_rmse(obs, a) - mean_site_rmse(obs, b)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        cols = rng.integers(0, obs.shape[1], obs.shape[1])
        draws[i] = mean_site_rmse(obs[:, cols], a[:, cols]) - mean_site_rmse(obs[:, cols], b[:, cols])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"n_sites": int(common.any(axis=1).sum()), "n_pairs": int(common.sum()),
            "mean_site_rmse_a": mean_site_rmse(obs, a), "mean_site_rmse_b": mean_site_rmse(obs, b),
            "difference_a_minus_b": estimate, "ci95_low": float(low), "ci95_high": float(high),
            "n_boot": n_boot, "bootstrap_unit": "PGP-UK participant"}


def compare(args: argparse.Namespace) -> None:
    rows = []
    for target in TARGETS:
        ext, samples = external_table(args.external_dir, target)
        observed = ext[[f"observed__{s}" for s in samples]].to_numpy(dtype=float)
        deployed = ext[[f"baseline__{s}" for s in samples]].to_numpy(dtype=float)
        native = pd.read_parquet(args.native_refit / f"{target}_predictions.parquet")
        assert (native["probe_id"].to_numpy() == ext["probe_id"].to_numpy()).all()
        native_pred = native[[f"pred__{s}" for s in samples]].to_numpy(dtype=float)
        for refit_dir in args.condition_refits:
            cond = pd.read_parquet(refit_dir / f"{target}_predictions.parquet")
            assert (cond["probe_id"].to_numpy() == ext["probe_id"].to_numpy()).all()
            cond_pred = cond[[f"pred__{s}" for s in samples]].to_numpy(dtype=float)
            both = np.isfinite(native_pred) & np.isfinite(cond_pred)
            result = paired_bootstrap(observed, cond_pred, native_pred, args.n_boot, args.seed)
            scale = paired_bootstrap(np.where(both, observed, np.nan), native_pred, deployed, args.n_boot, args.seed)
            rows.append({"target": target, "condition": refit_dir.name, **result,
                         "native39_minus_deployed205": scale["difference_a_minus_b"],
                         "native39_minus_deployed205_ci95_low": scale["ci95_low"],
                         "native39_minus_deployed205_ci95_high": scale["ci95_high"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)


def capped_truth(epic_dir: Path, condition: str, ext: pd.DataFrame, wgbs_ids: list[str], target: str,
                 min_depth: int) -> np.ndarray:
    columns = []
    for wgbs_id in wgbs_ids:
        frame = pd.read_parquet(epic_dir / f"pgpuk_c__{wgbs_id}__{condition}.parquet").set_index("locus")
        frame = frame.reindex(ext["locus"])
        value = frame["chalm"] - frame["m"] if target == "CAMDA" else frame["chalm"]
        columns.append(np.where(frame["ct"].to_numpy() >= min_depth, value.to_numpy(dtype=float), np.nan))
    return np.column_stack(columns)


def pgpuk_c(args: argparse.Namespace) -> None:
    pairs = pd.read_csv(args.pairs, sep="\t")
    pairs = pairs.loc[(pairs.study == "pgpuk") & (pairs.tissue == "blood")].set_index("array_id")
    rows, checks = [], {}
    for target in TARGETS:
        ext, samples = external_table(args.external_dir, target)
        wgbs_ids = pairs.loc[samples, "wgbs_id"].tolist()
        observed = ext[[f"observed__{s}" for s in samples]].to_numpy(dtype=float)
        deployed = ext[[f"baseline__{s}" for s in samples]].to_numpy(dtype=float)
        native = capped_truth(args.epic_dir, "native", ext, wgbs_ids, target, args.min_depth)
        both = np.isfinite(observed) & np.isfinite(native)
        checks[target] = {"locked_pairs": int(np.isfinite(observed).sum()),
                          "recomputed_native_pairs_on_locked": int(both.sum()),
                          "max_abs_diff_locked_vs_recomputed": float(np.nanmax(np.abs(observed - native)[both]))}
        for condition in args.conditions:
            capped = capped_truth(args.epic_dir, condition, ext, wgbs_ids, target, args.min_depth)
            keep = np.isfinite(observed) & np.isfinite(capped) & np.isfinite(deployed)
            # A = deployed vs capped truth, B = deployed vs locked native truth, same pairs.
            # B is passed as deployed - observed + capped so that B - capped = deployed - observed.
            result = paired_bootstrap(np.where(keep, capped, np.nan), deployed,
                                      np.where(keep, deployed - observed + capped, np.nan), args.n_boot, args.seed)
            truth_shift = (capped - observed)[keep]
            rows.append({"target": target, "condition": condition, **result,
                         "locked_pairs_retained": int(keep.sum()) / int(np.isfinite(observed).sum()),
                         "mean_truth_shift_capped_minus_native": float(truth_shift.mean())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    args.output.with_suffix(".checks.json").write_text(json.dumps(checks, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("refit")
    p.add_argument("--epic-dir", type=Path, required=True)
    p.add_argument("--experiment", required=True, help="e2 or e3 (native files exist under both)")
    p.add_argument("--condition", required=True)
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--min-train", type=int, required=True)
    p.add_argument("--external-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=refit)
    c = sub.add_parser("compare")
    c.add_argument("--external-dir", type=Path, required=True)
    c.add_argument("--native-refit", type=Path, required=True)
    c.add_argument("--condition-refits", type=Path, nargs="+", required=True)
    c.add_argument("--n-boot", type=int, default=2000)
    c.add_argument("--seed", type=int, default=20261015)
    c.add_argument("--output", type=Path, required=True)
    c.set_defaults(func=compare)
    g = sub.add_parser("pgpuk-c")
    g.add_argument("--epic-dir", type=Path, required=True)
    g.add_argument("--external-dir", type=Path, required=True)
    g.add_argument("--pairs", type=Path, required=True)
    g.add_argument("--conditions", nargs="+", default=["raw125", "raw100", "raw75"])
    g.add_argument("--min-depth", type=int, default=10)
    g.add_argument("--n-boot", type=int, default=2000)
    g.add_argument("--seed", type=int, default=20261015)
    g.add_argument("--output", type=Path, required=True)
    g.set_defaults(func=pgpuk_c)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
