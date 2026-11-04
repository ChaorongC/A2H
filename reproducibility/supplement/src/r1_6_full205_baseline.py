"""R1-6 full-reference baseline on the cell-type held-out samples.

The tissue-only holdout scores each held-out class (44 blood/immune, 83 epithelial
samples) with a model trained without that class. Here the same held-out observations
are scored by models whose reference keeps all 205 atlas samples:
  - donor_cv / sample_cv: out-of-fold predictions from the frozen 10x10 donor-grouped
    and sample-level atlas CV (read, not refit); per-CpG RMSE per repeat, mean of 10;
  - insample: one fit on all 205 samples (same fitter), so the scored sample is in
    training (optimistic).
Observations, CpGs (>=8 held-out observations) and the RMSE formula are those of the
published holdout. A CpG gets full-205 values only when every published observation has
a prediction in every repeat and model; otherwise they stay missing.
Uncertainty: donor bootstrap of the held-out samples (fixed predictions).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash, row_stats
from src.fit_atlas_fold import fit_sites
from src.run_atlas_cv_split import shard_arrays
from src.tissue_only_transfer import predict_coefficients

HOLDOUT = Path("results/tissue_only_r1_5_r1_6_20261008")
CV = Path("results/atlas_cv_full/original")
SPLITS = Path("results/atlas_splits/splits.tsv")
FULL_CACHE = Path("results/transfer_full/full_reference")
DEFAULT_OUT = Path("results/r1_6_full205_baseline_20261019")
ARMS = ("exclude_blood_immune", "exclude_epithelial")
TARGETS = ("CHALM", "CAMDA")
METHODS = {"A2H": "A2H", "linear": "reference_linear"}
BRANCHES = {"donor_cv": "donor_grouped", "sample_cv": "historical_sample"}
N_BOOT, SEED = 1000, 2026
FULL_COLUMNS = [f"{m}_full205_{v}_rmse" for v in ("donor_cv", "sample_cv", "insample") for m in METHODS]
COLUMNS = ["A2H_rmse", "linear_rmse"] + FULL_COLUMNS


def donor_weights(donors, n_boot=N_BOOT, seed=SEED):
    """Sample x draw multiplicities from resampling donors with replacement."""
    donors = np.asarray(donors)
    unique, index = np.unique(donors, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    counts = np.stack([np.bincount(d, minlength=len(unique)) for d in draws], axis=1)
    return counts[index].astype(float)


def boot_rmse_sum(err2, valid, weights):
    """Per draw: sum over CpGs of weighted RMSE, and number of CpGs with weight."""
    den = valid.astype(float) @ weights
    num = np.where(valid, err2, 0.0) @ weights
    rmse = np.sqrt(np.divide(num, den, out=np.zeros_like(num), where=den > 0))
    return rmse.sum(axis=0), (den > 0).sum(axis=0)


def read_cv(branch, target, shard_index, loci, ids, splits, truth):
    position = {s: i for i, s in enumerate(ids)}
    keep = set(loci)
    pred = {m: np.full((10, len(loci), len(ids)), np.nan) for m in METHODS.values()}
    for repeat in range(10):
        filled = np.zeros(len(ids), dtype=bool)
        for fold in range(10):
            split = repeat * 10 + fold
            part = splits.loc[(splits.branch == branch) & (splits.split_index == split)]
            test = part.loc[part.role == "test", "sample_id"].tolist()
            data = pd.read_parquet(CV / f"{branch}_split{split:03d}" / f"{target}_shard{shard_index:04d}_predictions.parquet",
                                   columns=["locus", "sample_id", "repeat", "fold", "observed", *METHODS.values()])
            data = data.loc[data.locus.isin(keep)]
            if (not np.array_equal(data.locus.to_numpy(), np.repeat(loci, len(test))) or
                    not np.array_equal(data.sample_id.to_numpy(), np.tile(test, len(loci))) or
                    not data.repeat.eq(repeat).all() or not data.fold.eq(fold).all()):
                raise ValueError(f"CV key mismatch: {branch} split {split} shard {shard_index}")
            cols = [position[s] for s in test]
            if filled[cols].any():
                raise ValueError("Sample held out twice within a repeat")
            observed = data.observed.to_numpy(float).reshape(len(loci), len(test))
            if not np.array_equal(observed, truth[:, cols], equal_nan=True):
                raise ValueError("CV truth differs from the atlas shard")
            for method in METHODS.values():
                pred[method][repeat][:, cols] = data[method].to_numpy(float).reshape(len(loci), len(test))
            filled[cols] = True
        if not filled.all():
            raise ValueError(f"Repeat {repeat} does not cover all samples")
    return pred


def run_shard(out, shard_index):
    path = out / "shards" / f"shard_{shard_index:04d}.parquet"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(HOLDOUT / "atlas_samples.tsv", sep="\t")
    ids = samples.sample_id.tolist()
    position = {s: i for i, s in enumerate(ids)}
    membership = pd.read_csv(HOLDOUT / "reference_membership.tsv", sep="\t")
    splits = pd.read_csv(SPLITS, sep="\t")
    row = pd.read_csv(HOLDOUT / "atlas_shards.tsv", sep="\t").set_index("shard_index").loc[shard_index]
    if file_hash(Path(row.path)) != row.sha256:
        raise ValueError("Atlas shard hash changed")
    mapping = pd.read_csv(HOLDOUT / "epic_mapping.tsv", sep="\t", usecols=["probe_id", "locus", "shard_index"])
    mapping = mapping.loc[mapping.shard_index == shard_index]
    frame = pd.read_parquet(row.path)
    frame = frame.loc[frame.locus.isin(mapping.locus)].reset_index(drop=True)
    loci = frame.locus.to_numpy()
    tables, boots, checks = [], [], []
    for target in TARGETS:
        truth = frame[[f"{target}__{s}" for s in ids]].to_numpy(float)
        x, y, _, _ = shard_arrays(frame, target, ids, [])
        quality, insample = fit_sites(x, y, x, min_train=50, include_coefficients=True)
        quality.insert(0, "locus", loci)
        # Known answer: the full fit reproduces the cached full-reference PGP-UK predictions.
        ext = pd.read_parquet(FULL_CACHE / f"shard_{shard_index:04d}" / f"{target}_external_predictions.parquet")
        ext = ext.loc[ext.probe_id.isin(mapping.probe_id)]
        if ext.empty:
            raise ValueError("No cached full-reference rows to check")
        ext_locus = ext.probe_id.map(mapping.set_index("probe_id").locus)
        expred = predict_coefficients(quality.set_index("locus").loc[ext_locus], ext.m.to_numpy()[:, None])
        for label, method in METHODS.items():
            np.testing.assert_allclose(expred[method][:, 0], ext[f"transfer_{method}"], atol=2e-10, rtol=2e-10, equal_nan=True)
        checks.append({"target": target, "check": "insample fit reproduces cached full-reference PGP-UK predictions",
                       "n_rows": len(ext), "pass": True})
        cv = {}
        for label, branch in BRANCHES.items():
            cv[label] = read_cv(branch, target, shard_index, loci, ids, splits, truth)
            # Known answer: all-sample per-repeat site RMSE equals the verified CV summary,
            # which scores every method on observations where A2H and linear are both finite.
            for repeat in range(10):
                joint = np.isfinite(cv[label]["A2H"][repeat]) & np.isfinite(cv[label]["reference_linear"][repeat])
                for method in METHODS.values():
                    pub = pd.read_parquet(CV / f"{branch}_summary" / f"{target}_repeat{repeat:02d}_shard{shard_index:04d}_{method}.parquet",
                                          columns=["locus", "n", "rmse"]).set_index("locus").loc[loci]
                    stats = row_stats(np.where(joint, truth, np.nan), cv[label][method][repeat])
                    if not np.array_equal(stats["n"], pub.n.to_numpy()):
                        raise ValueError(f"CV n mismatch: {branch} {target} repeat {repeat} {method}")
                    np.testing.assert_allclose(stats["rmse"], pub.rmse.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True)
            checks.append({"target": target, "check": f"{branch} all-sample site RMSE equals CV summary (10 repeats x 2 methods)",
                           "n_rows": len(loci), "pass": True})
        for arm_index, arm in enumerate(ARMS):
            part = membership.loc[(membership.model == arm) & (membership.role == "heldout")]
            test = part.sample_id.tolist()
            cols = [position[s] for s in test]
            base = HOLDOUT / "fits" / arm / f"shard_{shard_index:04d}"
            h = pd.read_parquet(base / f"{target}_heldout_predictions.parquet")
            if (not np.array_equal(h.locus.to_numpy(), np.repeat(loci, len(test))) or
                    not np.array_equal(h.sample_id.to_numpy(), np.tile(test, len(loci)))):
                raise ValueError(f"Holdout key mismatch: {arm} {target}")
            shape = (len(loci), len(test))
            obs = h.observed.to_numpy(float).reshape(shape)
            if not np.array_equal(obs, truth[:, cols], equal_nan=True):
                raise ValueError("Holdout truth differs from the atlas shard")
            valid = h.evaluation_valid.to_numpy(bool).reshape(shape)
            preds = {"A2H_rmse": [h.A2H.to_numpy(float).reshape(shape)],
                     "linear_rmse": [h.reference_linear.to_numpy(float).reshape(shape)]}
            for variant in BRANCHES:
                for label, method in METHODS.items():
                    preds[f"{label}_full205_{variant}_rmse"] = [cv[variant][method][r][:, cols] for r in range(10)]
            for label, method in METHODS.items():
                preds[f"{label}_full205_insample_rmse"] = [insample[method][:, cols]]
            site = pd.read_parquet(base / f"{target}_heldout_site_metrics.parquet").set_index(["method", "locus"])
            table = pd.DataFrame({"heldout_arm": arm, "target": target, "locus": loci})
            masked_obs = np.where(valid, obs, np.nan)
            complete = np.ones(len(loci), dtype=bool)
            for column in FULL_COLUMNS:
                for p in preds[column]:
                    complete &= ~(valid & ~np.isfinite(p)).any(axis=1)
            for column in COLUMNS:
                rmse = np.mean([row_stats(masked_obs, np.where(valid, p, np.nan))["rmse"] for p in preds[column]], axis=0)
                if column in FULL_COLUMNS:
                    rmse[~complete] = np.nan
                table[column] = rmse
            # Known answer: recomputed holdout columns equal the published site metrics.
            for column, method in (("A2H_rmse", "A2H"), ("linear_rmse", "reference_linear")):
                pub = site.loc[method].loc[loci]
                np.testing.assert_allclose(table[column].to_numpy(), pub.rmse.to_numpy(), rtol=1e-12, atol=0, equal_nan=True)
            n = valid.sum(axis=1)
            if not np.array_equal(n, site.loc["A2H"].loc[loci].n.to_numpy()):
                raise ValueError("Held-out observation counts differ from published")
            table.insert(3, "n_heldout_obs", n)
            table.insert(4, "eligible", site.loc["A2H"].loc[loci].eligible.to_numpy())
            table.insert(5, "full205_complete", complete)
            tables.append(table)
            keep = table.eligible.to_numpy() & complete
            weights = donor_weights(part.donor_id.to_numpy(), seed=[SEED, arm_index])
            for column in COLUMNS:
                total = np.zeros(N_BOOT)
                for p in preds[column]:
                    s, count = boot_rmse_sum(((p - obs) ** 2)[keep], valid[keep], weights)
                    total += s / len(preds[column])
                boots.append(pd.DataFrame({"heldout_arm": arm, "target": target, "column": column,
                                           "draw": np.arange(N_BOOT), "rmse_sum": total, "n_cpg": count}))
    pd.concat(tables, ignore_index=True).to_parquet(path, index=False)
    pd.concat(boots, ignore_index=True).to_parquet(path.with_name(f"bootstrap_{shard_index:04d}.parquet"), index=False)
    path.with_name(f"completion_{shard_index:04d}.json").write_text(json.dumps({
        "finished_utc": datetime.now(timezone.utc).isoformat(), "shard_index": shard_index, "n_loci": len(loci),
        "checks": checks, "code_sha256": file_hash(Path(__file__)), "command": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "n_boot": N_BOOT, "seed": SEED}, indent=2) + "\n")
    print(f"shard {shard_index}: {len(loci)} loci, {len(checks)} checks passed", flush=True)


def collect(out):
    shards = pd.read_csv(HOLDOUT / "atlas_shards.tsv", sep="\t").shard_index.tolist()
    missing = [i for i in shards if not (out / "shards" / f"completion_{i:04d}.json").exists()]
    if missing:
        raise FileNotFoundError(f"Missing shards: {missing}")
    table = pd.concat([pd.read_parquet(out / "shards" / f"shard_{i:04d}.parquet") for i in shards], ignore_index=True)
    boot = pd.concat([pd.read_parquet(out / "shards" / f"bootstrap_{i:04d}.parquet") for i in shards], ignore_index=True)
    boot = boot.groupby(["heldout_arm", "target", "column", "draw"], as_index=False)[["rmse_sum", "n_cpg"]].sum()
    boot["mean"] = boot.rmse_sum / boot.n_cpg
    eligible = table.loc[table.eligible].drop(columns="eligible")
    published = pd.read_csv(HOLDOUT / "summary/R1-6_summary.tsv", sep="\t")
    checks, rows = [], []
    for (arm, target), sub in eligible.groupby(["heldout_arm", "target"]):
        for column, method in (("A2H_rmse", "A2H"), ("linear_rmse", "reference_linear")):
            pub = published.loc[(published.model == arm) & (published.target == target) & (published.method == method)].iloc[0]
            checks.append({"item": f"{arm} {target} {method} mean per-CpG RMSE and CpG count vs published summary",
                           "published": pub.mean_site_rmse, "reproduced": sub[column].mean(),
                           "pass": bool(abs(pub.mean_site_rmse - sub[column].mean()) < 1e-12 and pub.n_sites == len(sub))})
        paired = sub.loc[sub.full205_complete]
        b = boot.loc[(boot.heldout_arm == arm) & (boot.target == target)].pivot(index="draw", columns="column", values="mean")
        lost = int((paired.shape[0] - boot.loc[(boot.heldout_arm == arm) & (boot.target == target) &
                                               (boot.column == "A2H_rmse")].n_cpg).max())
        for column in COLUMNS:
            method = "A2H" if column.startswith("A2H") else "linear"
            hold = f"{method}_rmse"
            diff = paired[hold] - paired[column]
            bd = b[hold] - b[column]
            rows.append({"heldout_arm": arm, "target": target, "column": column,
                         "n_cpg_eligible": len(sub), "n_cpg_paired": len(paired),
                         "n_heldout_samples": int(sub.n_heldout_obs.max()),
                         "mean_rmse": paired[column].mean(),
                         "mean_rmse_ci_low": b[column].quantile(0.025), "mean_rmse_ci_high": b[column].quantile(0.975),
                         "median_rmse": paired[column].median(),
                         "holdout_minus_this": diff.mean(),
                         "holdout_minus_this_ci_low": bd.quantile(0.025), "holdout_minus_this_ci_high": bd.quantile(0.975),
                         "fraction_cpg_holdout_higher": (diff > 0).mean(),
                         "max_cpg_lost_in_a_draw": lost})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.DataFrame(checks).to_csv(out / "checks.tsv", sep="\t", index=False)
    eligible.to_parquet(out / "per_cpg_heldout_with_full205.parquet", index=False)
    if not all(c["pass"] for c in checks):
        raise ValueError("Collect checks failed; see checks.tsv")
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "collect"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    if args.action == "run":
        run_shard(args.out, args.shard_index)
    else:
        collect(args.out)


if __name__ == "__main__":
    main()
