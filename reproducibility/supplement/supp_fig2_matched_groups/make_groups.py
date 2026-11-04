"""Resp 1.5 (R1-4): disjoint matched 10-CpG groups and group-level r / RMSE (reference implementation).

Reproduces the published R1-4 grouping exactly when run on the PGP-UK folder of this package with
--mask mask_primary_external and --extra R1-4/R1-4_old_grouping_extra_probes_outside_audited_rows.tsv.
Works on any cohort laid out the same way (probe_id, locus, one column per individual):
  <cohort>/base/array_beta.tsv.gz, base/wgbs_<T>.tsv.gz, base/A2H_<T>.tsv.gz, masks/<mask>_<T>.tsv.gz
for T in CHALM and CAMDA. Requires only numpy and pandas. See GROUPING_SPEC.md for the rule.

    python make_groups.py --cohort-dir PGPUK --mask mask_primary_external --out my_groups
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ("CHALM", "CAMDA")
METHODS = {"A2H": None, "m2": 2, "m3": 3}


def read(path):
    return pd.read_csv(path, sep="\t", na_values="NA", keep_default_na=False, float_precision="round_trip")


def load(cohort_dir, mask, extra):
    """Per target: CpG x individual arrays of observed truth, A2H and array beta, restricted to eligible CpGs."""
    cohort_dir = Path(cohort_dir)
    beta = read(cohort_dir / "base/array_beta.tsv.gz")
    ids = [c for c in beta.columns if c not in ("probe_id", "locus")]
    probes = beta.probe_id.to_numpy()
    b = beta[ids].to_numpy(np.float64)
    data, valid = {}, np.ones(b.shape, dtype=bool)
    for t in TARGETS:
        y = read(cohort_dir / f"base/wgbs_{t}.tsv.gz")[ids].to_numpy(np.float64)
        a = read(cohort_dir / f"base/A2H_{t}.tsv.gz")[ids].to_numpy(np.float64)
        keep = np.ones(b.shape, dtype=bool) if mask is None else read(cohort_dir / f"masks/{mask}_{t}.tsv.gz")[ids].to_numpy() == 1
        # Step 1: a cell is valid when it passes the chosen mask and truth, A2H and array beta are present.
        valid &= keep & np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
        data[t] = {"observed": y, "A2H": a}
    # Step 2: a CpG is eligible when every individual is valid for both targets.
    eligible = valid.all(axis=1)
    out = {"probes": probes[eligible], "ids": ids, "beta": b[eligible]}
    for t in TARGETS:
        out[t] = {k: v[eligible] for k, v in data[t].items()}
    if extra is not None:
        # Optional: CpGs outside the matrices (the 8 probes of the old grouping), long format, all cells present.
        e = pd.read_csv(extra, sep="\t", float_precision="round_trip")
        extra_probes = sorted(e.probe_id.unique())
        def wide(t, col):
            return e.loc[e.target == t].pivot(index="probe_id", columns="individual", values=col).reindex(index=extra_probes, columns=ids).to_numpy(np.float64)
        out["probes"] = np.concatenate([out["probes"], extra_probes])
        out["beta"] = np.vstack([out["beta"], wide("CHALM", "array_beta")])
        for t in TARGETS:
            out[t]["observed"] = np.vstack([out[t]["observed"], wide(t, "wgbs_observed_depth_ge10")])
            out[t]["A2H"] = np.vstack([out[t]["A2H"], wide(t, "A2H_deployed")])
    return out


def make_groups(probes, beta, block_size=100, group_size=10):
    """Steps 3-6: sort by mean beta into blocks, sort each block by variance, cut consecutive groups."""
    stats = pd.DataFrame({"probe_id": probes, "n_participants": beta.shape[1],
                          "mean_m": beta.mean(axis=1), "var_m": beta.var(axis=1, ddof=0)})
    ordered = stats.sort_values(["mean_m", "probe_id"], kind="stable").reset_index(drop=True)
    ordered["mean_rank"] = np.arange(len(ordered)) + 1
    ordered["block_id"] = np.arange(len(ordered)) // block_size
    ordered = ordered.sort_values(["block_id", "var_m", "mean_m", "probe_id"], kind="stable")
    rank = ordered.groupby("block_id", sort=False).cumcount()
    ordered["group_id"] = ordered.block_id * (block_size // group_size) + rank // group_size
    ordered["member_rank"] = rank % group_size + 1
    full = ordered.groupby("group_id").probe_id.transform("size") == group_size
    groups = ordered.loc[full].sort_values(["group_id", "member_rank"]).reset_index(drop=True)
    return groups, int((~full).sum())


def pearson_rmse(y, p):
    """Row-wise Pearson r and RMSE over each group's pooled observations."""
    yc, pc = y - y.mean(axis=1, keepdims=True), p - p.mean(axis=1, keepdims=True)
    sxy, syy, spp = (yc * pc).sum(axis=1), (yc * yc).sum(axis=1), (pc * pc).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((syy > 0) & (spp > 0), sxy / np.sqrt(syy * spp), np.nan)
    return r, np.sqrt(((p - y) ** 2).mean(axis=1))


def score(data, groups, group_size=10):
    """Step 7: per group, target and method, pool group_size CpGs x all individuals."""
    pos = pd.Index(data["probes"]).get_indexer(groups.probe_id)
    k = data["beta"].shape[1]
    ids = groups.group_id.to_numpy()[::group_size]
    rows = []
    for t in TARGETS:
        y = data[t]["observed"][pos].reshape(-1, group_size * k)
        for method, power in METHODS.items():
            p = (data[t]["A2H"] if power is None else data["beta"] ** power)[pos].reshape(-1, group_size * k)
            r, rmse = pearson_rmse(y, p)
            rows.append(pd.DataFrame({"group_id": ids, "target": t, "method": method, "n_cpgs": group_size,
                                      "n_participants": k, "n": group_size * k, "pearson_r": r, "rmse": rmse,
                                      "constant_observed": np.ptp(y, axis=1) == 0, "constant_prediction": np.ptp(p, axis=1) == 0,
                                      "correlation_defined": np.isfinite(r)}))
    return pd.concat(rows, ignore_index=True)


def run(cohort_dir, mask="mask_primary_external", extra=None, block_size=100, group_size=10):
    data = load(cohort_dir, mask, extra)
    groups, remainder = make_groups(data["probes"], data["beta"], block_size, group_size)
    metrics = score(data, groups, group_size)
    flow = {"eligible_cpgs": len(data["probes"]), "groups": groups.group_id.nunique(), "grouped_cpgs": len(groups),
            "remainder": remainder, "individuals": data["beta"].shape[1]}
    return groups, metrics, flow


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--mask", default="mask_primary_external", help="mask file stem (without _<target>.tsv.gz); 'none' for no mask")
    parser.add_argument("--extra", default=None)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--out", required=True)
    a = parser.parse_args()
    groups, metrics, flow = run(a.cohort_dir, None if a.mask == "none" else a.mask, a.extra, a.block_size, a.group_size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out / "groups.tsv", sep="\t", index=False)
    metrics.to_csv(out / "group_metrics.tsv", sep="\t", index=False)
    summary = metrics.groupby(["target", "method"]).agg(median_r=("pearson_r", "median"), median_rmse=("rmse", "median"),
                                                         n_groups=("group_id", "size")).reset_index()
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    print(flow)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
