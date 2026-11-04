"""Resp 3.11 (R3-S4/S5): unfiltered CpG x individual matrices for PGP-UK and GSE240482.

Rows are every array probe in the audited EPIC universe (S/E1/epic_mapping.tsv). Nothing is
masked by A2H input support, |array beta - WGBS beta|, WGBS depth above the caller floor
(4 reads) or a per-CpG sample minimum; missing cells are NaN. WGBS depth and A2H support are
written as matrices so the published rules can be reapplied, and the checks confirm that
doing so reproduces the published per-individual tables.
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

from src.evaluate_pgpuk import file_hash, map_array_samples, row_stats
from src.gse128734_external import read_truth

SRC = Path("external_validation")
V2 = SRC / "results/paired_validation_v2"
V2_D10 = SRC / "results/paired_validation_v2_depth10"
E1 = Path("results/r2_2_sensitivity_20261011/E1/epic_mapping.tsv")
PROBE_MAP = Path("legacy_model/A2H_cpg_probe.hg38.tsv")
PGP_OBS = Path("results/pgpuk_depth10_calibrated")
R3 = Path("results/r3_epic_participant_validation_20261012/full")
PKG311 = Path("collaborator_figure_data_20261017/package_v4/A2H_figure_data_20261017/Resp3.11_3.12_R3-4_R3-5_PGPUK_participants")
GSE = Path("results/gse240482_epicv1_epic_controls_20261017")
GSE_KEY = Path("collaborator_figure_data_20261017/cpg_level_20261018/A2H_cpg_level_GSE240482_20261018/pairing_key.tsv")
OUT = Path("results/r3_11_raw_matrices_20261020")
NAME = "A2H_Resp3.11-3.12-R3-4-R3-5_per-individual-unfiltered-matrices_PGPUK_GSE240482_20261020"
PACKAGE = Path("collaborator_figure_data_20261017/r3_11_raw_matrices_20261020") / NAME
TARGETS = ("CHALM", "CAMDA")
CHECKS = []


def check(cohort, item, expected, observed, ok):
    CHECKS.append({"cohort": cohort, "item": item, "expected": expected, "observed": observed, "pass": bool(ok)})


def pgpuk_pairs():
    pairs = pd.read_csv(SRC / "metadata/paired_validation_v2/paired_samples.tsv", sep="\t").query("study == 'pgpuk'")
    key = pd.read_csv(PKG311 / "pairing_key.tsv", sep="\t")
    pairs = key.merge(pairs[["array_id", "wgbs_id"]].rename(columns={"array_id": "sample_id"}), on="sample_id",
                      how="left", validate="one_to_one")
    if len(pairs) != 10 or pairs.wgbs_id.isna().any():
        raise ValueError("Expected ten PGP-UK pairs with WGBS IDs")
    return pairs


def pgpuk_rows():
    e1 = pd.read_csv(E1, sep="\t", usecols=["probe_id", "locus"])
    beta = pd.read_csv(V2 / "array/pgpuk_450K.beta.tsv", sep="\t", index_col=0, usecols=[0])
    return e1.loc[e1.probe_id.isin(beta.index)].sort_values("probe_id").reset_index(drop=True)


def depth(index):
    """Per-sample PGP-UK WGBS beta / CHALM values with read counts (caller floor 4)."""
    row = pgpuk_pairs().iloc[index]
    dest = OUT / "pgpuk_wgbs" / f"{row.sample_id}.parquet"
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = pgpuk_rows()
    result = rows.copy()
    for key, path in (("beta", V2 / f"wgbs/traditional/{row.wgbs_id}_methRatio.txt"),
                      ("CHALM", V2 / f"wgbs/chalm/{row.wgbs_id}_CHALM.txt")):
        table = read_truth(path, rows).reindex(rows.locus)
        result[key] = table.ratio.to_numpy(np.float64)
        result[f"{key}_depth"] = table.CT_count.to_numpy(np.float64)
    result.to_parquet(dest, index=False)
    print(f"{row.sample_id}: {int(np.isfinite(result.beta).sum())} beta, {int(np.isfinite(result.CHALM).sum())} CHALM", flush=True)


def wide(path, rows, columns):
    return pd.read_csv(path, sep="\t", index_col=0).reindex(index=rows, columns=columns).to_numpy(np.float64)


def gate_cells(gate, rows, samples):
    i, j = np.nonzero(gate)
    return pd.MultiIndex.from_arrays([rows[i], samples[j]])


def build_pgpuk():
    pairs = pgpuk_pairs()
    cols = pairs.sample_id.tolist()
    rows = pgpuk_rows()
    probes = rows.probe_id.to_numpy()
    idat = pd.read_csv(SRC / "metadata/paired_validation_v2/paired_array_idat_manifest.tsv", sep="\t")
    beta = map_array_samples(pd.read_csv(V2 / "array/pgpuk_450K.beta.tsv", sep="\t", index_col=0), idat, cols)
    m = {"array_beta": beta.reindex(probes).to_numpy(np.float64)}
    for target in TARGETS:
        m[f"A2H_{target}"] = wide(V2 / f"a2h_array/pgpuk_450K.{target.lower()}.tsv", probes, cols)
    m["A2H_input_support"] = wide(V2 / "a2h_array/pgpuk_450K.confidence.tsv", probes, cols)
    parts = [pd.read_parquet(OUT / "pgpuk_wgbs" / f"{s}.parquet") for s in cols]
    for p in parts:
        if not p.probe_id.equals(rows.probe_id):
            raise ValueError("Per-sample WGBS rows differ from the matrix rows")
    for key in ("beta", "beta_depth", "CHALM", "CHALM_depth"):
        m[f"wgbs_{key}"] = np.stack([p[key].to_numpy() for p in parts], axis=1)
    m["wgbs_CAMDA"] = m["wgbs_CHALM"] - m["wgbs_beta"]

    cohort = "PGP-UK"
    # The probe-space matrices used A2H's HM450 probe map; E1 probes absent from it have no cells there.
    hm450 = pd.read_csv(PROBE_MAP, sep="\t", usecols=["HM450.hg38"]).dropna()["HM450.hg38"]
    unmapped = ~rows.probe_id.isin(hm450).to_numpy()
    d10 = {"wgbs_beta": m["wgbs_beta_depth"] >= 10, "wgbs_CHALM": m["wgbs_CHALM_depth"] >= 10}
    d10["wgbs_CAMDA"] = d10["wgbs_beta"] & d10["wgbs_CHALM"]
    for ns, root, floor in (("depth>=4", V2, None), ("depth>=10 (via depth matrices)", V2_D10, 10)):
        for key, name in (("wgbs_beta", "traditional"), ("wgbs_CHALM", "chalm"), ("wgbs_CAMDA", "camda")):
            ref = wide(root / f"wgbs/probe_space/pgpuk_450K.{name}_probe_matrix.tsv", probes, cols)
            mine = m[key] if floor is None else np.where(d10[key], m[key], np.nan)
            known = np.isfinite(ref)
            tol = 1e-15 if key == "wgbs_CAMDA" else 0.0
            extra = np.isfinite(mine) & ~known
            check(cohort, f"{key} vs {ns} probe-space matrix: same cells and values (CAMDA tol 1e-15)",
                  int(known.sum()), int((known & np.isfinite(mine)).sum()),
                  np.isfinite(mine[known]).all() and np.abs(mine[known] - ref[known]).max() <= tol)
            check(cohort, f"{key} vs {ns}: extra cells only on {int(unmapped.sum())} E1 probes absent from A2H HM450 map",
                  0, int(extra[~unmapped].sum()), not extra[~unmapped].any())

    # Published rule: depth>=10 truth, support>=0.05, finite inputs, >=8 participants per CpG.
    samples = np.array(cols)
    metrics = pd.read_csv(R3 / "sample_metrics.tsv", sep="\t")
    for target in TARGETS:
        truth10 = np.where(d10[f"wgbs_{target}"], m[f"wgbs_{target}"], np.nan)
        gate = np.isfinite(truth10) & np.isfinite(m[f"A2H_{target}"]) & np.isfinite(m["array_beta"]) & (m["A2H_input_support"] >= 0.05)
        gate &= (gate.sum(axis=1) >= 8)[:, None]
        obs = pd.read_parquet(PGP_OBS / f"{target}_observations.parquet", columns=["probe_id", "sample_id", "observed", "A2H", "m"])
        obs = obs.loc[obs.probe_id.isin(rows.probe_id)].set_index(["probe_id", "sample_id"])
        cells = gate_cells(gate, probes, samples)
        check(cohort, f"{target} published cells (E1 rows) = 3.11 table rows", 1214068, len(obs), len(obs) == 1214068)
        check(cohort, f"{target} published cells all satisfy the reapplied rule", len(obs), int(obs.index.isin(cells).sum()), obs.index.isin(cells).all())
        check(cohort, f"{target} reapplied-rule cells not in published table (other filters)", 0, int((~cells.isin(obs.index)).sum()), True)
        r, c = probes.searchsorted(obs.index.get_level_values(0)), pd.Index(cols).get_indexer(obs.index.get_level_values(1))
        for column, key in (("observed", f"wgbs_{target}"), ("A2H", f"A2H_{target}"), ("m", "array_beta")):
            # Published CAMDA truth is CHALM - beta after a text round trip (<= 1 ulp).
            tol = 1e-15 if (target == "CAMDA" and column == "observed") else 0.0
            diff = float(np.abs(obs[column].to_numpy() - m[key][r, c]).max())
            check(cohort, f"{target} {column} values equal published (tol {tol:g})", 0.0, diff, diff <= tol)
        published = np.zeros_like(gate)
        published[r, c] = True
        for j, s in enumerate(cols):
            y = np.where(published[:, j], truth10[:, j], np.nan)[:, None].T
            p = np.where(published[:, j], m[f"A2H_{target}"][:, j], np.nan)[:, None].T
            st = row_stats(y, p)
            ref = metrics.loc[(metrics.target == target) & (metrics.sample_id == s)].iloc[0]
            check(cohort, f"{target} {s} n / RMSE / r vs published participant metrics", f"{ref.n} / {ref.rmse:.10g} / {ref.pearson_r:.10g}",
                  f"{int(st['n'][0])} / {st['rmse'][0]:.10g} / {st['pearson_r'][0]:.10g}",
                  int(st["n"][0]) == ref.n and abs(st["rmse"][0] - ref.rmse) < 1e-12 and abs(st["pearson_r"][0] - ref.pearson_r) < 1e-12)
    flag = pd.read_csv(PKG311 / "per_participant_cpg_observations.tsv.gz", sep="\t", usecols=["target", "probe_id", "sample_id", "dbeta_le_0.05"])
    for target in TARGETS:
        f = flag.loc[flag.target == target]
        r, c = probes.searchsorted(f.probe_id), pd.Index(cols).get_indexer(f.sample_id)
        wb10 = np.where(d10["wgbs_beta"], m["wgbs_beta"], np.nan)
        mine = np.abs(m["array_beta"][r, c] - wb10[r, c]) <= 0.05
        check(cohort, f"{target} |dbeta|<=0.05 flag recomputed from matrices (depth>=10 WGBS beta) equals shipped flag",
              int(f["dbeta_le_0.05"].sum()), int(mine.sum()), np.array_equal(mine, f["dbeta_le_0.05"].to_numpy()))
    return rows, pairs, cols, m


def build_gse():
    key = pd.read_csv(GSE_KEY, sep="\t")
    cols = key.array_id.tolist()
    e1 = pd.read_csv(E1, sep="\t", usecols=["probe_id", "locus"])
    mapping = pd.read_csv(GSE / "mapping.tsv", sep="\t")
    keep = mapping.probe_id.isin(e1.probe_id).to_numpy()
    rows = mapping.loc[keep].reset_index(drop=True)
    if not rows.merge(e1, on="probe_id", suffixes=("", "_e1")).pipe(lambda x: x.locus.eq(x.locus_e1).all()):
        raise ValueError("GSE240482 mapping disagrees with E1 loci")
    probes = rows.probe_id.to_numpy()
    m = {"array_beta": wide(GSE / "array_beta.tsv", probes, cols)}
    for target in TARGETS:
        m[f"A2H_{target}"] = wide(GSE / f"transformation/gse240482_EPICv1.{target.lower()}.tsv", probes, cols)
    m["A2H_input_support"] = wide(GSE / "transformation/gse240482_EPICv1.confidence.tsv", probes, cols)
    truth = {k: [] for k in ("beta", "beta_depth", "CHALM", "CHALM_depth", "CAMDA")}
    for a in cols:
        t = pd.read_parquet(GSE / "truth" / f"{a}.parquet")
        if not t.probe_id.equals(mapping.probe_id):
            raise ValueError("Truth row order differs from mapping")
        for k in truth:
            truth[k].append(t[k].to_numpy(np.float64)[keep])
    for k, v in truth.items():
        m[f"wgbs_{k}"] = np.stack(v, axis=1)
    cohort = "GSE240482"
    check(cohort, "rows = branch mapping probes in audited E1 universe", len(mapping), len(rows), True)
    samples = np.array(cols)
    published_metrics = pd.read_csv(GSE / "metrics.tsv", sep="\t")
    for target in TARGETS:
        gate = (np.isfinite(m[f"wgbs_{target}"]) & np.isfinite(m["array_beta"]) & np.isfinite(m[f"A2H_{target}"])
                & (m["A2H_input_support"] >= 0.05) & (m["wgbs_CHALM_depth"] >= 10))
        if target == "CAMDA":
            gate &= m["wgbs_beta_depth"] >= 10
        obs = pd.read_parquet(GSE / f"{target}_observations.parquet", columns=["probe_id", "sample_id", "observed", "m", "WGBS_beta", "A2H"])
        out_cells = obs.loc[~obs.probe_id.isin(rows.probe_id)]
        check(cohort, f"{target} published cells on probes outside E1 (not in matrices): "
              + ";".join(f"{p}/{s}" for p, s in zip(out_cells.probe_id, out_cells.sample_id)), 0, len(out_cells), True)
        outside_by_array = out_cells.sample_id.value_counts()
        obs = obs.loc[obs.probe_id.isin(rows.probe_id)].set_index(["probe_id", "sample_id"])
        cells = gate_cells(gate, probes, samples)
        check(cohort, f"{target} published cells all satisfy the reapplied rule", len(obs), int(obs.index.isin(cells).sum()), obs.index.isin(cells).all())
        check(cohort, f"{target} reapplied-rule cells not in published table (other filters)", 0, int((~cells.isin(obs.index)).sum()), True)
        pos = pd.Series(np.arange(len(probes)), index=probes)
        r, c = pos.loc[obs.index.get_level_values(0)].to_numpy(), pd.Index(cols).get_indexer(obs.index.get_level_values(1))
        for column, k in (("observed", f"wgbs_{target}"), ("m", "array_beta"), ("WGBS_beta", "wgbs_beta"), ("A2H", f"A2H_{target}")):
            check(cohort, f"{target} {column} values equal published", 0.0, float(np.nanmax(np.abs(obs[column].to_numpy() - m[k][r, c]))),
                  np.array_equal(obs[column].to_numpy(), m[k][r, c]))
        published = np.zeros_like(gate)
        published[r, c] = True
        for j, row in key.iterrows():
            y = np.where(published[:, j], m[f"wgbs_{target}"][:, j], np.nan)[None, :]
            p = np.where(published[:, j], m[f"A2H_{target}"][:, j], np.nan)[None, :]
            st = row_stats(y, p)
            ref = published_metrics.loc[(published_metrics.target == target) & (published_metrics.depth == 10) & (published_metrics.method == "A2H")
                                        & (published_metrics.scope == "sample") & (published_metrics.sample_id == row.tumour_id)].iloc[0]
            n_out = int(outside_by_array.get(row.array_id, 0))
            if n_out:
                # Published value includes out-of-universe cells; compare n after removing them and RMSE to the published cells kept.
                part = obs.xs(row.array_id, level="sample_id")
                ref_rmse = float(np.sqrt(np.mean((part.A2H.to_numpy() - part.observed.to_numpy()) ** 2)))
            else:
                ref_rmse = ref.rmse
            check(cohort, f"{target} {row.tumour_id} n / RMSE vs published per-tumour metrics"
                  + (f" (published n {ref.n}, RMSE {ref.rmse:.10g}, minus {n_out} out-of-universe cell)" if n_out else ""),
                  f"{ref.n - n_out} / {ref_rmse:.10g}", f"{int(st['n'][0])} / {st['rmse'][0]:.10g}",
                  int(st["n"][0]) == ref.n - n_out and abs(st["rmse"][0] - ref_rmse) < 1e-12)
    return rows, key, cols, m


MATRICES = ["array_beta", "wgbs_beta", "wgbs_CHALM", "wgbs_CAMDA", "A2H_CHALM", "A2H_CAMDA",
            "wgbs_beta_depth", "wgbs_CHALM_depth", "A2H_input_support"]


def write_cohort(folder, cohort, rows, ids, m, key):
    folder.mkdir(parents=True)
    key.to_csv(folder / "pairing_key.tsv", sep="\t", index=False)
    for name in MATRICES:
        frame = pd.DataFrame(m[name], columns=ids)
        if name.endswith("_depth"):
            frame = frame.astype("Int64")
        frame.insert(0, "locus", rows.locus.to_numpy())
        frame.insert(0, "probe_id", rows.probe_id.to_numpy())
        path = folder / f"{name}.tsv.gz"
        frame.to_csv(path, sep="\t", index=False, na_rep="NA")
        back = pd.read_csv(path, sep="\t", na_values="NA", float_precision="round_trip")
        values = back[ids].to_numpy(np.float64)
        check(cohort, f"{name}.tsv.gz round-trips exactly", int(np.isfinite(m[name]).sum()), int(np.isfinite(values).sum()),
              np.array_equal(values, m[name], equal_nan=True) and back.probe_id.equals(rows.probe_id))
    ok_depth = np.array_equal(m["wgbs_beta_depth"], m["wgbs_CHALM_depth"], equal_nan=True)
    check(cohort, "wgbs_beta_depth identical to wgbs_CHALM_depth (informational)", True, ok_depth, True)
    summary = []
    for j, s in enumerate(ids):
        row = {"individual": s, "n_rows": len(rows)}
        for name in MATRICES:
            row[f"finite_{name}"] = int(np.isfinite(m[name][:, j]).sum())
        for target in TARGETS:
            d = m["wgbs_CHALM_depth"][:, j] >= 10
            if target == "CAMDA":
                d &= m["wgbs_beta_depth"][:, j] >= 10
            usable = np.isfinite(m[f"wgbs_{target}"][:, j]) & np.isfinite(m[f"A2H_{target}"][:, j])
            row[f"{target}_pairs_any_depth"] = int(usable.sum())
            row[f"{target}_pairs_depth_ge10"] = int((usable & d).sum())
            row[f"{target}_pairs_depth_ge10_support_ge0.05"] = int((usable & d & (m["A2H_input_support"][:, j] >= 0.05)).sum())
        summary.append(row)
    pd.DataFrame(summary).to_csv(folder / "missingness_summary.tsv", sep="\t", index=False)
    return ok_depth


def assemble():
    if PACKAGE.exists():
        raise FileExistsError(PACKAGE)
    pg_rows, pg_pairs, pg_cols, pg = build_pgpuk()
    gs_rows, gs_key, gs_cols, gs = build_gse()
    renamed = {"PGP-UK": (pg_rows, pg_cols, pg, pg_pairs), "GSE240482": (gs_rows, gs_key.tumour_id.tolist(), gs, gs_key)}
    depth_same = {}
    for cohort, (rows, ids, m, key) in renamed.items():
        depth_same[cohort] = write_cohort(PACKAGE / cohort.replace("-", ""), cohort, rows, ids, m, key)
    checks = pd.DataFrame(CHECKS)
    checks.to_csv(OUT / "checks.tsv", sep="\t", index=False)
    checks.to_csv(PACKAGE / "CHECKS.tsv", sep="\t", index=False)
    (OUT / "provenance.json").write_text(json.dumps({
        "finished_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "code_sha256": file_hash(Path(__file__)), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "rows": {"PGP-UK": len(pg_rows), "GSE240482": len(gs_rows)}, "wgbs_caller_min_depth": 4,
        "depth_matrices_identical": depth_same, "package": str(PACKAGE)}, indent=2) + "\n")
    print(checks.to_string(index=False), flush=True)
    if not checks["pass"].all():
        raise SystemExit("checks failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["depth", "assemble"])
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.action == "depth":
        depth(args.index)
    else:
        assemble()


if __name__ == "__main__":
    main()
