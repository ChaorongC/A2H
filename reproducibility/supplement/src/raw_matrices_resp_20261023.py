"""Unfiltered per-individual matrices for team responses 1.5 (R1-4), 1.6 (R1-5), 2.3 (R2-2), 2.4 (R2-3).

User rule (2026-10-23): ship every raw prediction without filtering (no |dbeta| filter, depth
filter, support cutoff or completeness rule); filters are shipped as 0/1 mask matrices.
One standalone zip per response; cohorts PGP-UK and GSE240482; atlas per-sample data included.

Stages write unique files under STAGE (DFS3b) and a JSON record with known-answer checks under
OUT/stage_records; ``assemble`` maps staged files into the four zips. Run from R/:
    python -m src.raw_matrices_resp_20261023 <action> [--index i]
Matrix layout matches Resp 3.11: probe_id, locus (hg38 chr_pos, 1-based), one column per
individual, NA for missing, full float precision (read pandas with float_precision='round_trip').
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash, row_stats
from src.r2_2_common import fit_compatible, site_metrics
from src.r2_2_e2e3_external import predict as refit_predict, reference_matrices
from src.tissue_only_transfer import predict_coefficients

R3B = Path("work/a2h_raw_matrices_20261023")
STAGE = R3B / "staging"
ZIPS = R3B / "packages"
OUT = Path("results/raw_matrices_resp_20261023")
DOCS = Path("collaborator_figure_data_20261017/raw_matrices_resp_20261023")
S = Path("results/r2_2_sensitivity_20261011")
E1 = S / "E1"
PKG311 = Path("collaborator_figure_data_20261017/r3_11_raw_matrices_20261020/"
              "A2H_Resp3.11-3.12-R3-4-R3-5_per-individual-unfiltered-matrices_PGPUK_GSE240482_20261020")
TISSUE = Path("results/tissue_only_r1_5_r1_6_20261008")
CONTROLS = Path("results/reference_controls_epic_20261017")
GSE = Path("results/gse240482_epicv1_epic_controls_20261017")
GROUPS = Path("results/pgpuk_depth10_beta_block100_groups10_complete10")
PGP_OBS = Path("results/pgpuk_depth10_calibrated")
REFIT = Path("results/r2_2_e2e3_external_20261016/refit")
PGPUK_C = Path("results/r2_2_e2e3_pgpuk_c_20261016/pgpuk_c.tsv")
EPIC_DIR = Path("work/r2_2_e2e3_20261015/summary_20261016/epic")
E2E3_SUMMARY = Path("results/r2_2_e2e3_20261015/summary_20261016")
ATLAS39 = Path("results/r2_2_atlas39_a2h_conditions_20261021")
SEL39 = S / "E2_E3_access/selection_39_groups/E2_E3_sample_selection.tsv"
SAMPLES = Path("results/atlas_splits/atlas_samples.tsv")
SPLITS = Path("results/atlas_splits/splits.tsv")
SHARDS = Path("results/atlas_shards_original_full/shards.tsv")
CV = Path("results/atlas_cv_full/original")
TARGETS = ("CHALM", "CAMDA")
COHORTS = ("PGPUK", "GSE240482")
BASE = ["array_beta", "wgbs_beta", "wgbs_CHALM", "wgbs_CAMDA", "A2H_CHALM", "A2H_CAMDA",
        "wgbs_beta_depth", "wgbs_CHALM_depth", "A2H_input_support"]
CONDITIONS = ([("e2", "native")] + [("e2", f"seed{s}_depth{d}") for d in (20, 10, 5) for s in (1, 2, 3)]
              + [("e3", f"raw{r}") for r in (125, 100, 75)])
E1_RUNS = [(size, rep) for rep in range(20) for size in (75, 100, 150)]
ARMS = ("exclude_blood_immune", "exclude_epithelial")
CHECKS: list[dict] = []
FILES: list[dict] = []


def check(item, expected, observed, ok):
    CHECKS.append({"item": item, "expected": str(expected), "observed": str(observed), "pass": bool(ok)})
    if not ok:
        print(f"CHECK FAILED: {item}: expected {expected}, observed {observed}", flush=True)


def record(name):
    """Write this stage's file list and checks; fail the job if any check failed."""
    path = OUT / "stage_records" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": name, "finished_utc": datetime.now(timezone.utc).isoformat(),
                                "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": sys.argv,
                                "code_sha256": file_hash(Path(__file__)), "files": FILES, "checks": CHECKS},
                               indent=2, default=str) + "\n")
    failed = [c for c in CHECKS if not c["pass"]]
    print(f"{name}: {len(FILES)} files, {len(CHECKS)} checks, {len(failed)} failed", flush=True)
    if failed:
        raise SystemExit("checks failed")


def write_matrix(rel, rows, ids, values, integer=False):
    """Write probe_id, locus, one column per individual; verify an exact read-back."""
    path = STAGE / rel
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    frame = pd.DataFrame(values, columns=list(ids))
    if integer:
        frame = frame.astype("Int64")
    frame.insert(0, "locus", rows.locus.to_numpy())
    frame.insert(0, "probe_id", rows.probe_id.to_numpy())
    tmp = path.with_name(path.name + ".partial")
    frame.to_csv(tmp, sep="\t", index=False, na_rep="NA",
                 compression={"method": "gzip", "compresslevel": 6, "mtime": 0})
    del frame
    back = pd.read_csv(tmp, sep="\t", na_values="NA", keep_default_na=False, float_precision="round_trip",
                       compression="gzip")
    got = back[list(ids)].to_numpy(np.float64)
    ok = (np.array_equal(got, values.astype(np.float64), equal_nan=True)
          and np.array_equal(back.probe_id.to_numpy(), rows.probe_id.to_numpy())
          and np.array_equal(back.locus.to_numpy(), rows.locus.to_numpy()))
    if not ok:
        raise ValueError(f"Read-back differs: {path}")
    tmp.rename(path)
    FILES.append({"file": rel, "sha256": file_hash(path), "bytes": path.stat().st_size, "rows": len(rows),
                  "columns": len(ids), "non_missing": int(np.isfinite(values.astype(np.float64)).sum())})
    return path


def copy_file(source, rel):
    path = STAGE / rel
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, path)
    if file_hash(path) != file_hash(source):
        raise ValueError(f"Copy differs: {source}")
    FILES.append({"file": rel, "sha256": file_hash(path), "bytes": path.stat().st_size, "source": str(source)})


def write_table(frame, rel):
    path = STAGE / rel
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, na_rep="NA")
    FILES.append({"file": rel, "sha256": file_hash(path), "bytes": path.stat().st_size, "rows": len(frame)})


def read_matrix(path, columns=None):
    frame = pd.read_csv(path, sep="\t", na_values="NA", keep_default_na=False, float_precision="round_trip",
                        usecols=columns)
    return frame


def cohort(name, matrices=BASE):
    """Rows, individual IDs and base matrices of a cohort, from the verified Resp 3.11 package."""
    out = {}
    rows = ids = None
    for m in matrices:
        frame = read_matrix(PKG311 / name / f"{m}.tsv.gz")
        if rows is None:
            rows, ids = frame[["probe_id", "locus"]].copy(), [c for c in frame.columns if c not in ("probe_id", "locus")]
        elif not frame.probe_id.equals(rows.probe_id):
            raise ValueError("Base matrices differ in rows")
        out[m] = frame[ids].to_numpy(np.float64)
    return rows, ids, out


def atlas_rows():
    rows = pd.read_csv(E1 / "epic_mapping.tsv", sep="\t")
    if len(rows) != 824209 or rows.locus.duplicated().any():
        raise ValueError("Unexpected E1 EPIC universe")
    return rows


def by_locus(table, rows, columns):
    t = table.set_index("locus")
    if t.index.duplicated().any():
        raise ValueError("Duplicate locus in coefficient table")
    return t.reindex(rows.locus.to_numpy())[columns].reset_index(drop=True)


def cells_mask(rows, ids, probes, samples):
    mask = np.zeros((len(rows), len(ids)), dtype=np.int8)
    r = pd.Series(np.arange(len(rows)), index=rows.probe_id).reindex(probes).to_numpy()
    c = pd.Index(ids).get_indexer(samples)
    keep = ~np.isnan(r) & (c >= 0)
    mask[r[keep].astype(int), c[keep]] = 1
    return mask, int((~keep).sum()), r, c, keep


# ----------------------------------------------------------------------------------------------
# Stage: cohort base, masks, R1-4 comparators, R1-5 reference-composition predictions
# ----------------------------------------------------------------------------------------------

def stage_cohort(index):
    name = COHORTS[index]
    rows, ids, m = cohort(name)
    base = f"cohort/{name}"
    for f in BASE:
        copy_file(PKG311 / name / f"{f}.tsv.gz", f"{base}/base/{f}.tsv.gz")
    for f in ("pairing_key.tsv", "missingness_summary.tsv"):
        copy_file(PKG311 / name / f, f"{base}/base/{f}")

    # Masks (0/1). Depths are identical for beta and CHALM in both cohorts (checked), so one depth mask.
    same = np.array_equal(m["wgbs_beta_depth"], m["wgbs_CHALM_depth"], equal_nan=True)
    check(f"{name}: wgbs_beta_depth identical to wgbs_CHALM_depth", True, same, same)
    d10 = np.nan_to_num(m["wgbs_CHALM_depth"], nan=0) >= 10
    support = np.nan_to_num(m["A2H_input_support"], nan=-1) >= 0.05
    dbeta = d10 & np.isfinite(m["array_beta"]) & np.isfinite(m["wgbs_beta"]) & (np.abs(m["array_beta"] - m["wgbs_beta"]) <= 0.05)
    write_matrix(f"{base}/masks/mask_wgbs_depth_ge10.tsv.gz", rows, ids, d10.astype(np.int8), integer=True)
    write_matrix(f"{base}/masks/mask_A2H_support_ge0.05.tsv.gz", rows, ids, support.astype(np.int8), integer=True)
    write_matrix(f"{base}/masks/mask_dbeta_le0.05.tsv.gz", rows, ids, dbeta.astype(np.int8), integer=True)
    primary = {}
    for target in TARGETS:
        cell = d10 & np.isfinite(m[f"wgbs_{target}"]) & np.isfinite(m[f"A2H_{target}"]) & np.isfinite(m["array_beta"]) & support
        if name == "PGPUK":
            complete = (cell.sum(axis=1) >= 8)[:, None] & np.ones_like(cell)
            write_matrix(f"{base}/masks/mask_cpg_ge8_participants_{target}.tsv.gz", rows, ids, complete.astype(np.int8), integer=True)
            cell &= complete
        primary[target] = cell
        write_matrix(f"{base}/masks/mask_primary_external_{target}.tsv.gz", rows, ids, cell.astype(np.int8), integer=True)
    if name == "PGPUK":
        for target in TARGETS:
            check(f"PGPUK {target}: primary mask = published 3.11 cells", 1214068, int(primary[target].sum()),
                  int(primary[target].sum()) == 1214068)
            check(f"PGPUK {target}: dbeta mask within primary = published 337,845 pairs", 337845,
                  int((primary[target] & dbeta).sum()), int((primary[target] & dbeta).sum()) == 337845)
    else:
        for target in TARGETS:
            obs = pd.read_parquet(GSE / f"{target}_observations.parquet", columns=["probe_id", "sample_id"])
            key = pd.read_csv(PKG311 / name / "pairing_key.tsv", sep="\t").set_index("array_id").tumour_id
            mask, outside, *_ = cells_mask(rows, ids, obs.probe_id, obs.sample_id.map(key))
            check(f"GSE240482 {target}: primary mask = published cells (minus {outside} outside audited rows)",
                  len(obs) - outside, int(primary[target].sum()),
                  np.array_equal(mask.astype(bool), primary[target]))

    # R1-4 comparators: per-individual powers of array beta.
    for k, power in (("beta_squared", 2), ("beta_cubed", 3)):
        write_matrix(f"{base}/R1-4/{k}.tsv.gz", rows, ids, m["array_beta"] ** power)
    if name == "PGPUK":
        groups = pd.read_csv(GROUPS / "groups.tsv", sep="\t")
        inside = groups.probe_id.isin(rows.probe_id)
        complete10 = rows.probe_id.isin(groups.probe_id).to_numpy()[:, None] & np.ones((len(rows), len(ids)), bool)
        write_matrix(f"{base}/R1-4/mask_R1-4_old_grouped_cpgs.tsv.gz", rows, ids, complete10.astype(np.int8), integer=True)
        check("PGPUK R1-4: old grouped CpGs inside audited rows (8 coordinate-conflict probes outside)",
              40692, int(inside.sum()), int(inside.sum()) == 40692)
        for target in TARGETS:
            obs = pd.read_parquet(PGP_OBS / f"{target}_observations.parquet", columns=["probe_id", "sample_id", "m", "m2", "m3"])
            obs = obs.loc[obs.probe_id.isin(rows.probe_id)]
            _, _, r, c, keep = cells_mask(rows, ids, obs.probe_id, obs.sample_id)
            r = r[keep].astype(int)
            for col, power in (("m2", 2), ("m3", 3)):
                diff = float(np.abs(obs[col].to_numpy()[keep] - m["array_beta"][r, c[keep]] ** power).max())
                check(f"PGPUK {target}: published {col} equals array beta^{power} (tol 1e-12)", 0, diff, diff <= 1e-12)

    # R1-5: A2H and CpG-specific linear fits from the three atlas references.
    beta = m["array_beta"]
    for target in TARGETS:
        preds = {}
        for arm in ARMS:
            coef = pd.concat([pd.read_parquet(p) for p in sorted((TISSUE / "fits" / arm).glob(f"shard_*/{target}_coefficients.parquet"))])
            q = by_locus(coef, rows, ["a2h_intercept", "a2h_linear", "a2h_quadratic",
                                      "reference_linear_intercept", "reference_linear_slope"])
            p = predict_coefficients(q, beta)
            preds[f"A2H_{arm}"], preds[f"linear_{arm}"] = p["A2H"], p["reference_linear"]
        ctl = by_locus(pd.read_parquet(CONTROLS / f"{target}_controls.parquet"), rows, ["linear_intercept", "linear_slope"])
        preds["linear_full_reference"] = ctl.linear_intercept.to_numpy()[:, None] + ctl.linear_slope.to_numpy()[:, None] * beta
        for k in ("A2H_exclude_blood_immune", "A2H_exclude_epithelial", "linear_full_reference",
                  "linear_exclude_blood_immune", "linear_exclude_epithelial"):
            write_matrix(f"{base}/R1-5/{k}_{target}.tsv.gz", rows, ids, preds[k])
        # Known answer: published R1-5 predictions at their cells.
        if name == "PGPUK":
            pub = pd.read_parquet(TISSUE / f"summary/R1-5_{target}_common_observations.parquet")
            pairs = {"full_reference__A2H": (f"A2H_{target}", 1e-10), "full_reference__reference_linear": ("linear_full_reference", 1e-12),
                     "m": ("array_beta", 0.0)}
            for arm in ARMS:
                pairs[f"{arm}__A2H"] = (f"A2H_{arm}", 0.0)
                pairs[f"{arm}__reference_linear"] = (f"linear_{arm}", 0.0)
            samples = pub.sample_id
        else:
            pub = pd.read_parquet(GSE / f"{target}_transfer_predictions.parquet")
            key = pd.read_csv(PKG311 / name / "pairing_key.tsv", sep="\t").set_index("array_id").tumour_id
            pairs = {"full_reference": (f"A2H_{target}", 1e-10)}
            for arm in ARMS:
                pairs[arm] = (f"A2H_{arm}", 0.0)
            samples = pub.sample_id.map(key)
        mask, outside, r, c, keep = cells_mask(rows, ids, pub.probe_id, samples)
        write_matrix(f"{base}/R1-5/mask_R1-5_published_{target}.tsv.gz", rows, ids, mask, integer=True)
        check(f"{name} {target}: R1-5 published cells outside audited rows", 0, outside, True)
        r = r[keep].astype(int)
        for column, (k, tol) in pairs.items():
            source = preds.get(k, m.get(k))
            diff = float(np.nanmax(np.abs(pub[column].to_numpy()[keep] - source[r, c[keep]])))
            nan_mismatch = int((np.isfinite(pub[column].to_numpy()[keep]) != np.isfinite(source[r, c[keep]])).sum())
            check(f"{name} {target}: R1-5 published {column} reproduced (tol {tol:g})", f"0 / 0 NaN mismatches",
                  f"{diff:.3g} / {nan_mismatch}", diff <= tol and nan_mismatch == 0)
    record(f"cohort_{name}")


# ----------------------------------------------------------------------------------------------
# Stage: E1 reference-size subsets (60 fits) applied to both cohorts
# ----------------------------------------------------------------------------------------------

def stage_e1(index):
    size, rep = E1_RUNS[index]
    run = E1 / "runs" / f"N{size}_rep{rep:02d}"
    for name in COHORTS:
        rows, ids, m = cohort(name, ["array_beta"])
        beta = m["array_beta"]
        for target in TARGETS:
            q = by_locus(pd.read_parquet(run / f"{target}_quality.parquet"), rows, ["a2h_intercept", "a2h_linear", "a2h_quadratic"])
            pred = (q.a2h_intercept.to_numpy()[:, None] + q.a2h_linear.to_numpy()[:, None] * beta
                    + q.a2h_quadratic.to_numpy()[:, None] * beta ** 2)
            write_matrix(f"cohort/{name}/E1/{target}/A2H_N{size}_rep{rep:02d}_{target}.tsv.gz", rows, ids, pred)
            if name == "PGPUK":
                stored = pd.read_parquet(run / f"{target}_predictions.parquet")
                pos = pd.Series(np.arange(len(rows)), index=rows.probe_id).reindex(stored.probe_id).to_numpy()
                cols = [f"pred__{s}" for s in ids]
                sp = stored[cols].to_numpy(np.float64)
                mine = pred[pos.astype(int)]
                fin = np.isfinite(sp)
                check(f"E1 N{size} rep{rep:02d} {target}: stored predictions reproduced exactly", f"{int(fin.sum())} cells",
                      f"max |d| {float(np.abs(sp[fin] - mine[fin]).max()):.3g}", np.array_equal(sp[fin], mine[fin]))
    record(f"e1_N{size}_rep{rep:02d}")


# ----------------------------------------------------------------------------------------------
# Stage: 39-representative refits (option A) per condition, predicting both cohorts
# ----------------------------------------------------------------------------------------------

def stage_refit39(index):
    experiment, condition = CONDITIONS[index]
    folder = REFIT / f"{experiment}__{condition}"
    samples = json.loads((folder / "refit.json").read_text())["reference_samples"]
    min_train = json.loads((folder / "refit.json").read_text())["min_train"]
    loci = pd.Index(atlas_rows().locus)
    cohorts = {name: cohort(name, ["array_beta"]) for name in COHORTS}
    for target in TARGETS:
        x, chalm = reference_matrices(EPIC_DIR, experiment, condition, samples, loci)
        y = chalm - x if target == "CAMDA" else chalm
        quality = fit_compatible(x, y, min_train=min_train)
        quality.insert(0, "locus", loci)
        qpath = STAGE / f"internal/refit39_coefficients/{experiment}__{condition}_{target}.parquet"
        qpath.parent.mkdir(parents=True, exist_ok=True)
        quality.to_parquet(qpath, index=False)
        for name, (rows, ids, m) in cohorts.items():
            q = quality.set_index("locus").reindex(rows.locus.to_numpy()).reset_index(drop=True)
            pred = refit_predict(q, m["array_beta"])
            write_matrix(f"cohort/{name}/refit39/{target}/A2H_refit39_{experiment}__{condition}_{target}.tsv.gz", rows, ids, pred)
            if name == "PGPUK":
                stored = pd.read_parquet(folder / f"{target}_predictions.parquet")
                pos = pd.Series(np.arange(len(rows)), index=rows.probe_id).reindex(stored.probe_id).to_numpy().astype(int)
                sp = stored[[f"pred__{s}" for s in ids]].to_numpy(np.float64)
                fin = np.isfinite(sp)
                check(f"refit39 {experiment}__{condition} {target}: stored predictions reproduced exactly",
                      f"{int(fin.sum())} cells", f"max |d| {float(np.abs(sp[fin] - pred[pos][fin]).max()):.3g}",
                      np.array_equal(sp[fin], pred[pos][fin]))
                same_status = (stored.fit_status.to_numpy() == q.fit_status.to_numpy()[pos]).all()
                check(f"refit39 {experiment}__{condition} {target}: fit_status identical at stored loci", True, same_status, same_status)
    record(f"refit39_{experiment}__{condition}")


# ----------------------------------------------------------------------------------------------
# Stage: PGP-UK truth recomputed from read-length-capped calls (E3 option C)
# ----------------------------------------------------------------------------------------------

def stage_e3pg():
    rows, ids, m = cohort("PGPUK")
    pairs = pd.read_csv(PGP_OBS / "pairs.tsv", sep="\t").set_index("array_id")
    wgbs = pairs.loc[ids, "wgbs_id"].tolist()
    published = pd.read_csv(PGPUK_C, sep="\t")
    for condition in ("native", "raw125", "raw100", "raw75"):
        val = {k: [] for k in ("beta", "CHALM", "depth")}
        for w in wgbs:
            f = pd.read_parquet(EPIC_DIR / f"pgpuk_c__{w}__{condition}.parquet").set_index("locus").reindex(rows.locus.to_numpy())
            val["beta"].append(f.m.to_numpy(np.float64))
            val["CHALM"].append(f.chalm.to_numpy(np.float64))
            val["depth"].append(f.ct.to_numpy(np.float64))
        val = {k: np.column_stack(v) for k, v in val.items()}
        val["CAMDA"] = val["CHALM"] - val["beta"]
        if condition == "native":
            for k, base in (("beta", "wgbs_beta"), ("CHALM", "wgbs_CHALM"), ("depth", "wgbs_CHALM_depth")):
                same = np.array_equal(val[k], m[base], equal_nan=True)
                check(f"E3 PGP-UK native recall {k} equals Resp 3.11 base {base} at every cell", True, same, same)
            continue
        folder = f"cohort/PGPUK/E3_readlength_truth/{condition}"
        write_matrix(f"{folder}/wgbs_beta_{condition}.tsv.gz", rows, ids, val["beta"])
        write_matrix(f"{folder}/wgbs_CHALM_{condition}.tsv.gz", rows, ids, val["CHALM"])
        write_matrix(f"{folder}/wgbs_CAMDA_{condition}.tsv.gz", rows, ids, val["CAMDA"])
        write_matrix(f"{folder}/wgbs_depth_{condition}.tsv.gz", rows, ids, val["depth"], integer=True)
        d10 = np.nan_to_num(val["depth"], nan=0) >= 10
        write_matrix(f"{folder}/mask_wgbs_depth_ge10_{condition}.tsv.gz", rows, ids, d10.astype(np.int8), integer=True)
        # Known answer: published deployed-A2H-vs-capped-truth arm on the locked primary cells.
        for target in TARGETS:
            locked = pd.read_parquet(PGP_OBS / f"{target}_observations.parquet", columns=["probe_id", "sample_id"])
            locked = locked.loc[locked.probe_id.isin(rows.probe_id)]
            mask, *_ = cells_mask(rows, ids, locked.probe_id, locked.sample_id)
            capped = np.where(d10, val[target], np.nan)
            keep = mask.astype(bool) & np.isfinite(capped) & np.isfinite(m[f"A2H_{target}"])
            err = np.where(keep, (m[f"A2H_{target}"] - capped) ** 2, 0.0)
            n = keep.sum(axis=1)
            site = np.sqrt(err[n > 0].sum(axis=1) / n[n > 0])
            ref = published.loc[(published.target == target) & (published.condition == condition)].iloc[0]
            check(f"E3 PGP-UK {condition} {target}: published n_pairs and mean per-CpG RMSE (deployed vs capped) reproduced",
                  f"{ref.n_pairs} / {ref.mean_site_rmse_a:.12g}", f"{int(keep.sum())} / {site.mean():.12g}",
                  int(keep.sum()) == ref.n_pairs and abs(site.mean() - ref.mean_site_rmse_a) < 1e-12)
    record("e3_pgpuk_readlength")


# ----------------------------------------------------------------------------------------------
# Stage: 39 atlas representatives, per-condition calls (E2/E3) and archive values
# ----------------------------------------------------------------------------------------------

def selected39():
    sel = pd.read_csv(SEL39, sep="\t", usecols=["sample_id", "group", "refined_group", "source_tissue", "cell_type",
                                                 "germ_layer", "donor_id"])
    if len(sel) != 39:
        raise ValueError("Expected 39 representatives")
    return sel


def condition_values(experiment, condition, samples, rows):
    val = {k: [] for k in ("beta", "CHALM", "depth")}
    for s in samples:
        f = pd.read_parquet(EPIC_DIR / f"{experiment}__{s}__{condition}.parquet").set_index("locus").reindex(rows.locus.to_numpy())
        val["beta"].append(f.m.to_numpy(np.float64))
        val["CHALM"].append(f.chalm.to_numpy(np.float64))
        val["depth"].append(f.ct.to_numpy(np.float64))
    val = {k: np.column_stack(v) for k, v in val.items()}
    val["CAMDA"] = val["CHALM"] - val["beta"]
    return val


def atlas_values(samples, rows, columns):
    """Archive (training-matrix) values for samples at the E1 loci, from the 52 atlas shards."""
    out = {c: np.full((len(rows), len(samples)), np.nan) for c in columns}
    pos = pd.Series(np.arange(len(rows)), index=rows.locus)
    for shard in pd.read_csv(SHARDS, sep="\t").itertuples(index=False):
        frame = pd.read_parquet(shard.path, columns=["locus"] + [f"{c}__{s}" for c in columns for s in samples])
        frame = frame.loc[frame.locus.isin(pos.index)]
        r = pos.loc[frame.locus].to_numpy()
        for c in columns:
            out[c][r] = frame[[f"{c}__{s}" for s in samples]].to_numpy(np.float64)
    return out


def stage_atlas39_calls(index):
    sel = selected39()
    samples = sel.sample_id.tolist()
    rows = atlas_rows()[["probe_id", "locus"]]
    if index == len(CONDITIONS):
        arch = atlas_values(samples, rows, ["m", "CHALM", "CAMDA"])
        for k, name in (("m", "beta"), ("CHALM", "CHALM"), ("CAMDA", "CAMDA")):
            write_matrix(f"atlas39/archive_native/wgbs_{name}_archive.tsv.gz", rows, samples, arch[k])
        record("atlas39_calls_archive")
        return
    experiment, condition = CONDITIONS[index]
    val = condition_values(experiment, condition, samples, rows)
    folder = f"atlas39/calls/{experiment}__{condition}"
    for k in ("beta", "CHALM", "CAMDA"):
        write_matrix(f"{folder}/wgbs_{k}_{experiment}__{condition}.tsv.gz", rows, samples, val[k])
    write_matrix(f"{folder}/wgbs_depth_{experiment}__{condition}.tsv.gz", rows, samples, val["depth"], integer=True)
    deltas = pd.read_csv(E2E3_SUMMARY / "paired_epic_deltas.tsv", sep="\t")
    if condition == "native":
        other = condition_values("e3", "native", samples, rows)
        same = all(np.array_equal(val[k], other[k], equal_nan=True) for k in val)
        check("atlas39: e2 and e3 native calls identical (one native shipped)", True, same, same)
    else:
        native = condition_values("e2", "native", samples, rows)
        bad = 0
        for j, s in enumerate(samples):
            for target, key in (("m", "beta"), ("chalm", "CHALM"), ("camda", "CAMDA")):
                ref = deltas.loc[(deltas.experiment == experiment) & (deltas.sample_id == s) & (deltas.condition == condition)
                                 & (deltas.target == target)].iloc[0]
                both = np.isfinite(val[key][:, j]) & np.isfinite(native[key][:, j])
                d = val[key][both, j] - native[key][both, j]
                ok = int(both.sum()) == ref.n_matched and abs(d.mean() - ref.mean_signed_delta) < 1e-12
                bad += not ok
        check(f"atlas39 {experiment}__{condition}: published n_matched and mean shift vs native reproduced (39 samples x 3 targets)",
              0, bad, bad == 0)
    record(f"atlas39_calls_{experiment}__{condition}")


def stage_atlas39_ldo(index):
    """Leave-donor-out A2H predictions for the 39 representatives (inputs A and B, plus archive input)."""
    target = TARGETS[index]
    sel = selected39()
    samples = sel.sample_id.tolist()
    rows = atlas_rows()[["probe_id", "locus"]]
    shards = pd.read_csv(SHARDS, sep="\t").shard_index.tolist()
    f = pd.concat([pd.read_parquet(ATLAS39 / "fits" / f"{target}_shard_{i:04d}.parquet") for i in shards])
    f = f.set_index("locus").reindex(rows.locus.to_numpy())
    coef = {k: f[[f"{k}__{s}" for s in samples]].to_numpy(np.float64)
            for k in ("a2h_intercept", "a2h_linear", "a2h_quadratic", "archive_m", "archive_truth")}
    del f

    def predict(m):
        return coef["a2h_intercept"] + coef["a2h_linear"] * m + coef["a2h_quadratic"] * m ** 2

    calls = {(e, c): condition_values(e, c, samples, rows) for e, c in CONDITIONS}
    truth = {key: v[target] for key, v in calls.items()}
    truth[("archive", "native")] = coef["archive_truth"]
    preds = {"A_native_input": predict(calls[("e2", "native")]["beta"]), "archive_input": predict(coef["archive_m"])}
    for e, c in CONDITIONS[1:]:
        preds[f"B_{e}__{c}"] = predict(calls[(e, c)]["beta"])
    folder = f"atlas39/A2H_leave_donor_out/{target}"
    for k, v in preds.items():
        write_matrix(f"{folder}/A2H_LDO_{k}_{target}.tsv.gz", rows, samples, v)
    common = np.ones_like(preds["A_native_input"], dtype=bool)
    for v in list(preds.values()) + list(truth.values()):
        common &= np.isfinite(v)
    write_matrix(f"{folder}/mask_atlas39_common_cells_{target}.tsv.gz", rows, samples, common.astype(np.int8), integer=True)
    # Known answer: published pooled RMSE and n on the common cells.
    pub = pd.read_csv(ATLAS39 / "summary_common_cells.tsv", sep="\t")
    pub = pub.loc[pub.target == target]
    bad = 0
    for r in pub.itertuples(index=False):
        key = ("e2", "native") if r.experiment == "fresh" else (r.experiment, r.condition)
        if r.experiment == "archive":
            p = preds["archive_input"]
        elif r.input == "A_native_input" or key == ("e2", "native"):
            p = preds["A_native_input"]
        else:
            p = preds[f"B_{key[0]}__{key[1]}"]
        y = truth[key]
        e = np.where(common, (p - y) ** 2, 0.0)
        rmse = float(np.sqrt(e.sum() / common.sum()))
        bad += not (int(common.sum()) == r.n_cells and abs(rmse - r.pooled_rmse) < 1e-12)
    check(f"atlas39 LDO {target}: published common-cell n and pooled RMSE reproduced ({len(pub)} rows)", 0, bad, bad == 0)
    record(f"atlas39_ldo_{target}")


# ----------------------------------------------------------------------------------------------
# Stage: 205-sample atlas truth and out-of-fold A2H predictions (historical sample-level CV)
# ----------------------------------------------------------------------------------------------

def stage_atlas205_truth(index):
    column = ("m", "CHALM", "CAMDA")[index]
    samples = pd.read_csv(SAMPLES, sep="\t").sample_id.tolist()
    rows = atlas_rows()[["probe_id", "locus"]]
    val = atlas_values(samples, rows, [column])[column]
    write_matrix(f"atlas205/truth/wgbs_{'beta' if column == 'm' else column}_atlas205.tsv.gz", rows, samples, val)
    record(f"atlas205_truth_{column}")


def stage_oof(index):
    target, repeat = TARGETS[index // 10], index % 10
    samples = pd.read_csv(SAMPLES, sep="\t").sample_id.tolist()
    col = {s: i for i, s in enumerate(samples)}
    mapping = atlas_rows()
    rows = mapping[["probe_id", "locus"]]
    pos = pd.Series(np.arange(len(rows)), index=rows.locus)
    splits = pd.read_csv(SPLITS, sep="\t").query("branch == 'historical_sample'")
    predicted = np.full((len(rows), len(samples)), np.nan)
    observed = np.full_like(predicted, np.nan)
    filled = np.zeros(len(samples), dtype=bool)
    for fold in range(10):
        split = repeat * 10 + fold
        part = splits.loc[splits.split_index == split]
        test = part.loc[part.role == "test", "sample_id"].tolist()
        meta = json.loads((CV / f"historical_sample_split{split:03d}/completion.json").read_text())
        if meta["test_ids"] != test or meta["repeat"] != repeat or meta["fold"] != fold:
            raise ValueError(f"Split metadata mismatch {split}")
        c = np.array([col[s] for s in test])
        if filled[c].any():
            raise ValueError("Sample held out twice")
        for shard in sorted(mapping.shard_index.unique()):
            loci = mapping.loc[mapping.shard_index == shard, "locus"]
            data = pd.read_parquet(CV / f"historical_sample_split{split:03d}/{target}_shard{shard:04d}_predictions.parquet",
                                   columns=["locus", "sample_id", "observed", "A2H"])
            data = data.loc[data.locus.isin(loci)]
            order = data.locus.drop_duplicates().to_numpy()
            if (len(data) != len(order) * len(test) or not np.array_equal(data.sample_id.to_numpy(), np.tile(test, len(order)))
                    or len(order) != len(loci)):
                raise ValueError(f"OOF key mismatch split {split} shard {shard}")
            r = pos.loc[order].to_numpy()
            predicted[np.ix_(r, c)] = data.A2H.to_numpy(np.float64).reshape(len(order), len(test))
            observed[np.ix_(r, c)] = data.observed.to_numpy(np.float64).reshape(len(order), len(test))
        filled[c] = True
    if not filled.all():
        raise ValueError("Repeat incomplete")
    truth = atlas_values(samples, rows, [target])[target]
    same = np.array_equal(observed, truth, equal_nan=True)
    check(f"OOF {target} repeat {repeat}: CV observed equals atlas truth matrix", True, same, same)
    metrics = site_metrics(truth, predicted)
    pub = pd.concat([pd.read_parquet(p, columns=["locus", "repeat", "n", "rmse"]).query("repeat == @repeat")
                     for p in sorted((S / "E4").glob(f"shard*/{target}_per_site_by_repeat.parquet"))]).set_index("locus").reindex(rows.locus)
    ok = (np.array_equal(pub.n.to_numpy(), metrics.n.to_numpy())
          and np.allclose(pub.rmse.to_numpy(), metrics.rmse.to_numpy(), rtol=0, atol=1e-12, equal_nan=True))
    check(f"OOF {target} repeat {repeat}: published E4 per-site n and RMSE (this repeat) reproduced at {len(rows)} CpGs", True, ok, ok)
    write_matrix(f"atlas205/A2H_out_of_fold/{target}/A2H_oof_repeat{repeat}_{target}.tsv.gz", rows, samples, predicted)
    record(f"oof_{target}_repeat{repeat}")


# ----------------------------------------------------------------------------------------------
# Stage: cross-file checks (E4 averaged per-site table; R1-4 grouping spec reproduction)
# ----------------------------------------------------------------------------------------------

def stage_check(index):
    if index < 2:
        target = TARGETS[index]
        samples = pd.read_csv(SAMPLES, sep="\t").sample_id.tolist()
        rows = atlas_rows()
        truth = read_matrix(STAGE / f"atlas205/truth/wgbs_{target}_atlas205.tsv.gz")[samples].to_numpy(np.float64)
        total = np.zeros_like(truth)
        count = np.zeros(truth.shape, dtype=np.uint8)
        for repeat in range(10):
            p = read_matrix(STAGE / f"atlas205/A2H_out_of_fold/{target}/A2H_oof_repeat{repeat}_{target}.tsv.gz")[samples].to_numpy(np.float64)
            valid = np.isfinite(p) & np.isfinite(truth)
            total += np.where(valid, p, 0.0)
            count += valid.astype(np.uint8)
        averaged = np.divide(total, np.maximum(count, 1), out=np.full_like(total, np.nan), where=count > 0)
        metrics = site_metrics(truth, averaged)
        pub = pd.read_parquet(S / "E4/summary/E4_per_site_oof_metrics.parquet").query("target == @target").set_index("locus").reindex(rows.locus)
        ok = (np.array_equal(pub.n.to_numpy(), metrics.n.to_numpy())
              and np.allclose(pub.rmse.to_numpy(), metrics.rmse.to_numpy(), rtol=0, atol=1e-12, equal_nan=True))
        check(f"E4 {target}: published per-site OOF n and RMSE (10-repeat average) reproduced from shipped matrices", True, ok, ok)
        record(f"check_E4_{target}")
        return
    if index == 3:
        # R1-5: published PGP-UK mean per-CpG RMSE for every reference x method, from the shipped matrices.
        pub = pd.read_csv(Path("collaborator_figure_data_20261017/package_v4/A2H_figure_data_20261017/"
                               "Resp1.6_R1-5_reference_composition/summary_with_participant_bootstrap.tsv"), sep="\t")
        folder = STAGE / "cohort/PGPUK"
        beta = read_matrix(folder / "base/array_beta.tsv.gz")
        ids = [c for c in beta.columns if c not in ("probe_id", "locus")]
        for target in TARGETS:
            mask = read_matrix(folder / f"R1-5/mask_R1-5_published_{target}.tsv.gz")[ids].to_numpy() == 1
            y = read_matrix(folder / f"base/wgbs_{target}.tsv.gz")[ids].to_numpy(np.float64)
            for r in pub.loc[pub.target == target].itertuples(index=False):
                if r.method == "m":
                    rel = "base/array_beta.tsv.gz"
                elif r.method == "A2H":
                    rel = f"base/A2H_{target}.tsv.gz" if r.model == "full_reference" else f"R1-5/A2H_{r.model}_{target}.tsv.gz"
                else:
                    rel = f"R1-5/linear_{r.model}_{target}.tsv.gz"
                p = read_matrix(folder / rel)[ids].to_numpy(np.float64)
                valid = mask & np.isfinite(y) & np.isfinite(p)
                n = valid.sum(axis=1)
                site = np.sqrt(np.where(valid, (p - y) ** 2, 0.0).sum(axis=1)[n > 0] / n[n > 0])
                ok = int(valid.sum()) == r.n_observations and int((n > 0).sum()) == r.n_sites and abs(site.mean() - r.mean_site_rmse) < 1e-12
                check(f"R1-5 PGP-UK {target} {r.model} {r.method}: published n and mean per-CpG RMSE reproduced",
                      f"{r.n_sites} / {r.n_observations} / {r.mean_site_rmse:.12g}",
                      f"{int((n > 0).sum())} / {int(valid.sum())} / {site.mean():.12g}", ok)
        record("check_R1-5")
        return
    # R1-4: the shipped grouping script reproduces the old membership and group metrics.
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_groups", DOCS / "make_groups.py")
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    folder = STAGE / "cohort/PGPUK"
    old = pd.read_csv(GROUPS / "groups.tsv", sep="\t")
    old_metrics = pd.read_csv(GROUPS / "group_metrics.tsv", sep="\t")
    extra = STAGE / "cohort/PGPUK/R1-4/R1-4_old_grouping_extra_probes_outside_audited_rows.tsv"
    for use_extra in (True, False):
        groups, metrics, flow = mg.run(folder, mask="mask_primary_external", extra=extra if use_extra else None)
        if use_extra:
            g = groups[old.columns].reset_index(drop=True)
            keys = ["probe_id", "n_participants", "mean_rank", "block_id", "group_id", "member_rank"]
            stat = float(max(np.abs(g.mean_m - old.mean_m).max(), np.abs(g.var_m - old.var_m).max()))
            same = g[keys].equals(old[keys]) and stat <= 1e-15
            check("R1-4 make_groups.py (+8 extra probes) reproduces old membership exactly (mean/var tol 1e-15)",
                  "identical", f"membership {g[keys].equals(old[keys])}, max |d| mean/var {stat:.3g}", same)
            a = old_metrics.sort_values(["target", "method", "group_id"]).reset_index(drop=True)
            b = metrics.sort_values(["target", "method", "group_id"]).reset_index(drop=True)
            diff = float(max(np.nanmax(np.abs(a.rmse - b.rmse)), np.nanmax(np.abs(a.pearson_r - b.pearson_r))))
            check("R1-4 make_groups.py reproduces old group_metrics (r, RMSE; tol 1e-12)", len(a), f"{len(b)} rows, max |d| {diff:.3g}",
                  len(a) == len(b) and diff <= 1e-12 and a.group_id.equals(b.group_id))
        else:
            check("R1-4 make_groups.py on audited rows only (informational)", "old: 40,700 CpGs / 4,070 groups",
                  f"{flow['eligible_cpgs']} CpGs / {flow['groups']} groups / {flow['remainder']} remainder", True)
    record("check_R1-4")


def stage_extra():
    """Small tables: R1-4 old grouping files + the 8 extra probes; memberships; published summaries."""
    rows, ids, _ = cohort("PGPUK", ["array_beta"])
    groups = pd.read_csv(GROUPS / "groups.tsv", sep="\t")
    missing = groups.loc[~groups.probe_id.isin(rows.probe_id), "probe_id"]
    conflicts = pd.read_csv(TISSUE / "epic_mapping_conflicts.tsv", sep="\t")
    parts = []
    for target in TARGETS:
        obs = pd.read_parquet(PGP_OBS / f"{target}_observations.parquet",
                              columns=["probe_id", "sample_id", "m", "observed", "A2H", "m2", "m3"])
        parts.append(obs.loc[obs.probe_id.isin(missing)].assign(target=target))
    extra = pd.concat(parts).rename(columns={"sample_id": "individual", "m": "array_beta", "observed": "wgbs_observed_depth_ge10",
                                             "A2H": "A2H_deployed", "m2": "beta_squared", "m3": "beta_cubed"})
    extra["reason_outside_audited_rows"] = np.where(extra.probe_id.isin(conflicts.probe_id),
                                                    "coordinate conflict (EPIC manifest vs A2H probe map)",
                                                    "absent from audited one-to-one EPIC universe")
    check("R1-4 extra probes: 8 probes x 10 individuals x 2 targets", 160, len(extra), len(extra) == 160)
    write_table(extra[["target", "probe_id", "individual", "array_beta", "beta_squared", "beta_cubed",
                       "wgbs_observed_depth_ge10", "A2H_deployed", "reason_outside_audited_rows"]],
                "cohort/PGPUK/R1-4/R1-4_old_grouping_extra_probes_outside_audited_rows.tsv")
    for name in ("groups.tsv", "cpg_beta_stats.tsv", "group_metrics.tsv", "group_quality.tsv", "group_quality_summary.tsv",
                 "summary.tsv", "flow.tsv", "eligibility.tsv"):
        copy_file(GROUPS / name, f"cohort/PGPUK/R1-4/old_published/{name}")
    # Memberships and sample sheets.
    atlas = pd.read_csv(SAMPLES, sep="\t")
    write_table(atlas, "tables/atlas205_samples.tsv")
    copy_file(TISSUE / "reference_membership.tsv", "tables/R1-5_reference_membership.tsv")
    copy_file(TISSUE / "reference_models.tsv", "tables/R1-5_reference_models.tsv")
    draws = pd.read_csv(E1 / "draws.tsv", sep="\t")
    member = pd.concat([draws.loc[draws["rank"] <= n].assign(n_reference=n) for n in (75, 100, 150)])
    write_table(member[["n_reference", "replicate", "seed", "rank", "sample_id", "Group", "donor_id"]]
                .sort_values(["n_reference", "replicate", "rank"]), "tables/E1_subset_membership.tsv")
    same = 0
    for size, rep in E1_RUNS:
        done = json.loads((E1 / "runs" / f"N{size}_rep{rep:02d}" / "completion.json").read_text())
        mine = member.loc[(member.n_reference == size) & (member.replicate == rep), "sample_id"]
        same += sorted(done["training_ids"]) == sorted(mine)
    check("E1: subset memberships equal the runs' recorded training IDs", 60, same, same == 60)
    write_table(selected39(), "tables/atlas39_samples.tsv")
    folds = pd.read_csv(SPLITS, sep="\t").query("branch == 'historical_sample' and role == 'test'")
    write_table(folds[["repeat", "fold", "split_index", "sample_id", "donor_id"]], "tables/atlas205_cv_historical_sample_folds.tsv")
    # Published summaries shipped for reference and checking.
    for f in ("condition_summary.tsv", "paired_epic_deltas.tsv", "qc_runs.tsv", "realized_epic_depth.tsv"):
        copy_file(E2E3_SUMMARY / f, f"published/R2-2_E2_E3/{f}")
    copy_file(PGPUK_C, "published/R2-2_E3_PGPUK/pgpuk_c.tsv")
    for f in ("compare_e2.tsv", "compare_e3.tsv", "diagnostic_bias_spread.tsv"):
        copy_file(REFIT.parent / f, f"published/R2-2_refit39/{f}")
    for f in ("summary_common_cells.tsv", "summary_paired_with_native.tsv", "per_sample_common_cells.tsv"):
        copy_file(ATLAS39 / f, f"published/R2-2_atlas39_leave_donor_out/{f}")
    for f in ("E1_reference_size_metrics.tsv", "E1_fit_stability_summary.tsv", "E1_replicate_spread.tsv"):
        copy_file(E1 / "summary" / f, f"published/R2-2_E1/{f}")
    for f in ("E4_by_reference_support.tsv", "E4_site_distribution.tsv", "E4_overall_oof_metrics.tsv"):
        copy_file(S / "E4/summary" / f, f"published/R2-3_E4/{f}")
    e4 = pd.read_parquet(S / "E4/summary/E4_per_site_oof_metrics.parquet")
    write_table(e4, "published/R2-3_E4/E4_per_site_oof_metrics.tsv.gz")
    fu = pd.read_parquet(Path("results/fit_uncertainty_original_grouped_split000/full/fit_prediction_intervals.parquet"))
    write_table(fu, "published/R2-3_fit_uncertainty/donor_bootstrap_fit_prediction_intervals.tsv")
    record("extra_tables")


# ----------------------------------------------------------------------------------------------
# Assemble: one standalone zip per response, from staged files + docs
# ----------------------------------------------------------------------------------------------

PACKAGES = {
    "1.5": ("A2H_Resp1.5-R1-4_per-individual-unfiltered-matrices_PGPUK_GSE240482_20261023",
            ["cohort/*/base/", "cohort/*/masks/", "cohort/*/R1-4/"], [],
            ["cohort_", "extra_tables", "check_R1-4"], {"GROUPING_SPEC.md": "GROUPING_SPEC.md", "make_groups.py": "code/make_groups.py"}),
    "1.6": ("A2H_Resp1.6-R1-5_per-individual-unfiltered-matrices_PGPUK_GSE240482_20261023",
            ["cohort/*/base/", "cohort/*/masks/", "cohort/*/R1-5/"],
            ["tables/R1-5_reference_membership.tsv", "tables/R1-5_reference_models.tsv", "tables/atlas205_samples.tsv"],
            ["cohort_", "extra_tables", "check_R1-5"], {}),
    # 2026-10-24 (user): 2.3 and 2.4 ship only what the response uses: PGP-UK (no GSE240482); 2.3 without the
    # atlas39 leave-donor-out analysis and archive values. The 2026-10-23 versions are in packages/superseded/.
    "2.3": ("A2H_Resp2.3-R2-2_per-individual-unfiltered-matrices_PGPUK_atlas39_20261024",
            ["cohort/PGPUK/base/", "cohort/PGPUK/masks/", "cohort/PGPUK/E1/", "cohort/PGPUK/refit39/",
             "cohort/PGPUK/E3_readlength_truth/", "atlas39/calls/", "published/R2-2_E1/", "published/R2-2_E2_E3/",
             "published/R2-2_E3_PGPUK/", "published/R2-2_refit39/"],
            ["tables/E1_subset_membership.tsv", "tables/atlas39_samples.tsv", "tables/atlas205_samples.tsv"],
            ["cohort_PGPUK", "e1_", "refit39_", "e3_pgpuk", "atlas39_calls_e", "extra_tables"], {}),
    "2.4": ("A2H_Resp2.4-R2-3_per-individual-unfiltered-matrices_PGPUK_atlas205_20261024",
            ["cohort/PGPUK/base/", "cohort/PGPUK/masks/", "cohort/PGPUK/E1/", "atlas205/", "published/R2-3_", "published/R2-2_E1/"],
            ["tables/E1_subset_membership.tsv", "tables/atlas205_samples.tsv", "tables/atlas205_cv_historical_sample_folds.tsv"],
            ["cohort_PGPUK", "e1_", "atlas205_", "oof_", "check_E4", "extra_tables"], {}),
}
COMMON = {"1.5": "README_common.md", "1.6": "README_common.md", "2.3": "README_common_PGPUK.md", "2.4": "README_common_PGPUK.md"}


def staged_files():
    files = {}
    for path in sorted((OUT / "stage_records").glob("*.json")):
        for f in json.loads(path.read_text())["files"]:
            if f["file"] in files:
                raise ValueError(f"File staged twice: {f['file']}")
            files[f["file"]] = f
    return files


def arcname(rel):
    return rel[len("cohort/"):] if rel.startswith("cohort/") else rel


def stage_assemble(index):
    import fnmatch
    key = list(PACKAGES)[index]
    name, prefixes, extras, record_prefixes, docs = PACKAGES[key]
    files = staged_files()
    chosen = [rel for rel in files if any(fnmatch.fnmatch(rel, p + "*") for p in prefixes) or rel in extras]
    records = [json.loads(p.read_text()) for p in sorted((OUT / "stage_records").glob("*.json"))
               if any(p.stem.startswith(r) for r in record_prefixes)]
    checks = pd.DataFrame([{"stage": r["stage"], **c} for r in records for c in r["checks"]])
    if checks.empty or not checks["pass"].all():
        raise SystemExit("Missing or failed checks for this package")
    ZIPS.mkdir(parents=True, exist_ok=True)
    target = ZIPS / f"{name}.zip"
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(".zip.partial")
    manifest = []
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        def add(source, arc):
            compress = zipfile.ZIP_STORED if arc.endswith(".gz") else zipfile.ZIP_DEFLATED
            z.write(source, f"{name}/{arc}", compress_type=compress)
            manifest.append({"file": arc, "bytes": Path(source).stat().st_size, "sha256": file_hash(Path(source))})
        readme = (DOCS / f"README_Resp{key}.md").read_text().replace("__COMMON__", (DOCS / COMMON[key]).read_text())
        import re
        if re.search(r"__[A-Z_]+__", readme) or any(re.search(r"__[A-Z_]+__", (DOCS / s).read_text()) for s in docs if s.endswith(".md")):
            raise ValueError("Unfilled placeholder in package docs")
        merged = OUT / f"README_Resp{key}.md"
        merged.write_text(readme)
        add(merged, "README.md")
        for source, arc in docs.items():
            add(DOCS / source, arc)
        for rel in chosen:
            if file_hash(STAGE / rel) != files[rel]["sha256"]:
                raise ValueError(f"Staged file changed: {rel}")
            add(STAGE / rel, arcname(rel))
        checks_path = OUT / f"CHECKS_Resp{key}.tsv"
        checks.to_csv(checks_path, sep="\t", index=False)
        add(checks_path, "CHECKS.tsv")
        manifest_path = OUT / f"MANIFEST_Resp{key}.tsv"
        pd.DataFrame(manifest).to_csv(manifest_path, sep="\t", index=False)
        z.write(manifest_path, f"{name}/MANIFEST.tsv")
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"Zip CRC failure: {bad}")
        n = len(z.namelist())
    tmp.rename(target)
    digest = file_hash(target)
    (ZIPS / f"{name}.zip.sha256").write_text(f"{digest}  {target.name}\n")
    summary = {"package": key, "zip": str(target), "sha256": digest, "bytes": target.stat().st_size, "members": n,
               "checks": len(checks), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"assemble_Resp{key}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["cohort", "e1", "refit39", "e3pg", "atlas39_calls", "atlas39_ldo",
                                           "atlas205_truth", "oof", "extra", "check", "assemble"])
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    actions = {"cohort": stage_cohort, "e1": stage_e1, "refit39": stage_refit39, "atlas39_calls": stage_atlas39_calls,
               "atlas39_ldo": stage_atlas39_ldo, "atlas205_truth": stage_atlas205_truth, "oof": stage_oof,
               "check": stage_check, "assemble": stage_assemble}
    if args.action == "e3pg":
        stage_e3pg()
    elif args.action == "extra":
        stage_extra()
    else:
        actions[args.action](args.index)


if __name__ == "__main__":
    main()
