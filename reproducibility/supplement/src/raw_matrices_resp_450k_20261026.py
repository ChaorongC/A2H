"""Resp 1.5 / 1.6 / 2.4: complete PGP-UK matrices over all 450K probes (2026-10-26).

User request: rebuild the PGP-UK matrices without splitting EPIC and 450K CpGs, and add a supplement listing
the ~12,500 CpGs of the collaborator's list that were missing (450K-only probes). Rows are the 390,266
audited-EPIC rows of the 2026-10-23 packages plus every 450K-only probe on the PGP-UK arrays with a one-to-one
probe-locus mapping in A2H's HM450 map. EPIC probes excluded by the coordinate audit stay excluded (user rule).

Nothing is filtered (masks only), as in src/raw_matrices_resp_20261023.py, whose helpers are reused with
STAGE/OUT pointed at this build. New fits for the 450K-only loci use the unchanged fitting code and training
sets; each refit task first refits stored EPIC loci and requires exact agreement (Intel AVX-512 nodes).
Run from R/: python -m src.raw_matrices_resp_450k_20261026 <action> [--index i]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import src.raw_matrices_resp_20261023 as rm
from src.evaluate_pgpuk import file_hash, map_array_samples
from src.fit_atlas_fold import fit_sites
from src.gse128734_external import read_truth
from src.r2_2_common import fit_compatible
from src.run_atlas_cv_split import shard_arrays
from src.tissue_only_transfer import members, predict_coefficients

ROOT = Path("work/a2h_raw_matrices_450k_20261026")
OLD_STAGE, OLD_OUT = rm.STAGE, rm.OUT
STAGE, ZIPS = ROOT / "staging", ROOT / "packages"
OUT = Path("results/raw_matrices_resp_450k_20261026")
DOCS = Path("collaborator_figure_data_20261017/raw_matrices_resp_450k_20261026")
rm.STAGE, rm.OUT = STAGE, OUT  # reuse write_matrix / copy_file / write_table / check / record
SRC = Path("external_validation")
V2 = SRC / "results/paired_validation_v2"
PROBE_MAP = Path("legacy_model/A2H_cpg_probe.hg38.tsv")
EPIC_MANIFEST = Path("reference/EPIC.hg38.manifest.gencode.v36.tsv")
COLLAB = Path("work/a2h_raw_matrices_20261023/collab_list_R1.5_1.6_198064cpgs_20261026.csv")
TARGETS, ARMS, E1_RUNS = rm.TARGETS, rm.ARMS, rm.E1_RUNS
# E1 runs whose original jobs ran on AMD EPYC nodes (sacct NodeList of each run's completion.json job:
# N75_rep00 58021822, N100_rep00 58021823, N150_rep19 58021820 on hpc3-l18-01; N75_rep19 58021879 on
# hpc3-19-17). Fits are CPU-dependent in the last bits. AMD nodes were unavailable (2026-10-26: 16 of 17
# allocated), so all refits run on Intel AVX-512 nodes: exact known-answer checks for the other 56 runs and the
# R1-5 arms; for these 4 runs coefficients must agree within 1e-8 with identical fit_status (max diff recorded).
AMD_RUNS = {(75, 0), (100, 0), (150, 19), (75, 19)}
AMD_TOL = 1e-8
PG = "cohort/PGPUK"


def record(name):
    """rm.record, but also stamping this module's hash."""
    rm.record(name)
    path = OUT / "stage_records" / f"{name}.json"
    rec = json.loads(path.read_text())
    rec["helper_code_sha256"] = rec.pop("code_sha256")
    rec["code_sha256"] = file_hash(Path(__file__))
    path.write_text(json.dumps(rec, indent=2, default=str) + "\n")


def rows_table():
    return pd.read_csv(STAGE / "internal/rows.tsv", sep="\t")


def pairs():
    key = pd.read_csv(rm.PKG311 / "PGPUK/pairing_key.tsv", sep="\t")
    return key


# ----------------------------------------------------------------------------------------------
# rows: full 450K row universe, row annotation, collaborator supplement
# ----------------------------------------------------------------------------------------------

def stage_rows():
    old, ids, _ = rm.cohort("PGPUK", ["array_beta"])
    beta_ids = pd.read_csv(V2 / "array/pgpuk_450K.beta.tsv", sep="\t", usecols=[0]).iloc[:, 0]
    pm = pd.read_csv(PROBE_MAP, sep="\t", usecols=["loci.hg38", "HM450.hg38"]).dropna()
    pm = pm.rename(columns={"HM450.hg38": "probe_id", "loci.hg38": "locus"})
    epic = set(pd.read_csv(EPIC_MANIFEST, sep="\t", usecols=["probeID"]).probeID)
    # The map repeats a probe/locus pair once per EPIC v2 replicate probe; count distinct loci and probes.
    pm = pm.drop_duplicates(["probe_id", "locus"])
    cand = pm.loc[pm.probe_id.isin(beta_ids) & ~pm.probe_id.isin(epic)].copy()
    multi_locus = cand.probe_id.map(pm.groupby("probe_id").locus.nunique()) > 1
    shared_locus = cand.locus.map(pm.groupby("locus").probe_id.nunique()) > 1
    clash = cand.locus.isin(old.locus)
    cand["ambiguous"] = multi_locus | shared_locus | clash
    new = cand.loc[~cand.ambiguous, ["probe_id", "locus"]].drop_duplicates()
    rows = pd.concat([old.assign(row_set="EPIC_audited"), new.assign(row_set="450K_only")])
    rows = rows.sort_values("probe_id").reset_index(drop=True)
    rows["in_A2H_probe_map"] = rows.probe_id.isin(pm.probe_id)
    rm.check("rows: probe IDs and loci one-to-one", True, True, not rows.probe_id.duplicated().any() and not rows.locus.duplicated().any())
    rm.check("rows: 2026-10-23 EPIC rows kept with identical loci", len(old), int((rows.row_set == "EPIC_audited").sum()),
             old.merge(rows, on=["probe_id", "locus"]).shape[0] == len(old))
    rm.check("rows: 450K-only rows added (ambiguous probe-locus mappings excluded)",
             f"{cand.probe_id.nunique()} candidates", f"{len(new)} added, {int(cand.loc[cand.ambiguous].probe_id.nunique())} ambiguous", True)
    path = STAGE / "internal/rows.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, sep="\t", index=False)
    rm.write_table(rows[["probe_id", "locus", "row_set", "in_A2H_probe_map"]], f"{PG}/row_annotation.tsv")
    # Supplement: the collaborator's CpGs that were missing from the 2026-10-23 packages.
    lst = pd.read_csv(COLLAB, encoding="utf-8-sig")
    lst_cpg = lst.drop_duplicates("cpgID")[["cpgID", "locus"]].rename(columns={"cpgID": "probe_id", "locus": "locus_in_list"})
    miss = lst_cpg.loc[~lst_cpg.probe_id.isin(old.probe_id)].copy()
    audit = pd.read_csv(rm.TISSUE / "epic_mapping_audit.tsv", sep="\t").drop_duplicates("probe_id").set_index("probe_id")
    on_epic = miss.probe_id.isin(epic)
    miss["platform"] = np.where(on_epic, "EPIC_and_450K", "450K_only")
    miss["locus_in_matrices"] = miss.probe_id.map(rows.set_index("probe_id").locus)
    miss["included_now"] = miss.probe_id.isin(rows.probe_id)
    status = np.where(miss.included_now, "450K-only probe; added as a row in this build",
             np.where(~on_epic, "450K-only probe; excluded: ambiguous probe-locus mapping",
             np.where(miss.probe_id.isin(audit.index[~audit.eligible_mapping.astype(bool)]),
                      "EPIC probe; excluded by the coordinate audit (EPIC manifest vs A2H probe map)",
                      "EPIC probe; not in the audited one-to-one EPIC universe")))
    miss["status"] = status
    rm.write_table(miss[["probe_id", "locus_in_list", "locus_in_matrices", "platform", "included_now", "status"]]
                   .sort_values(["included_now", "probe_id"], ascending=[False, True]),
                   f"{PG}/supplement_collaborator_list_CpGs_missing_on_20261023.tsv")
    added = miss.loc[miss.included_now]
    rm.check("supplement: collaborator CpGs missing on 2026-10-23", 12588, len(miss), len(miss) == 12588)
    rm.check("supplement: missing 450K-only CpGs now added; locus equals the list's",
             int((~on_epic).sum()), f"{len(added)} added, {int((added.locus_in_list == added.locus_in_matrices).sum())} loci agree",
             (added.locus_in_list == added.locus_in_matrices).all())
    record("rows")


# ----------------------------------------------------------------------------------------------
# wgbs: per-participant WGBS values for the new rows (+ a known-answer sample of old rows)
# ----------------------------------------------------------------------------------------------

def stage_wgbs(index):
    key = pairs().iloc[index]
    rows = rows_table()
    rng = np.random.default_rng(20261026)
    old = rows.loc[rows.row_set == "EPIC_audited"]
    sample = old.iloc[np.sort(rng.choice(len(old), 2000, replace=False))]
    target = pd.concat([rows.loc[rows.row_set == "450K_only"], sample])
    result = target[["probe_id", "locus", "row_set"]].reset_index(drop=True)
    for k, path in (("beta", V2 / f"wgbs/traditional/{key.wgbs_id}_methRatio.txt"), ("CHALM", V2 / f"wgbs/chalm/{key.wgbs_id}_CHALM.txt")):
        table = read_truth(path, result).reindex(result.locus)
        result[k] = table.ratio.to_numpy(np.float64)
        result[f"{k}_depth"] = table.CT_count.to_numpy(np.float64)
    dest = STAGE / f"internal/wgbs/{key.sample_id}.parquet"
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(dest, index=False)
    print(f"{key.sample_id}: {int(np.isfinite(result.CHALM).sum())} CHALM values", flush=True)


# ----------------------------------------------------------------------------------------------
# base: base matrices, masks, R1-4 comparators for all rows
# ----------------------------------------------------------------------------------------------

def wide(path, probes, cols):
    return pd.read_csv(path, sep="\t", index_col=0).reindex(index=probes, columns=cols).to_numpy(np.float64)


def stage_base():
    rows = rows_table()
    probes = rows.probe_id.to_numpy()
    old_rows, ids, old = rm.cohort("PGPUK")
    is_old = (rows.row_set == "EPIC_audited").to_numpy()
    pos_old = pd.Index(old_rows.probe_id).get_indexer(rows.probe_id[is_old])
    idat = pd.read_csv(SRC / "metadata/paired_validation_v2/paired_array_idat_manifest.tsv", sep="\t")
    m = {"array_beta": map_array_samples(pd.read_csv(V2 / "array/pgpuk_450K.beta.tsv", sep="\t", index_col=0), idat, ids)
         .reindex(probes).to_numpy(np.float64)}
    for t in TARGETS:
        m[f"A2H_{t}"] = wide(V2 / f"a2h_array/pgpuk_450K.{t.lower()}.tsv", probes, ids)
    m["A2H_input_support"] = wide(V2 / "a2h_array/pgpuk_450K.confidence.tsv", probes, ids)
    for k in ("array_beta", "A2H_CHALM", "A2H_CAMDA", "A2H_input_support"):
        same = np.array_equal(m[k][is_old], old[k][pos_old], equal_nan=True)
        rm.check(f"base {k}: values from source equal the 2026-10-23 matrices at every EPIC row", True, same, same)
    for k in ("wgbs_beta", "wgbs_beta_depth", "wgbs_CHALM", "wgbs_CHALM_depth"):
        m[k] = np.full((len(rows), len(ids)), np.nan)
        m[k][is_old] = old[k][pos_old]
    newpos = pd.Series(np.arange(len(rows)), index=rows.probe_id)
    for j, s in enumerate(ids):
        w = pd.read_parquet(STAGE / f"internal/wgbs/{s}.parquet")
        ka = w.loc[w.row_set == "EPIC_audited"]
        r = newpos.loc[ka.probe_id].to_numpy()
        for k, col in (("wgbs_beta", "beta"), ("wgbs_beta_depth", "beta_depth"), ("wgbs_CHALM", "CHALM"), ("wgbs_CHALM_depth", "CHALM_depth")):
            same = np.array_equal(ka[col].to_numpy(np.float64), m[k][r, j], equal_nan=True)
            if not same:
                rm.check(f"WGBS parser known answer {s} {k} (2,000 EPIC rows)", True, same, same)
        nw = w.loc[w.row_set == "450K_only"]
        r = newpos.loc[nw.probe_id].to_numpy()
        for k, col in (("wgbs_beta", "beta"), ("wgbs_beta_depth", "beta_depth"), ("wgbs_CHALM", "CHALM"), ("wgbs_CHALM_depth", "CHALM_depth")):
            m[k][r, j] = nw[col].to_numpy(np.float64)
    rm.check("WGBS parser known answer: 2,000 EPIC rows x 10 participants equal the 2026-10-23 values", True, True,
             not any("WGBS parser known answer" in c["item"] for c in rm.CHECKS))
    m["wgbs_CAMDA"] = m["wgbs_CHALM"] - m["wgbs_beta"]
    same = np.array_equal(m["wgbs_CAMDA"][is_old], old["wgbs_CAMDA"][pos_old], equal_nan=True)
    rm.check("base wgbs_CAMDA equals the 2026-10-23 matrix at every EPIC row", True, same, same)
    for k in rm.BASE:
        rm.write_matrix(f"{PG}/base/{k}.tsv.gz", rows, ids, m[k], integer=k.endswith("_depth"))
    rm.copy_file(rm.PKG311 / "PGPUK/pairing_key.tsv", f"{PG}/base/pairing_key.tsv")
    # Masks.
    same = np.array_equal(m["wgbs_beta_depth"], m["wgbs_CHALM_depth"], equal_nan=True)
    rm.check("wgbs_beta_depth identical to wgbs_CHALM_depth (all rows)", True, same, same)
    d10 = np.nan_to_num(m["wgbs_CHALM_depth"], nan=0) >= 10
    support = np.nan_to_num(m["A2H_input_support"], nan=-1) >= 0.05
    dbeta = d10 & np.isfinite(m["array_beta"]) & np.isfinite(m["wgbs_beta"]) & (np.abs(m["array_beta"] - m["wgbs_beta"]) <= 0.05)
    epic_rows = np.repeat(is_old[:, None], len(ids), axis=1)
    for name, v in (("mask_wgbs_depth_ge10", d10), ("mask_A2H_support_ge0.05", support), ("mask_dbeta_le0.05", dbeta),
                    ("mask_rows_EPIC_audited", epic_rows)):
        rm.write_matrix(f"{PG}/masks/{name}.tsv.gz", rows, ids, v.astype(np.int8), integer=True)
    locked_all = 0
    for t in TARGETS:
        cell = d10 & np.isfinite(m[f"wgbs_{t}"]) & np.isfinite(m[f"A2H_{t}"]) & np.isfinite(m["array_beta"]) & support
        complete = (cell.sum(axis=1) >= 8)[:, None] & np.ones_like(cell)
        primary = cell & complete
        rm.write_matrix(f"{PG}/masks/mask_cpg_ge8_participants_{t}.tsv.gz", rows, ids, complete.astype(np.int8), integer=True)
        rm.write_matrix(f"{PG}/masks/mask_primary_external_{t}.tsv.gz", rows, ids, primary.astype(np.int8), integer=True)
        rm.write_matrix(f"{PG}/masks/mask_R1-4_published_universe_{t}.tsv.gz", rows, ids, (primary & epic_rows).astype(np.int8), integer=True)
        rm.check(f"{t}: primary mask on EPIC rows = published 3.11 cells", 1214068, int((primary & epic_rows).sum()), int((primary & epic_rows).sum()) == 1214068)
        rm.check(f"{t}: dbeta within primary on EPIC rows = 337,845", 337845, int((primary & epic_rows & dbeta).sum()),
                 int((primary & epic_rows & dbeta).sum()) == 337845)
        # Locked primary branch (all 450K evaluation CpGs): same cells and values on the rows here.
        obs = pd.read_parquet(rm.PGP_OBS / f"{t}_observations.parquet", columns=["probe_id", "sample_id", "observed", "A2H", "m"])
        inside = obs.probe_id.isin(rows.probe_id)
        locked_all = len(obs)
        o = obs.loc[inside]
        mask, _, r, c, keep = rm.cells_mask(rows, ids, o.probe_id, o.sample_id)
        rm.check(f"{t}: locked primary-branch cells on these rows equal the primary mask on all rows",
                 f"{len(o)} locked cells ({int((~inside).sum())} on probes outside the rows)", int(primary.sum()),
                 np.array_equal(mask.astype(bool), primary))
        r = r[keep].astype(int)
        for col, k in (("observed", f"wgbs_{t}"), ("A2H", f"A2H_{t}"), ("m", "array_beta")):
            tol = 1e-15 if (t == "CAMDA" and col == "observed") else 0.0
            diff = float(np.abs(o[col].to_numpy() - m[k][r, c[keep]]).max())
            rm.check(f"{t}: locked {col} values reproduced (tol {tol:g})", 0.0, diff, diff <= tol)
    # R1-4 comparators and the old membership mask.
    for k, power in (("beta_squared", 2), ("beta_cubed", 3)):
        rm.write_matrix(f"{PG}/R1-4/{k}.tsv.gz", rows, ids, m["array_beta"] ** power)
    groups = pd.read_csv(rm.GROUPS / "groups.tsv", sep="\t")
    grouped = rows.probe_id.isin(groups.probe_id).to_numpy()[:, None] & np.ones((len(rows), len(ids)), bool)
    rm.write_matrix(f"{PG}/R1-4/mask_R1-4_old_grouped_cpgs.tsv.gz", rows, ids, grouped.astype(np.int8), integer=True)
    # Missingness summary.
    summary = []
    for j, s in enumerate(ids):
        row = {"individual": s, "n_rows": len(rows), "n_rows_EPIC_audited": int(is_old.sum()), "n_rows_450K_only": int((~is_old).sum())}
        for k in rm.BASE:
            row[f"finite_{k}"] = int(np.isfinite(m[k][:, j]).sum())
        for t in TARGETS:
            usable = np.isfinite(m[f"wgbs_{t}"][:, j]) & np.isfinite(m[f"A2H_{t}"][:, j])
            row[f"{t}_pairs_any_depth"] = int(usable.sum())
            row[f"{t}_pairs_depth_ge10"] = int((usable & d10[:, j]).sum())
            row[f"{t}_pairs_depth_ge10_support_ge0.05"] = int((usable & d10[:, j] & support[:, j]).sum())
        summary.append(row)
    rm.write_table(pd.DataFrame(summary), f"{PG}/base/missingness_summary.tsv")
    # Collaborator list: every listed pair on these rows has inputs, truth and A2H.
    lst = pd.read_csv(COLLAB, encoding="utf-8-sig")
    on = lst.loc[lst.cpgID.isin(rows.probe_id)]
    _, _, r, c, keep = rm.cells_mask(rows, ids, on.cpgID, on["sample"])
    r = r[keep].astype(int)
    ok = np.logical_and.reduce([np.isfinite(m[k][r, c[keep]]) for k in ("array_beta", "wgbs_CHALM", "wgbs_CAMDA", "A2H_CHALM", "A2H_CAMDA")])
    rm.check("collaborator list: CpGs on these rows / pairs with array beta, WGBS CHALM+CAMDA and A2H CHALM+CAMDA",
             f"{lst.cpgID.nunique()} CpGs / {len(lst)} pairs", f"{on.cpgID.nunique()} CpGs / {int(ok.sum())} of {len(on)} pairs complete", ok.all())
    record("base")


# ----------------------------------------------------------------------------------------------
# refit: E1 subsets and R1-5 reduced references at the 450K-only loci, per atlas shard
# ----------------------------------------------------------------------------------------------

def cpu_vendor():
    text = Path("/proc/cpuinfo").read_text()
    return "AMD" if "AuthenticAMD" in text else ("Intel_avx512" if "GenuineIntel" in text and "avx512f" in text else "other")


def stage_refit(index):
    shard_index, arch = index, "Intel_avx512"
    if cpu_vendor() != arch:
        raise SystemExit(f"refit task {index} needs a {arch} node, got {cpu_vendor()}")
    runs = E1_RUNS
    amd_max = 0.0
    rows = rows_table()
    shard = pd.read_csv(rm.SHARDS, sep="\t").set_index("shard_index").loc[shard_index]
    loci = pd.read_parquet(shard.path, columns=["locus"]).locus
    new = loci.loc[loci.isin(rows.loc[rows.row_set == "450K_only", "locus"])]
    e1map = pd.read_csv(rm.E1 / "epic_mapping.tsv", sep="\t").query("shard_index == @shard_index")
    tmap = pd.read_csv(rm.TISSUE / "epic_mapping.tsv", sep="\t").query("shard_index == @shard_index")
    ka = loci.loc[loci.isin(e1map.locus) & loci.isin(tmap.locus)].iloc[:50]
    frame = pd.read_parquet(shard.path)
    frame = frame.loc[frame.locus.isin(set(new) | set(ka))].reset_index(drop=True)
    is_new = frame.locus.isin(new).to_numpy()
    is_ka = frame.locus.isin(ka).to_numpy()
    out = STAGE / "internal/refit450k"
    out.mkdir(parents=True, exist_ok=True)
    draws = pd.read_csv(rm.E1 / "draws.tsv", sep="\t")
    cols = ["a2h_intercept", "a2h_linear", "a2h_quadratic"]
    bad = {"E1": 0, "R1-5": 0}
    for target in TARGETS:
        parts = []
        for size, rep in runs:
            train = draws.loc[(draws.replicate == rep) & (draws["rank"] <= size), "sample_id"].tolist()
            x = frame[[f"m__{s}" for s in train]].to_numpy(dtype=float)
            y = frame[[f"{target}__{s}" for s in train]].to_numpy(dtype=float)
            q = fit_compatible(x, y, min_train=50)
            q.insert(0, "locus", frame.locus.to_numpy())
            stored = pd.read_parquet(rm.E1 / "runs" / f"N{size}_rep{rep:02d}" / f"{target}_quality.parquet",
                                     columns=["locus", "fit_status"] + cols).set_index("locus").loc[frame.locus[is_ka]]
            mine = q.loc[is_ka].set_index("locus")
            same_status = (mine.fit_status.to_numpy() == stored.fit_status.to_numpy()).all()
            if (size, rep) in AMD_RUNS:
                d = np.abs(mine[cols].to_numpy() - stored[cols].to_numpy())
                d = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
                amd_max = max(amd_max, d)
                same_nan = np.array_equal(np.isnan(mine[cols].to_numpy()), np.isnan(stored[cols].to_numpy()))
                bad["E1_AMD"] = bad.get("E1_AMD", 0) + (not (same_status and same_nan and d <= AMD_TOL))
            else:
                bad["E1"] += not (np.array_equal(mine[cols].to_numpy(), stored[cols].to_numpy(), equal_nan=True) and same_status)
            parts.append(q.loc[is_new, ["locus", "fit_status", "n_train_joint"] + cols].assign(n_reference=size, replicate=rep))
        pd.concat(parts).to_parquet(out / f"E1_{target}_shard{shard_index:04d}.parquet", index=False)
        samples = pd.read_csv(rm.TISSUE / "atlas_samples.tsv", sep="\t")
        manifest = pd.read_csv(rm.TISSUE / "reference_membership.tsv", sep="\t")
        parts = []
        for model in ARMS:
            train, test = members(manifest, samples, model)
            x, y, xt, _ = shard_arrays(frame, target, train, test)
            quality, _ = fit_sites(x, y, xt, min_train=50, include_coefficients=True)
            quality.insert(0, "locus", frame.locus.to_numpy())
            ccols = cols + ["reference_linear_intercept", "reference_linear_slope"]
            stored = pd.read_parquet(rm.TISSUE / "fits" / model / f"shard_{shard_index:04d}" / f"{target}_coefficients.parquet",
                                     columns=["locus", "fit_status"] + ccols).set_index("locus").loc[frame.locus[is_ka]]
            mine = quality.loc[is_ka].set_index("locus")
            bad["R1-5"] += not (np.array_equal(mine[ccols].to_numpy(), stored[ccols].to_numpy(), equal_nan=True)
                                and (mine.fit_status.to_numpy() == stored.fit_status.to_numpy()).all())
            parts.append(quality.loc[is_new, ["locus", "fit_status", "n_train_joint"] + ccols].assign(model=model))
        pd.concat(parts).to_parquet(out / f"R1-5_{target}_shard{shard_index:04d}.parquet", index=False)
    rm.check(f"refit shard {shard_index}: E1 fits at {int(is_ka.sum())} stored EPIC loci reproduced exactly (56 Intel-origin runs x 2 targets)",
             0, bad["E1"], bad["E1"] == 0)
    rm.check(f"refit shard {shard_index}: E1 fits of the 4 AMD-origin runs within {AMD_TOL:g}, same fit_status (x 2 targets)",
             0, f"{bad.get('E1_AMD', 0)} failing; max |coef diff| {amd_max:.3g}", bad.get("E1_AMD", 0) == 0)
    rm.check(f"refit shard {shard_index}: R1-5 arm fits at {int(is_ka.sum())} stored EPIC loci reproduced exactly (2 arms x 2 targets)",
             0, bad["R1-5"], bad["R1-5"] == 0)
    rm.check(f"refit shard {shard_index}: 450K-only loci fitted", "-", int(is_new.sum()), True)
    record(f"refit_shard{shard_index:04d}")


def refit_table(kind, target):
    return pd.concat([pd.read_parquet(p) for p in sorted((STAGE / "internal/refit450k").glob(f"{kind}_{target}_shard*.parquet"))])


# ----------------------------------------------------------------------------------------------
# r15 and e1: predictions for all rows
# ----------------------------------------------------------------------------------------------

def beta_matrix():
    rows = rows_table()
    f = rm.read_matrix(STAGE / f"{PG}/base/array_beta.tsv.gz")
    ids = [c for c in f.columns if c not in ("probe_id", "locus")]
    if not f.probe_id.equals(rows.probe_id):
        raise ValueError("Row order differs")
    return rows, ids, f[ids].to_numpy(np.float64)


def coefficients(rows, epic_table, new_table, columns):
    """EPIC rows from the stored table, 450K-only rows from the new fits; reindexed to rows."""
    is_new = (rows.row_set == "450K_only").to_numpy()
    q = rm.by_locus(epic_table, rows, columns)
    n = rm.by_locus(new_table, rows, columns)
    q.loc[is_new, columns] = n.loc[is_new, columns].to_numpy()
    missing = int(n.loc[is_new, columns[0]].isna().sum())
    return q, missing


def stage_r15():
    rows, ids, beta = beta_matrix()
    m = {"array_beta": beta}
    ccols = ["a2h_intercept", "a2h_linear", "a2h_quadratic", "reference_linear_intercept", "reference_linear_slope"]
    for target in TARGETS:
        m[f"A2H_{target}"] = rm.read_matrix(STAGE / f"{PG}/base/A2H_{target}.tsv.gz")[ids].to_numpy(np.float64)
        new = refit_table("R1-5", target)
        preds = {}
        for arm in ARMS:
            stored = pd.concat([pd.read_parquet(p) for p in sorted((rm.TISSUE / "fits" / arm).glob(f"shard_*/{target}_coefficients.parquet"))])
            q, _ = coefficients(rows, stored, new.loc[new.model == arm], ccols)
            p = predict_coefficients(q, beta)
            preds[f"A2H_{arm}"], preds[f"linear_{arm}"] = p["A2H"], p["reference_linear"]
        ctl = rm.by_locus(pd.read_parquet(rm.CONTROLS / f"{target}_controls.parquet"), rows, ["linear_intercept", "linear_slope"])
        preds["linear_full_reference"] = ctl.linear_intercept.to_numpy()[:, None] + ctl.linear_slope.to_numpy()[:, None] * beta
        for k in ("A2H_exclude_blood_immune", "A2H_exclude_epithelial", "linear_full_reference",
                  "linear_exclude_blood_immune", "linear_exclude_epithelial"):
            rm.write_matrix(f"{PG}/R1-5/{k}_{target}.tsv.gz", rows, ids, preds[k])
            new_rows = (rows.row_set == "450K_only").to_numpy()
            rm.check(f"R1-5 {k}_{target}: 450K-only cells with a prediction (array beta present)",
                     int(np.isfinite(beta[new_rows]).sum()), int(np.isfinite(preds[k][new_rows]).sum()), True)
        pub = pd.read_parquet(rm.TISSUE / f"summary/R1-5_{target}_common_observations.parquet")
        mask, outside, r, c, keep = rm.cells_mask(rows, ids, pub.probe_id, pub.sample_id)
        rm.write_matrix(f"{PG}/R1-5/mask_R1-5_published_{target}.tsv.gz", rows, ids, mask, integer=True)
        r = r[keep].astype(int)
        pairs_ = {"full_reference__A2H": (f"A2H_{target}", 1e-10), "full_reference__reference_linear": ("linear_full_reference", 1e-12),
                  "m": ("array_beta", 0.0)}
        for arm in ARMS:
            pairs_[f"{arm}__A2H"] = (f"A2H_{arm}", 0.0)
            pairs_[f"{arm}__reference_linear"] = (f"linear_{arm}", 0.0)
        for column, (k, tol) in pairs_.items():
            src = preds.get(k, m.get(k))
            diff = float(np.nanmax(np.abs(pub[column].to_numpy()[keep] - src[r, c[keep]])))
            rm.check(f"PGPUK {target}: R1-5 published {column} reproduced (tol {tol:g}; outside rows {outside})", 0, diff, diff <= tol)
    record("r15")


def stage_e1(index):
    size, rep = E1_RUNS[index]
    rows, ids, beta = beta_matrix()
    run = rm.E1 / "runs" / f"N{size}_rep{rep:02d}"
    cols = ["a2h_intercept", "a2h_linear", "a2h_quadratic"]
    for target in TARGETS:
        new = pd.concat([pd.read_parquet(p) for p in sorted((STAGE / "internal/refit450k").glob(f"E1_{target}_shard*.parquet"))])
        new = new.loc[(new.n_reference == size) & (new.replicate == rep)]
        q, missing = coefficients(rows, pd.read_parquet(run / f"{target}_quality.parquet"), new, cols)
        pred = (q.a2h_intercept.to_numpy()[:, None] + q.a2h_linear.to_numpy()[:, None] * beta
                + q.a2h_quadratic.to_numpy()[:, None] * beta ** 2)
        rm.write_matrix(f"{PG}/E1/{target}/A2H_N{size}_rep{rep:02d}_{target}.tsv.gz", rows, ids, pred)
        stored = pd.read_parquet(run / f"{target}_predictions.parquet")
        pos = pd.Series(np.arange(len(rows)), index=rows.probe_id).reindex(stored.probe_id).to_numpy().astype(int)
        sp = stored[[f"pred__{s}" for s in ids]].to_numpy(np.float64)
        fin = np.isfinite(sp)
        rm.check(f"E1 N{size} rep{rep:02d} {target}: stored predictions reproduced exactly", f"{int(fin.sum())} cells",
                 f"max |d| {float(np.abs(sp[fin] - pred[pos][fin]).max()):.3g}", np.array_equal(sp[fin], pred[pos][fin]))
    record(f"e1_N{size}_rep{rep:02d}")


# ----------------------------------------------------------------------------------------------
# extra (copies from the 2026-10-23 staging) and checks
# ----------------------------------------------------------------------------------------------

def stage_extra():
    for rel in sorted(p.relative_to(OLD_STAGE).as_posix() for p in (OLD_STAGE / "cohort/PGPUK/R1-4").rglob("*")
                      if p.is_file() and ("old_published" in p.as_posix() or "extra_probes" in p.name)):
        rm.copy_file(OLD_STAGE / rel, rel)
    for rel in ["tables/R1-5_reference_membership.tsv", "tables/R1-5_reference_models.tsv", "tables/atlas205_samples.tsv",
                "tables/E1_subset_membership.tsv", "tables/atlas205_cv_historical_sample_folds.tsv"]:
        rm.copy_file(OLD_STAGE / rel, rel)
    for p in sorted((OLD_STAGE / "published").rglob("*")):
        rel = p.relative_to(OLD_STAGE).as_posix()
        if p.is_file() and (rel.startswith("published/R2-3_") or rel.startswith("published/R2-2_E1/")):
            rm.copy_file(p, rel)
    rm.check("extra: copied 2026-10-23 tables unchanged (sha256 checked per file)", True, True, True)
    record("extra")


def stage_check(index):
    if index == 0:
        import importlib.util
        spec = importlib.util.spec_from_file_location("make_groups", DOCS / "make_groups.py")
        mg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mg)
        old = pd.read_csv(rm.GROUPS / "groups.tsv", sep="\t")
        old_metrics = pd.read_csv(rm.GROUPS / "group_metrics.tsv", sep="\t")
        extra = STAGE / f"{PG}/R1-4/R1-4_old_grouping_extra_probes_outside_audited_rows.tsv"
        groups, metrics, flow = mg.run(STAGE / PG, mask="mask_R1-4_published_universe", extra=extra)
        keys = ["probe_id", "n_participants", "mean_rank", "block_id", "group_id", "member_rank"]
        g = groups[old.columns].reset_index(drop=True)
        stat = float(max(np.abs(g.mean_m - old.mean_m).max(), np.abs(g.var_m - old.var_m).max()))
        rm.check("R1-4 make_groups.py (mask_R1-4_published_universe + 8 extra probes) reproduces old membership",
                 "identical", f"membership {g[keys].equals(old[keys])}, max |d| mean/var {stat:.3g}", g[keys].equals(old[keys]) and stat <= 1e-15)
        a = old_metrics.sort_values(["target", "method", "group_id"]).reset_index(drop=True)
        b = metrics.sort_values(["target", "method", "group_id"]).reset_index(drop=True)
        diff = float(max(np.nanmax(np.abs(a.rmse - b.rmse)), np.nanmax(np.abs(a.pearson_r - b.pearson_r))))
        rm.check("R1-4 make_groups.py reproduces old group_metrics (tol 1e-12)", len(a), f"{len(b)} rows, max |d| {diff:.3g}",
                 len(a) == len(b) and diff <= 1e-12)
        _, _, flow_all = mg.run(STAGE / PG, mask="mask_primary_external")
        rm.check("R1-4 make_groups.py with mask_primary_external on all 450K rows (informational)", "-",
                 f"{flow_all['eligible_cpgs']} CpGs / {flow_all['groups']} groups / {flow_all['remainder']} remainder", True)
        record("check_R1-4")
    else:
        pub = pd.read_csv(Path("collaborator_figure_data_20261017/package_v4/A2H_figure_data_20261017/"
                               "Resp1.6_R1-5_reference_composition/summary_with_participant_bootstrap.tsv"), sep="\t")
        folder = STAGE / PG
        beta = rm.read_matrix(folder / "base/array_beta.tsv.gz")
        ids = [c for c in beta.columns if c not in ("probe_id", "locus")]
        for target in TARGETS:
            mask = rm.read_matrix(folder / f"R1-5/mask_R1-5_published_{target}.tsv.gz")[ids].to_numpy() == 1
            y = rm.read_matrix(folder / f"base/wgbs_{target}.tsv.gz")[ids].to_numpy(np.float64)
            for r in pub.loc[pub.target == target].itertuples(index=False):
                rel = ("base/array_beta.tsv.gz" if r.method == "m" else
                       (f"base/A2H_{target}.tsv.gz" if r.model == "full_reference" else f"R1-5/A2H_{r.model}_{target}.tsv.gz")
                       if r.method == "A2H" else f"R1-5/linear_{r.model}_{target}.tsv.gz")
                p = rm.read_matrix(folder / rel)[ids].to_numpy(np.float64)
                valid = mask & np.isfinite(y) & np.isfinite(p)
                n = valid.sum(axis=1)
                site = np.sqrt(np.where(valid, (p - y) ** 2, 0.0).sum(axis=1)[n > 0] / n[n > 0])
                ok = int(valid.sum()) == r.n_observations and int((n > 0).sum()) == r.n_sites and abs(site.mean() - r.mean_site_rmse) < 1e-12
                rm.check(f"R1-5 PGP-UK {target} {r.model} {r.method}: published n and mean per-CpG RMSE reproduced",
                         f"{r.n_sites} / {r.n_observations} / {r.mean_site_rmse:.12g}", f"{int((n > 0).sum())} / {int(valid.sum())} / {site.mean():.12g}", ok)
        record("check_R1-5")


# ----------------------------------------------------------------------------------------------
# assemble: Resp 1.5, 1.6, 2.4 (PGP-UK from this build; GSE240482 / atlas205 from 2026-10-23)
# ----------------------------------------------------------------------------------------------

PACKAGES = {
    "1.5": ("A2H_Resp1.5-R1-4_per-individual-unfiltered-matrices_PGPUK-all450K_GSE240482_20261026",
            [f"{PG}/base/", f"{PG}/masks/", f"{PG}/R1-4/", f"{PG}/row_annotation.tsv", f"{PG}/supplement_"],
            ["cohort/GSE240482/base/", "cohort/GSE240482/masks/", "cohort/GSE240482/R1-4/"],
            ["rows", "base", "extra", "check_R1-4"], ["cohort_GSE240482"],
            {"GROUPING_SPEC.md": "GROUPING_SPEC.md", "make_groups.py": "code/make_groups.py"}),
    "1.6": ("A2H_Resp1.6-R1-5_per-individual-unfiltered-matrices_PGPUK-all450K_GSE240482_20261026",
            [f"{PG}/base/", f"{PG}/masks/", f"{PG}/R1-5/", f"{PG}/row_annotation.tsv", f"{PG}/supplement_",
             "tables/R1-5_reference_membership.tsv", "tables/R1-5_reference_models.tsv", "tables/atlas205_samples.tsv"],
            ["cohort/GSE240482/base/", "cohort/GSE240482/masks/", "cohort/GSE240482/R1-5/"],
            ["rows", "base", "refit_shard", "r15", "extra", "check_R1-5"], ["cohort_GSE240482"], {}),
    "2.4": ("A2H_Resp2.4-R2-3_per-individual-unfiltered-matrices_PGPUK-all450K_atlas205_20261026",
            [f"{PG}/base/", f"{PG}/masks/", f"{PG}/E1/", f"{PG}/row_annotation.tsv", f"{PG}/supplement_",
             "tables/E1_subset_membership.tsv", "tables/atlas205_samples.tsv", "tables/atlas205_cv_historical_sample_folds.tsv",
             "published/R2-3_", "published/R2-2_E1/"],
            ["atlas205/"],
            ["rows", "base", "refit_shard", "e1_", "extra"], ["atlas205_", "oof_", "check_E4"], {}),
}


def staged(out, stage):
    files = {}
    for path in sorted((out / "stage_records").glob("*.json")):
        for f in json.loads(path.read_text())["files"]:
            files[f["file"]] = (stage / f["file"], f["sha256"])
    return files


def stage_assemble(index):
    key = list(PACKAGES)[index]
    name, new_prefixes, old_prefixes, new_records, old_records, docs = PACKAGES[key]
    new, old = staged(OUT, STAGE), staged(OLD_OUT, OLD_STAGE)
    chosen = [(rel, *new[rel]) for rel in new if any(fnmatch.fnmatch(rel, p + "*") for p in new_prefixes)]
    chosen += [(rel, *old[rel]) for rel in old if any(fnmatch.fnmatch(rel, p + "*") for p in old_prefixes)]
    recs = [json.loads(p.read_text()) for p in sorted((OUT / "stage_records").glob("*.json")) if any(p.stem.startswith(r) for r in new_records)]
    recs += [json.loads(p.read_text()) for p in sorted((OLD_OUT / "stage_records").glob("*.json")) if any(p.stem.startswith(r) for r in old_records)]
    checks = pd.DataFrame([{"stage": r["stage"], **c} for r in recs for c in r["checks"]])
    if checks.empty or not checks["pass"].all():
        raise SystemExit("Missing or failed checks for this package")
    common = "README_common_PGPUK.md" if key == "2.4" else "README_common.md"
    readme = (DOCS / f"README_Resp{key}.md").read_text().replace("__COMMON__", (DOCS / common).read_text())
    if re.search(r"__[A-Z_]+__", readme) or any(re.search(r"__[A-Z_]+__", (DOCS / s).read_text()) for s in docs if s.endswith(".md")):
        raise ValueError("Unfilled placeholder in package docs")
    ZIPS.mkdir(parents=True, exist_ok=True)
    target = ZIPS / f"{name}.zip"
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(".zip.partial")
    manifest = []
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        def add(source, arc):
            z.write(source, f"{name}/{arc}", compress_type=zipfile.ZIP_STORED if arc.endswith(".gz") else zipfile.ZIP_DEFLATED)
            manifest.append({"file": arc, "bytes": Path(source).stat().st_size, "sha256": file_hash(Path(source))})
        merged = OUT / f"README_Resp{key}.md"
        merged.write_text(readme)
        add(merged, "README.md")
        for source, arc in docs.items():
            add(DOCS / source, arc)
        for rel, path, digest in sorted(chosen):
            if file_hash(path) != digest:
                raise ValueError(f"Staged file changed: {path}")
            add(path, rm.arcname(rel))
        cp = OUT / f"CHECKS_Resp{key}.tsv"
        checks.to_csv(cp, sep="\t", index=False)
        add(cp, "CHECKS.tsv")
        mp = OUT / f"MANIFEST_Resp{key}.tsv"
        pd.DataFrame(manifest).to_csv(mp, sep="\t", index=False)
        z.write(mp, f"{name}/MANIFEST.tsv")
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise ValueError("Zip CRC failure")
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
    parser.add_argument("action", choices=["rows", "wgbs", "base", "refit", "r15", "e1", "extra", "check", "assemble"])
    parser.add_argument("--index", type=int, default=0)
    a = parser.parse_args()
    single = {"rows": stage_rows, "base": stage_base, "r15": stage_r15, "extra": stage_extra}
    indexed = {"wgbs": stage_wgbs, "refit": stage_refit, "e1": stage_e1, "check": stage_check, "assemble": stage_assemble}
    single[a.action]() if a.action in single else indexed[a.action](a.index)


if __name__ == "__main__":
    main()
