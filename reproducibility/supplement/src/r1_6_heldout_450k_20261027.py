"""Resp 1.7 (R1-6): cell-type held-out predictions at the 28,344 450K-only CpGs (2026-10-27).

User request: only the held-out predictions for the 450K-only CpGs (the EPIC results are published).
Arms as in tissue_only_r1_5_r1_6_20261008: exclude_blood_immune (train 161, held out 44) and
exclude_epithelial (train 122, held out 83). Coefficients for the 450K-only loci come from the 2026-10-26
all-450K build (same fit_sites code and training sets, verified there). Held-out input and truth are the
held-out samples' atlas values. Per-CpG metrics follow paired_site_metrics: a held-out observation counts when
truth, A2H, linear and input are all finite; a CpG is eligible with >= 8 such observations.
Known answer: the same code applied to stored EPIC coefficients reproduces the stored held-out predictions and
site metrics exactly (50 EPIC loci per shard).
Run from R/: python -m src.r1_6_heldout_450k_20261027 build | assemble
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import src.raw_matrices_resp_20261023 as rm
from src.evaluate_pgpuk import file_hash, row_stats
from src.run_atlas_cv_split import shard_arrays
from src.tissue_only_transfer import members, predict_coefficients

ROOT = Path("work/a2h_raw_matrices_resp17_450k_20261027")
STAGE, ZIPS = ROOT / "staging", ROOT / "packages"
OUT = Path("results/r1_6_heldout_450k_20261027")
DOCS = Path("collaborator_figure_data_20261017/raw_matrices_resp17_450k_20261027")
S450 = Path("work/a2h_raw_matrices_450k_20261026/staging")
rm.STAGE, rm.OUT = STAGE, OUT
TARGETS, ARMS = rm.TARGETS, rm.ARMS
NAME = "A2H_Resp1.7-R1-6_heldout-predictions_EPIC-plus-450K-only_atlas_20261027"  # 2026-10-27: + combined per-CpG table
EPIC_TABLE = Path("collaborator_figure_data_20261017/package_v4/A2H_figure_data_20261017/Resp1.7_R1-6_celltype_holdouts/per_cpg_heldout_A2H_vs_linear.tsv.gz")


def record(name):
    rm.record(name)
    path = OUT / "stage_records" / f"{name}.json"
    rec = json.loads(path.read_text())
    rec["helper_code_sha256"] = rec.pop("code_sha256")
    rec["code_sha256"] = file_hash(Path(__file__))
    path.write_text(json.dumps(rec, indent=2, default=str) + "\n")


def site(loci, truth, a2h, lin, m):
    valid = np.isfinite(truth) & np.isfinite(a2h) & np.isfinite(lin) & np.isfinite(m)
    n = valid.sum(axis=1)
    out = {"locus": loci, "n": n}
    for k, p in (("A2H", a2h), ("reference_linear", lin)):
        out[k] = row_stats(np.where(valid, truth, np.nan), np.where(valid, p, np.nan))["rmse"]
    return pd.DataFrame(out)


def build():
    rows = pd.read_csv(S450 / "internal/rows.tsv", sep="\t")
    new = rows.loc[rows.row_set == "450K_only", ["probe_id", "locus"]].sort_values("probe_id").reset_index(drop=True)
    samples = pd.read_csv(rm.TISSUE / "atlas_samples.tsv", sep="\t")
    manifest = pd.read_csv(rm.TISSUE / "reference_membership.tsv", sep="\t")
    shards = pd.read_csv(rm.SHARDS, sep="\t")
    tmap = pd.read_csv(rm.TISSUE / "epic_mapping.tsv", sep="\t")
    cols = ["a2h_intercept", "a2h_linear", "a2h_quadratic", "reference_linear_intercept", "reference_linear_slope"]
    parts = {(arm, t): [] for arm in ARMS for t in TARGETS}
    ka_bad, ka_n = 0, 0
    for sh in shards.itertuples(index=False):
        frame = pd.read_parquet(sh.path)
        ka_loci = tmap.loc[tmap.shard_index == sh.shard_index, "locus"].drop_duplicates()
        ka_loci = frame.locus[frame.locus.isin(ka_loci)].iloc[:50]
        fnew = frame.loc[frame.locus.isin(new.locus)].reset_index(drop=True)
        fka = frame.loc[frame.locus.isin(ka_loci)].reset_index(drop=True)
        for arm in ARMS:
            train, test = members(manifest, samples, arm)
            for t in TARGETS:
                # 450K-only loci: coefficients from the all-450K build.
                coef = pd.read_parquet(S450 / f"internal/refit450k/R1-5_{t}_shard{sh.shard_index:04d}.parquet")
                coef = coef.loc[coef.model == arm].set_index("locus").reindex(fnew.locus)
                _, _, xt, truth = shard_arrays(fnew, t, train, test)
                p = predict_coefficients(coef[cols].reset_index(drop=True), xt)
                parts[(arm, t)].append((fnew.locus.to_numpy(), xt, truth, p["A2H"], p["reference_linear"]))
                # Known answer at stored EPIC loci.
                st = pd.read_parquet(rm.TISSUE / "fits" / arm / f"shard_{sh.shard_index:04d}" / f"{t}_coefficients.parquet").set_index("locus").reindex(fka.locus)
                _, _, xk, tk = shard_arrays(fka, t, train, test)
                pk = predict_coefficients(st[cols].reset_index(drop=True), xk)
                hp = pd.read_parquet(rm.TISSUE / "fits" / arm / f"shard_{sh.shard_index:04d}" / f"{t}_heldout_predictions.parquet")
                hp = hp.loc[hp.locus.isin(fka.locus)]
                order = pd.MultiIndex.from_arrays([np.repeat(fka.locus.to_numpy(), len(test)), np.tile(test, len(fka))])
                hp = hp.set_index(["locus", "sample_id"]).reindex(order)
                same = all(np.array_equal(hp[c].to_numpy(np.float64), v.ravel(), equal_nan=True)
                           for c, v in (("A2H", pk["A2H"]), ("reference_linear", pk["reference_linear"]), ("m", xk), ("observed", tk)))
                sm = pd.read_parquet(rm.TISSUE / "fits" / arm / f"shard_{sh.shard_index:04d}" / f"{t}_heldout_site_metrics.parquet")
                mine = site(fka.locus.to_numpy(), tk, pk["A2H"], pk["reference_linear"], xk).set_index("locus")
                for method in ("A2H", "reference_linear"):
                    s = sm.loc[sm.method == method].set_index("locus").reindex(fka.locus)
                    same &= np.array_equal(s.n.to_numpy(), mine.n.to_numpy()) and np.allclose(s.rmse.to_numpy(), mine[method].to_numpy(), rtol=0, atol=1e-15, equal_nan=True)
                ka_bad += not same
                ka_n += len(fka)
    rm.check("known answer: stored EPIC held-out predictions (A2H, linear, input, truth) and site n/RMSE reproduced at 50 loci per shard x 2 arms x 2 targets",
             f"{ka_n} locus-arm-target cases", f"{ka_bad} shard-arm-target failures", ka_bad == 0)
    tables = []
    for arm in ARMS:
        _, test = members(manifest, samples, arm)
        for t in TARGETS:
            loci = np.concatenate([x[0] for x in parts[(arm, t)]])
            pos = pd.Index(loci).get_indexer(new.locus)
            if (pos < 0).any():
                raise ValueError("450K-only locus missing from atlas shards")
            arr = {k: np.vstack([x[i] for x in parts[(arm, t)]])[pos] for i, k in ((1, "input_m"), (2, "observed"), (3, "A2H"), (4, "linear"))}
            for k in ("A2H", "linear", "observed"):
                rm.write_matrix(f"{arm}/{k}_{t}_{arm}.tsv.gz", new, test, arr[k])
            if t == "CHALM":
                rm.write_matrix(f"{arm}/input_methylation_ratio_{arm}.tsv.gz", new, test, arr["input_m"])
            s = site(new.locus.to_numpy(), arr["observed"], arr["A2H"], arr["linear"], arr["input_m"])
            elig = s.loc[s.n >= 8]
            tables.append(pd.DataFrame({"heldout_arm": arm, "target": t, "locus": elig.locus, "n_heldout_obs": elig.n,
                                        "A2H_rmse": elig.A2H, "linear_rmse": elig.reference_linear}))
            rm.check(f"{arm} {t}: 450K-only CpGs eligible (>= 8 held-out observations)", len(new), len(elig), True)
            rm.check(f"{arm} {t}: mean per-CpG RMSE at 450K-only CpGs, A2H / linear (informational)", "-",
                     f"{elig.A2H.mean():.5f} / {elig.reference_linear.mean():.5f}", True)
    table = pd.concat(tables, ignore_index=True)
    path = STAGE / "per_cpg_heldout_A2H_vs_linear_450K-only.tsv.gz"
    table.to_csv(path, sep="\t", index=False, float_format="%.6g")
    rm.FILES.append({"file": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size, "rows": len(table)})
    rm.write_table(new, "row_annotation_450K-only.tsv")
    rm.copy_file(rm.TISSUE / "reference_membership.tsv", "tables/R1-6_reference_membership.tsv")
    rm.copy_file(rm.TISSUE / "reference_models.tsv", "tables/R1-6_reference_models.tsv")
    record("build")


def combine():
    """Original EPIC per-CpG table + the 450K-only rows, same columns, plus cpg_set (user request 2026-10-27)."""
    epic_raw = EPIC_TABLE.read_bytes()
    import gzip
    epic_lines = gzip.decompress(epic_raw).decode().splitlines()
    new_lines = gzip.decompress((STAGE / "per_cpg_heldout_A2H_vs_linear_450K-only.tsv.gz").read_bytes()).decode().splitlines()
    rm.check("header of the 450K-only table equals the EPIC table header", epic_lines[0], new_lines[0], epic_lines[0] == new_lines[0])
    header = epic_lines[0] + "\tcpg_set"
    body = [l + "\tEPIC" for l in epic_lines[1:]] + [l + "\t450K_only" for l in new_lines[1:]]
    path = STAGE / "per_cpg_heldout_A2H_vs_linear_EPIC_and_450K-only.tsv.gz"
    if path.exists():
        raise FileExistsError(path)
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(("\n".join([header] + body) + "\n").encode())
    rm.FILES.append({"file": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size, "rows": len(body)})
    back = pd.read_csv(path, sep="\t")
    epic = pd.read_csv(EPIC_TABLE, sep="\t")
    part = back.loc[back.cpg_set == "EPIC"].drop(columns="cpg_set").reset_index(drop=True)
    rm.check("EPIC rows identical to the original table (values and order)", len(epic), len(part), part.equals(epic))
    rm.check("no CpG repeated within arm x target", 0, int(back.duplicated(["heldout_arm", "target", "locus"]).sum()),
             not back.duplicated(["heldout_arm", "target", "locus"]).any())
    reported = {("exclude_blood_immune", "CHALM"): (0.05466, 0.06010), ("exclude_blood_immune", "CAMDA"): (0.05465, 0.06009),
                ("exclude_epithelial", "CHALM"): (0.05743, 0.06188), ("exclude_epithelial", "CAMDA"): (0.05743, 0.06188)}
    for (arm, tg), (a2h, lin) in reported.items():
        s = back.loc[(back.heldout_arm == arm) & (back.target == tg)]
        e = s.loc[s.cpg_set == "EPIC"]
        ok = abs(e.A2H_rmse.mean() - a2h) <= 5e-6 and abs(e.linear_rmse.mean() - lin) <= 5e-6
        rm.check(f"{arm} {tg}: EPIC rows give the published means A2H {a2h} / linear {lin}", f"{a2h} / {lin}",
                 f"{e.A2H_rmse.mean():.5f} / {e.linear_rmse.mean():.5f}", ok)
        rm.check(f"{arm} {tg}: combined rows and means (informational)", "-",
                 f"EPIC {len(e)} + 450K-only {int((s.cpg_set == '450K_only').sum())} = {len(s)} CpGs; A2H {s.A2H_rmse.mean():.5f} / linear {s.linear_rmse.mean():.5f}", True)
    record("combine")


NAME_450K_ONLY = "A2H_Resp1.7-R1-6_heldout-predictions_450K-only-CpGs_atlas_20261027"


def assemble(name=None, stages=("build", "combine"), readme_file="README.md"):
    """Combined-table zip by default; ``assemble450`` = the 450K-only zip with merge instructions in its README."""
    global NAME
    NAME = name or NAME
    files = [f for r in stages for f in json.loads((OUT / f"stage_records/{r}.json").read_text())["files"]]
    checks = pd.concat([pd.DataFrame(json.loads((OUT / f"stage_records/{r}.json").read_text())["checks"]).assign(stage=r)
                        for r in stages], ignore_index=True)
    if not checks["pass"].all():
        raise SystemExit("failed checks")
    readme = (DOCS / readme_file).read_text()
    if re.search(r"__[A-Z_]+__", readme):
        raise ValueError("placeholder")
    ZIPS.mkdir(parents=True, exist_ok=True)
    target = ZIPS / f"{NAME}.zip"
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(".zip.partial")
    manifest = []
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        def add(src, arc):
            z.write(src, f"{NAME}/{arc}", compress_type=zipfile.ZIP_STORED if arc.endswith(".gz") else zipfile.ZIP_DEFLATED)
            manifest.append({"file": arc, "bytes": Path(src).stat().st_size, "sha256": file_hash(Path(src))})
        merged_readme = OUT / f"README_{NAME}.md"
        merged_readme.write_text(readme)
        add(merged_readme, "README.md")
        for f in sorted(files, key=lambda x: x["file"]):
            if file_hash(STAGE / f["file"]) != f["sha256"]:
                raise ValueError(f["file"])
            add(STAGE / f["file"], f["file"])
        cp = OUT / "CHECKS.tsv"
        checks.to_csv(cp, sep="\t", index=False)
        add(cp, "CHECKS.tsv")
        mp = OUT / "MANIFEST.tsv"
        pd.DataFrame(manifest).to_csv(mp, sep="\t", index=False)
        z.write(mp, f"{NAME}/MANIFEST.tsv")
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise ValueError("CRC")
        n = len(z.namelist())
    tmp.rename(target)
    digest = file_hash(target)
    (ZIPS / f"{NAME}.zip.sha256").write_text(f"{digest}  {target.name}\n")
    summary = {"zip": str(target), "sha256": digest, "bytes": target.stat().st_size, "members": n,
               "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / f"assemble_{NAME}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=["build", "combine", "assemble", "assemble450"])
    {"build": build, "combine": combine, "assemble": assemble,
     "assemble450": lambda: assemble(NAME_450K_ONLY, ("build",), "README_450K-only.md")}[a.parse_args().action]()
