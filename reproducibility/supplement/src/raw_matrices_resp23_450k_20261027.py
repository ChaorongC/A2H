"""Resp 2.3 (R2-2): extend the PGP-UK and atlas39 matrices to the 28,344 450K-only CpGs (2026-10-27).

User request: rerun only the missing 450K-only CpGs and match the format of the 2026-10-24 Resp 2.3 package.

Design. Window-mode E2/E3 calls are valid only at window centres (src/r2_2_epic_window_call.py), so no
450K-only value is taken from the existing EPIC windows. New windows (450K-only CpG +/- 600 bp, merged) are
extracted from the 39 native BAMs (indexed ``-M -L``; record stream checked against streaming ``-L``) and all
12 E2/E3 conditions are rerun on them with the unchanged runners (E2 fraction = target / native EPIC median,
same seeds; E3 caps). Natives and PGP-UK capped calls are genome-wide and are read directly. Known answers:
the 91 conditions that were originally genome-wide must give identical calls at the 450K-only CpGs; refits
must reproduce stored refit39 coefficients at EPIC loci; EPIC rows of every matrix must equal the 2026-10-24
package. Run from R/: python -m src.raw_matrices_resp23_450k_20261027 <action> [--index i]
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import src.raw_matrices_resp_20261023 as rm
from src.evaluate_pgpuk import file_hash
from src.r2_2_common import fit_compatible
from src.r2_2_e2e3_external import predict as refit_predict
from src.r2_2_e2e3_summarize import epic_calls
from src.r2_2_epic_window_call import flagstat_json, idxstats, window_bed

ROOT = Path("work/a2h_raw_matrices_resp23_450k_20261027")
STAGE, ZIPS, WIN, RUNS = ROOT / "staging", ROOT / "packages", ROOT / "window_native", ROOT / "runs"
BED = ROOT / "window_bed/w600_450k_only.bed"
OLD_STAGE, OLD_OUT = rm.STAGE, rm.OUT
S450 = Path("work/a2h_raw_matrices_450k_20261026/staging")
OUT450 = Path("results/raw_matrices_resp_450k_20261026")
OUT = Path("results/raw_matrices_resp23_450k_20261027")
DOCS = Path("collaborator_figure_data_20261017/raw_matrices_resp23_450k_20261027")
rm.STAGE, rm.OUT = STAGE, OUT
E2E3 = Path("work/r2_2_e2e3_20261015")
NATIVES = Path("results/r2_2_e2e3_20261015/manifests/window_native_manifest.tsv")
QC = Path("results/r2_2_e2e3_20261015/summary_20261016/qc_runs.tsv")
SAMTOOLS = "/opt/apps/samtools/1.15.1/bin/samtools"
CONDITIONS = rm.CONDITIONS  # e2 native first, then 9 E2, then 3 E3
TARGETS = rm.TARGETS


def record(name):
    rm.record(name)
    path = OUT / "stage_records" / f"{name}.json"
    rec = json.loads(path.read_text())
    rec["helper_code_sha256"] = rec.pop("code_sha256")
    rec["code_sha256"] = file_hash(Path(__file__))
    path.write_text(json.dumps(rec, indent=2, default=str) + "\n")


def rows450():
    return pd.read_csv(S450 / "internal/rows.tsv", sep="\t")


def new_rows():
    r = rows450()
    return r.loc[r.row_set == "450K_only", ["probe_id", "locus"]].sort_values("probe_id").reset_index(drop=True)


def natives():
    return pd.read_csv(NATIVES, sep="\t")


# ----------------------------------------------------------------------------------------------
# bed, windows, run manifests
# ----------------------------------------------------------------------------------------------

def stage_bed():
    loci = new_rows()
    tmp = ROOT / "window_bed/450k_only_loci.tsv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    loci.to_csv(tmp, sep="\t", index=False)
    info = window_bed(tmp, 600, BED)
    BED.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    rm.check("bed: 450K-only CpG +/- 600 bp windows", 28344, f"{info['n_loci']} loci, {info['n_intervals']} intervals, {info['covered_bp']} bp",
             info["n_loci"] == 28344)
    # Run manifests for the unchanged window-mode wrappers (all 12 conditions x 39 samples).
    n = natives()
    e2, e3 = [], []
    for row in n.itertuples(index=False):
        wdir = WIN / row.sample_id
        for seed in (1, 2, 3):
            for depth in (20, 10, 5):
                e2.append({"task": len(e2), "sample_id": row.sample_id, "seed": seed, "target_depth": depth, "native_dir": row.native_dir,
                           "window_dir": str(wdir), "out_dir": str(RUNS / "e2" / row.sample_id / f"seed{seed}_depth{depth}"), "e2_manifest_task": -1})
        for raw in (125, 100, 75):
            e3.append({"task": len(e3), "sample_id": row.sample_id, "condition": f"raw{raw}", "native_dir": row.native_dir,
                       "window_dir": str(wdir), "out_dir": str(RUNS / "e3" / row.sample_id / f"raw{raw}"), "e3_manifest_task": -1})
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(e2).to_csv(OUT / "manifests/e2_450k_window_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(e3).to_csv(OUT / "manifests/e3_450k_window_manifest.tsv", sep="\t", index=False)
    pd.DataFrame([{"task": i, "sample_id": r.sample_id, "native_dir": r.native_dir, "window_dir": str(WIN / r.sample_id)}
                  for i, r in enumerate(n.itertuples(index=False))]).to_csv(OUT / "manifests/window_450k_manifest.tsv", sep="\t", index=False)
    rm.check("manifests: 351 E2 + 117 E3 runs on the new windows", "351 / 117", f"{len(e2)} / {len(e3)}", len(e2) == 351 and len(e3) == 117)
    record("bed")


def stage_window(index):
    row = natives().iloc[index]
    out = WIN / row.sample_id
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    source = (Path(row.native_dir) / "sorted.bam").resolve()
    threads = str(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) - 1))
    command = [SAMTOOLS, "view", "-@", threads, "-b", "-M", "-L", str(BED), "-o", str(out / "window.bam"), str(source)]
    subprocess.run(command, check=True)
    subprocess.run([SAMTOOLS, "index", str(out / "window.bam")], check=True)
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_bam": str(source), "bed": str(BED.resolve()),
            "bed_sha256": file_hash(BED), "command": command, "counts": flagstat_json(SAMTOOLS, out / "window.bam"),
            "source_counts_from_index": idxstats(SAMTOOLS, source), "valid_rows": "window-centre loci only (450K-only CpGs)",
            "extraction": "indexed -M -L; record stream identical to the streaming -L used for the EPIC windows (job 58161051; rechecked by window_ka)",
            "state": "complete"}
    (out / "window_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"sample": row.sample_id, "records": meta["counts"]["total"]}), flush=True)


def stage_window_ka():
    """Streaming -L extract of one native must give the same record stream as the indexed window."""
    row = natives().iloc[0]
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")))
    streamed = tmp / "streamed.bam"
    subprocess.run([SAMTOOLS, "view", "-@", "3", "-b", "-L", str(BED), "-o", str(streamed), str(Path(row.native_dir) / "sorted.bam")], check=True)

    def md5(bam):
        h = hashlib.md5()
        proc = subprocess.Popen([SAMTOOLS, "view", str(bam)], stdout=subprocess.PIPE)
        for chunk in iter(lambda: proc.stdout.read(1 << 20), b""):
            h.update(chunk)
        if proc.wait() != 0:
            raise RuntimeError("samtools view failed")
        return h.hexdigest()
    a, b = md5(WIN / row.sample_id / "window.bam"), md5(streamed)
    rm.check(f"window_ka {row.sample_id}: indexed -M -L and streaming -L record streams identical (md5)", a, b, a == b)
    streamed.unlink()
    record("window_ka")


# ----------------------------------------------------------------------------------------------
# extract: per sample, native + 12 conditions at the 450K-only CpGs; known answers
# ----------------------------------------------------------------------------------------------

def stage_extract(index):
    row = natives().iloc[index]
    loci = pd.Index(new_rows().locus)
    qc = pd.read_csv(QC, sep="\t")
    parts = []
    nat = epic_calls(Path(row.native_dir), loci).reindex(loci)
    parts.append(nat.assign(experiment="e2", condition="native").reset_index())
    bad_full, n_full = 0, 0
    for exp, cond in CONDITIONS[1:]:
        run = RUNS / exp / row.sample_id / cond
        meta = json.loads((run / "run_metadata.json").read_text())
        if meta.get("status") == "unavailable_target_not_below_native":
            v = pd.DataFrame(index=loci, columns=["m", "chalm", "ct"], dtype=float)
        else:
            v = epic_calls(run, loci).reindex(loci)
        parts.append(v.assign(experiment=exp, condition=cond).reset_index())
        mode = qc.loc[(qc.experiment == exp) & (qc.sample_id == row.sample_id) & (qc.condition == cond), "mode"]
        if len(mode) and mode.iloc[0] == "full":
            n_full += 1
            orig = epic_calls(E2E3 / exp / row.sample_id / cond, loci).reindex(loci)
            same = all(np.array_equal(v[c].to_numpy(float), orig[c].to_numpy(float), equal_nan=True) for c in ("m", "chalm", "ct"))
            bad_full += not same
    rm.check(f"extract {row.sample_id}: new-window calls equal the original genome-wide calls at 450K-only CpGs",
             f"{n_full} genome-wide conditions", f"{bad_full} differing", bad_full == 0)
    # EPIC rows: the native genome-wide call at 500 EPIC loci equals the 2026-10-24 atlas39 matrices.
    old = rm.read_matrix(OLD_STAGE / "atlas39/calls/e2__native/wgbs_CHALM_e2__native.tsv.gz", ["probe_id", "locus", row.sample_id])
    sample = old.iloc[np.linspace(0, len(old) - 1, 500).astype(int)]
    nat_epic = epic_calls(Path(row.native_dir), pd.Index(sample.locus)).reindex(sample.locus)
    same = np.array_equal(nat_epic.chalm.to_numpy(float), sample[row.sample_id].to_numpy(float), equal_nan=True)
    rm.check(f"extract {row.sample_id}: native genome-wide CHALM at 500 EPIC loci equals the 2026-10-24 matrix", True, same, same)
    out = STAGE / f"internal/extract/{row.sample_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    record(f"extract_{row.sample_id}")


# ----------------------------------------------------------------------------------------------
# condition: refit39 at the 450K-only loci, PGP-UK predictions (all rows), atlas39 call matrices
# ----------------------------------------------------------------------------------------------

def condition_matrix(samples, exp, cond):
    loci = new_rows().locus
    vals = {k: [] for k in ("m", "chalm", "ct")}
    for s in samples:
        e = pd.read_parquet(STAGE / f"internal/extract/{s}.parquet")
        e = e.loc[(e.experiment == exp) & (e.condition == cond)].set_index("locus").reindex(loci)
        for k in vals:
            vals[k].append(e[k].to_numpy(np.float64))
    return {k: np.column_stack(v) for k, v in vals.items()}


def stage_condition(index):
    exp, cond = CONDITIONS[index]
    key = f"{exp}__{cond}"
    refit_dir = rm.REFIT / key
    meta = json.loads((refit_dir / "refit.json").read_text())
    samples, min_train = meta["reference_samples"], meta["min_train"]
    sel = rm.selected39().sample_id.tolist()
    nr = new_rows()
    v_new = condition_matrix(sel, exp, cond)
    # atlas39 call matrices: 2026-10-24 rows unchanged + 450K-only rows appended.
    for k, name in (("m", "beta"), ("chalm", "CHALM"), (None, "CAMDA"), ("ct", "depth")):
        old = rm.read_matrix(OLD_STAGE / f"atlas39/calls/{key}/wgbs_{name}_{key}.tsv.gz")
        new_vals = v_new["chalm"] - v_new["m"] if k is None else v_new[k]
        rows = pd.concat([old[["probe_id", "locus"]], nr], ignore_index=True)
        values = np.vstack([old[sel].to_numpy(np.float64), new_vals])
        rm.write_matrix(f"atlas39/calls/{key}/wgbs_{name}_{key}.tsv.gz", rows, sel, values, integer=(name == "depth"))
    # refit39: new fits at the 450K-only loci; known answer at stored EPIC loci.
    rows = rows450()
    b = rm.read_matrix(S450 / "cohort/PGPUK/base/array_beta.tsv.gz")
    ids = [c for c in b.columns if c not in ("probe_id", "locus")]
    beta = b[ids].to_numpy(np.float64)
    order = [sel.index(s) for s in samples]
    calls_old = {k: rm.read_matrix(OLD_STAGE / f"atlas39/calls/{key}/wgbs_{k}_{key}.tsv.gz") for k in ("beta", "CHALM")}
    ka_idx = np.linspace(0, len(calls_old["beta"]) - 1, 400).astype(int)
    for target in TARGETS:
        x, chalm = v_new["m"][:, order], v_new["chalm"][:, order]
        y = chalm - x if target == "CAMDA" else chalm
        q_new = fit_compatible(x, y, min_train=min_train)
        q_new.insert(0, "locus", nr.locus.to_numpy())
        stored = pd.read_parquet(OLD_STAGE / f"internal/refit39_coefficients/{key}_{target}.parquet")
        xo = calls_old["beta"][samples].to_numpy(np.float64)[ka_idx]
        co = calls_old["CHALM"][samples].to_numpy(np.float64)[ka_idx]
        q_ka = fit_compatible(xo, co - xo if target == "CAMDA" else co, min_train=min_train)
        st = stored.set_index("locus").loc[calls_old["beta"].locus.to_numpy()[ka_idx]]
        cols = ["a2h_intercept", "a2h_linear", "a2h_quadratic"]
        same = (np.array_equal(q_ka[cols].to_numpy(), st[cols].to_numpy(), equal_nan=True)
                and (q_ka.fit_status.to_numpy() == st.fit_status.to_numpy()).all())
        rm.check(f"{key} {target}: refit at 400 stored EPIC loci reproduces the stored refit39 coefficients exactly", True, same, same)
        allq = pd.concat([stored, q_new[stored.columns.intersection(q_new.columns)]], ignore_index=True)
        q = allq.set_index("locus").reindex(rows.locus.to_numpy()).reset_index(drop=True)
        pred = refit_predict(q, beta)
        rm.write_matrix(f"cohort/PGPUK/refit39/{target}/A2H_refit39_{key}_{target}.tsv.gz", rows, ids, pred)
        old = rm.read_matrix(OLD_STAGE / f"cohort/PGPUK/refit39/{target}/A2H_refit39_{key}_{target}.tsv.gz").set_index("probe_id")
        pos = pd.Series(np.arange(len(rows)), index=rows.probe_id).loc[old.index].to_numpy()
        same = np.array_equal(old[ids].to_numpy(np.float64), pred[pos], equal_nan=True)
        rm.check(f"{key} {target}: PGP-UK predictions at the 390,266 EPIC rows identical to the 2026-10-24 package", True, same, same)
        rm.check(f"{key} {target}: 450K-only loci with a fit (min_train {min_train})", len(nr), int(q_new.fit_status.isin(["quadratic", "constant_kde", "quadratic_rank_warning"]).sum()), True)
    record(f"condition_{key}")


# ----------------------------------------------------------------------------------------------
# pgpukc: PGP-UK read-length truth on all rows
# ----------------------------------------------------------------------------------------------

def stage_pgpukc():
    rows = rows450()
    b = rm.read_matrix(S450 / "cohort/PGPUK/base/array_beta.tsv.gz")
    ids = [c for c in b.columns if c not in ("probe_id", "locus")]
    pairs = pd.read_csv(rm.PGP_OBS / "pairs.tsv", sep="\t").set_index("array_id")
    wgbs = pairs.loc[ids, "wgbs_id"].tolist()
    is_new = (rows.row_set == "450K_only").to_numpy()
    loci_new = pd.Index(rows.locus[is_new])
    # The native PGP-UK truth is the base WGBS (the same genome-wide calls); no separate capped "native" run exists.
    for cond in ("raw125", "raw100", "raw75"):
        val = {k: np.full((len(rows), len(ids)), np.nan) for k in ("beta", "CHALM", "depth")}
        for j, w in enumerate(wgbs):
            e = epic_calls(E2E3 / "pgpuk_c" / w / cond, loci_new).reindex(loci_new)
            val["beta"][is_new, j], val["CHALM"][is_new, j], val["depth"][is_new, j] = e.m, e.chalm, e.ct
        folder = f"cohort/PGPUK/E3_readlength_truth/{cond}"
        old = {k: rm.read_matrix(OLD_STAGE / f"{folder}/{f}_{cond}.tsv.gz").set_index("probe_id")
               for k, f in (("beta", "wgbs_beta"), ("CHALM", "wgbs_CHALM"), ("depth", "wgbs_depth"))}
        pos = pd.Series(np.arange(len(rows)), index=rows.probe_id).loc[old["beta"].index].to_numpy()
        for k in val:
            val[k][pos] = old[k][ids].to_numpy(np.float64)
        val["CAMDA"] = val["CHALM"] - val["beta"]
        for f, k in (("wgbs_beta", "beta"), ("wgbs_CHALM", "CHALM"), ("wgbs_CAMDA", "CAMDA")):
            rm.write_matrix(f"{folder}/{f}_{cond}.tsv.gz", rows, ids, val[k])
        rm.write_matrix(f"{folder}/wgbs_depth_{cond}.tsv.gz", rows, ids, val["depth"], integer=True)
        d10 = np.nan_to_num(val["depth"], nan=0) >= 10
        rm.write_matrix(f"{folder}/mask_wgbs_depth_ge10_{cond}.tsv.gz", rows, ids, d10.astype(np.int8), integer=True)
        oldc = rm.read_matrix(OLD_STAGE / f"{folder}/wgbs_CAMDA_{cond}.tsv.gz").set_index("probe_id")
        same = np.array_equal(oldc[ids].to_numpy(np.float64), val["CAMDA"][pos], equal_nan=True)
        rm.check(f"PGP-UK {cond}: CAMDA at EPIC rows identical to the 2026-10-24 package", True, same, same)
        rm.check(f"PGP-UK {cond}: 450K-only cells with capped CHALM", "-", int(np.isfinite(val["CHALM"][is_new]).sum()), True)
    record("pgpukc")


# ----------------------------------------------------------------------------------------------
# extra and assemble
# ----------------------------------------------------------------------------------------------

def stage_extra():
    nr = new_rows()
    old = rm.read_matrix(OLD_STAGE / "atlas39/calls/e2__native/wgbs_depth_e2__native.tsv.gz", ["probe_id", "locus"])
    ann = pd.concat([old.assign(row_set="EPIC_audited"), nr.assign(row_set="450K_only")], ignore_index=True)
    rm.write_table(ann, "atlas39/row_annotation.tsv")
    rm.check("atlas39 rows: 824,209 EPIC + 28,344 450K-only", 852553, len(ann), len(ann) == 852553 and not ann.locus.duplicated().any())
    record("extra")


NEW_PREFIXES = ["cohort/PGPUK/refit39/", "cohort/PGPUK/E3_readlength_truth/", "atlas39/calls/", "atlas39/row_annotation.tsv"]
S450_PREFIXES = ["cohort/PGPUK/base/", "cohort/PGPUK/masks/", "cohort/PGPUK/E1/", "cohort/PGPUK/row_annotation.tsv", "cohort/PGPUK/supplement_",
                 "tables/E1_subset_membership.tsv", "tables/atlas205_samples.tsv", "published/R2-2_E1/"]
OLD_PREFIXES = ["published/R2-2_E2_E3/", "published/R2-2_E3_PGPUK/", "published/R2-2_refit39/", "tables/atlas39_samples.tsv"]
NAME = "A2H_Resp2.3-R2-2_per-individual-unfiltered-matrices_PGPUK-all450K_atlas39_20261027"


def staged(out, stage):
    files = {}
    for path in sorted((out / "stage_records").glob("*.json")):
        for f in json.loads(path.read_text())["files"]:
            files[f["file"]] = (stage / f["file"], f["sha256"])
    return files


def stage_assemble():
    sources = [(staged(OUT, STAGE), NEW_PREFIXES), (staged(OUT450, S450), S450_PREFIXES), (staged(OLD_OUT, OLD_STAGE), OLD_PREFIXES)]
    chosen = {}
    for files, prefixes in sources:
        for rel, (path, digest) in files.items():
            if any(fnmatch.fnmatch(rel, p + "*") for p in prefixes) and rel not in chosen:
                chosen[rel] = (path, digest)
    recs = [json.loads(p.read_text()) for p in sorted((OUT / "stage_records").glob("*.json"))]
    recs += [json.loads(p.read_text()) for p in sorted((OUT450 / "stage_records").glob("*.json"))
             if p.stem in ("rows", "base", "extra") or p.stem.startswith(("refit_shard", "e1_"))]
    recs += [json.loads(p.read_text()) for p in sorted((OLD_OUT / "stage_records").glob("*.json"))
             if p.stem.startswith(("refit39_", "e3_pgpuk", "atlas39_calls_e"))]
    checks = pd.DataFrame([{"stage": r["stage"], **c} for r in recs for c in r["checks"]])
    if checks.empty or not checks["pass"].all():
        raise SystemExit("Missing or failed checks")
    readme = (DOCS / "README_Resp2.3.md").read_text().replace("__COMMON__", (DOCS / "README_common_PGPUK.md").read_text())
    if re.search(r"__[A-Z_]+__", readme):
        raise ValueError("Unfilled placeholder in README")
    ZIPS.mkdir(parents=True, exist_ok=True)
    target = ZIPS / f"{NAME}.zip"
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix(".zip.partial")
    manifest = []
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        def add(source, arc):
            z.write(source, f"{NAME}/{arc}", compress_type=zipfile.ZIP_STORED if arc.endswith(".gz") else zipfile.ZIP_DEFLATED)
            manifest.append({"file": arc, "bytes": Path(source).stat().st_size, "sha256": file_hash(Path(source))})
        merged = OUT / "README_Resp2.3.md"
        merged.write_text(readme)
        add(merged, "README.md")
        for rel, (path, digest) in sorted(chosen.items()):
            if file_hash(path) != digest:
                raise ValueError(f"Staged file changed: {path}")
            add(path, rm.arcname(rel))
        cp = OUT / "CHECKS_Resp2.3.tsv"
        checks.to_csv(cp, sep="\t", index=False)
        add(cp, "CHECKS.tsv")
        mp = OUT / "MANIFEST_Resp2.3.tsv"
        pd.DataFrame(manifest).to_csv(mp, sep="\t", index=False)
        z.write(mp, f"{NAME}/MANIFEST.tsv")
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise ValueError("Zip CRC failure")
        n = len(z.namelist())
    tmp.rename(target)
    digest = file_hash(target)
    (ZIPS / f"{NAME}.zip.sha256").write_text(f"{digest}  {target.name}\n")
    summary = {"package": "2.3", "zip": str(target), "sha256": digest, "bytes": target.stat().st_size, "members": n,
               "checks": len(checks), "finished_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "assemble_Resp2.3.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["bed", "window", "window_ka", "extract", "condition", "pgpukc", "extra", "assemble"])
    p.add_argument("--index", type=int, default=0)
    a = p.parse_args()
    single = {"bed": stage_bed, "window_ka": stage_window_ka, "pgpukc": stage_pgpukc, "extra": stage_extra, "assemble": stage_assemble}
    indexed = {"window": stage_window, "extract": stage_extract, "condition": stage_condition}
    single[a.action]() if a.action in single else indexed[a.action](a.index)


if __name__ == "__main__":
    main()
