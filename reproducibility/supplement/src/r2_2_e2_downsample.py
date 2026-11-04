"""Run one E2 BAM-level paired-template downsampling condition.

Templates are sampled from the sample's fresh native ``sorted.bam`` with
``samtools view --subsample F --subsample-seed S``.  samtools keeps a record
when a hash of its read name and the seed falls below ``F``, so both mates of a
template are kept or dropped together and, for one seed, lower depths are
subsets of higher ones.  The unchanged historical CHALM.py ``trad`` calls then
remove duplicates (``-r``) and apply the >=4-read rule, as in a shallower run
of the same library.  No remapping, CpG-count thinning or table subsampling.

The fraction is ``target_depth / native_median_ct``.  ``native_median_ct`` is
the median native methRatio ``CT_count`` over the audited EPIC universe; loci
absent from the >=4-filtered table have fewer than 4 reads, so the median is
exact while fewer than half of the universe is absent.  Targets at or above
the native median are recorded as unavailable, never oversampled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.r2_2_e3_aligned_call_cap import METRIC_FILES, run_original_call, sha256
from src.r2_2_epic_window_call import call_by_contig


COMPLETE_STATE = "native_calls_complete_unverified_archive"


def epic_loci(path: Path) -> pd.Index:
    loci = pd.read_csv(path, sep="\t", usecols=["locus"])["locus"]
    if loci.duplicated().any():
        raise ValueError(f"Duplicate EPIC loci in {path}")
    return pd.Index(loci)


def native_epic_depth(methratio: Path, universe: pd.Index) -> dict[str, object]:
    """Summarize native CT_count over a fixed locus universe."""

    table = pd.read_csv(methratio, sep="\t", usecols=["chr", "pos", "CT_count"],
                        dtype={"chr": "category", "pos": np.int64, "CT_count": np.int64})
    keys = table["chr"].astype(str) + "_" + table["pos"].astype(str)
    ct = pd.Series(table["CT_count"].to_numpy(), index=keys)
    if ct.index.duplicated().any():
        raise ValueError(f"Duplicate loci in {methratio}")
    present = ct.reindex(universe).dropna().to_numpy()
    n_universe = len(universe)
    n_absent = n_universe - len(present)
    if 2 * n_absent >= n_universe:
        raise ValueError("At least half of the universe is below 4 reads; median is censored")
    # Absent loci have CT < 4; any value below 4 gives the same median here.
    median = float(np.median(np.concatenate([present, np.zeros(n_absent)])))
    if median < 4:
        raise ValueError("Native median depth is censored below 4")
    return {
        "n_universe": n_universe,
        "n_present_ge4": int(len(present)),
        "n_absent_lt4": int(n_absent),
        "median_ct": median,
        "mean_ct_lower_bound": float(present.sum() / n_universe),
        "mean_ct_upper_bound": float((present.sum() + 3 * n_absent) / n_universe),
        "mean_ct_present": float(present.mean()),
    }


def flagstat_counts(samtools: str, bam: Path) -> dict[str, int]:
    output = subprocess.run([samtools, "flagstat", "-O", "json", str(bam)], check=True,
                            capture_output=True, text=True).stdout
    passed = json.loads(output)["QC-passed reads"]
    return {key: int(passed[key]) for key in ("total", "read1", "read2", "properly paired",
                                               "secondary", "supplementary", "mapped")}


def native_flagstat(path: Path) -> dict[str, int]:
    """Parse total/read1/read2 from a samtools flagstat text report."""

    counts = {}
    for line in path.read_text().splitlines():
        for key, label in (("total", "in total"), ("read1", "read1"), ("read2", "read2")):
            if line.endswith(label) or f" {label} " in f"{line} ":
                counts.setdefault(key, int(line.split()[0]))
    return counts


def check_templates(counts: dict[str, int], native: dict[str, int], fraction: float) -> dict[str, float]:
    """Check a template-sampled BAM against its native source.

    The native BSMAP BAMs carry slightly more read1 than read2 records (about
    0.2%), so mate counts are not compared for equality.  Template integrity is
    guaranteed by samtools' read-name hash and was verified per read name in the
    pilot; here the realized record fraction must match the target within 1%.
    """

    if counts["read1"] + counts["read2"] != counts["total"]:
        raise RuntimeError(f"Records other than read1/read2 present: {counts}")
    realized = counts["total"] / native["total"]
    if abs(realized / fraction - 1) > 0.01:
        raise RuntimeError(f"Realized fraction {realized:.5f} differs from target {fraction:.5f}")
    return {"realized_record_fraction": realized,
            "native_read1_minus_read2": native["read1"] - native["read2"],
            "sampled_read1_minus_read2": counts["read1"] - counts["read2"],
            "expected_sampled_imbalance": (native["read1"] - native["read2"]) * realized}


def caller_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle) - 1


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {out_dir}")
    native_dir = args.native_dir.resolve()
    native = json.loads((native_dir / "recovery_metadata.json").read_text())
    if native["state"] != COMPLETE_STATE or native["sample_id"] != args.sample_id:
        raise RuntimeError(f"Native baseline not complete for {args.sample_id}: {native['state']}")
    if native["min_depth"] != args.min_depth:
        raise RuntimeError("Native min_depth differs from requested min_depth")
    window = None
    if args.window_dir is not None:
        # EPIC-window mode: sample from the native's EPIC +/- flank extract; only EPIC rows are valid.
        window = json.loads((args.window_dir / "window_metadata.json").read_text())
        if window.get("state") != "complete" or Path(window["source_bam"]) != (native_dir / "sorted.bam").resolve():
            raise RuntimeError(f"Window BAM is not a complete extract of this native: {args.window_dir}")
        source_bam = (args.window_dir / "window.bam").resolve()
    else:
        source_bam = (args.source_bam or native_dir / "sorted.bam").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    depth = native_epic_depth(native_dir / METRIC_FILES["methRatio"], epic_loci(args.epic_mapping))
    fraction = args.target_depth / depth["median_ct"]
    metadata: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": args.sample_id,
        "seed": args.seed,
        "target_depth": args.target_depth,
        "native_dir": str(native_dir),
        "native_validated_sorted_records": native["validated_sorted_records"],
        "native_call_sha256": native["call_sha256"],
        "native_epic_depth": depth,
        "epic_mapping": str(args.epic_mapping.resolve()),
        "fraction": fraction,
        "source_bam": str(source_bam),
        "sampling": "samtools view --subsample F --subsample-seed S (read-name hash; templates kept whole)",
    }
    if fraction >= 1:
        metadata["status"] = "unavailable_target_not_below_native"
        (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        return metadata

    sampled_bam = out_dir / f"{args.sample_id}.seed{args.seed}.depth{args.target_depth:g}.bam"
    command = [args.samtools, "view", "-@", str(args.threads), "-b",
               "--subsample", f"{fraction:.10f}", "--subsample-seed", str(args.seed),
               "-o", str(sampled_bam), str(source_bam)]
    subprocess.run(command, check=True)
    subprocess.run([args.samtools, "quickcheck", "-v", str(sampled_bam)], check=True)
    counts = flagstat_counts(args.samtools, sampled_bam)
    native_counts = native_flagstat(native_dir / "flagstat.txt")
    if native_counts["total"] != native["validated_sorted_records"]:
        raise RuntimeError("Native flagstat total differs from validated record count")
    metadata["subsample_command"] = command
    metadata["sampled_counts"] = counts
    metadata["native_flagstat"] = native_counts
    # The realized fraction is checked against the BAM that was actually sampled.
    metadata["template_check"] = check_templates(counts, window["counts"] if window else native_counts, fraction)
    if window:
        metadata["epic_window"] = {"window_dir": str(args.window_dir.resolve()), "bed_sha256": window["bed_sha256"],
                                   "window_counts": window["counts"], "valid_rows": "EPIC loci only"}

    if args.call_processes > 1:
        calls = call_by_contig(sampled_bam, out_dir, args.contig_fasta_dir, args.chalm_py, args.samtools_dir,
                               args.python_executable, args.min_depth, args.samtools, args.call_processes)
    else:
        calls = run_original_call(
            bam=sampled_bam,
            out_dir=out_dir,
            reference=args.reference,
            chalm_py=args.chalm_py,
            samtools_dir=args.samtools_dir,
            python_executable=args.python_executable,
            min_depth=args.min_depth,
            chromosome=args.chromosome,
            parallel=args.parallel_callers,
        )
    rows = {target: caller_rows(out_dir / name) for target, name in METRIC_FILES.items()}
    if min(rows.values()) <= 0:
        raise RuntimeError(f"Caller output has no data rows: {rows}")
    metadata["calls"] = calls
    metadata["caller_data_rows"] = rows
    metadata["chalm_py_sha256"] = sha256(args.chalm_py)
    metadata["min_depth"] = args.min_depth
    metadata["sampled_bam_sha256"] = sha256(sampled_bam)
    if not args.keep_bam:
        sampled_bam.unlink()
        Path(f"{sampled_bam}.bai").unlink(missing_ok=True)
        metadata["sampled_bam_removed"] = True
    metadata["status"] = "complete"
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--source-bam", type=Path, help="Override native sorted.bam (pilot region BAMs)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-depth", type=float, required=True)
    parser.add_argument("--epic-mapping", type=Path, required=True)
    parser.add_argument("--samtools", required=True, help="samtools >=1.13 used for --subsample")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--chalm-py", type=Path, required=True)
    parser.add_argument("--samtools-dir", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--chromosome", help="Optional chromosome passed to CHALM.py (pilot only)")
    parser.add_argument("--keep-bam", action="store_true")
    parser.add_argument("--parallel-callers", action="store_true",
                        help="Run the methRatio and CHALM calls concurrently")
    parser.add_argument("--window-dir", type=Path,
                        help="EPIC-window extract of the native (r2_2_epic_window_call window); EPIC rows only")
    parser.add_argument("--call-processes", type=int, default=1,
                        help=">1: per-contig parallel calls (needs --contig-fasta-dir)")
    parser.add_argument("--contig-fasta-dir", type=Path)
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, default=str))


if __name__ == "__main__":
    main()
