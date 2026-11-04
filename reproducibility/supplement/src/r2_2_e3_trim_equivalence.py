"""Known-answer check for the E3 ``raw*`` cap emulation on a small region BAM.

A separate implementation physically shortens every mate to its retained
sequencer 5-prime prefix (shorter SEQ/QUAL/CIGAR, reverse reads start later,
left-mate PNEXT = trimmed mate start), keeps the record order, and runs the same
caller.  Its call tables must be byte-identical to the masked + PNEXT-adjusted
condition produced by ``r2_2_e3_aligned_call_cap.py``.  The masked condition
WITHOUT the PNEXT adjustment is also called, to record what omitting it would
have changed.  Only all-match CIGARs are supported (true for BSMAP output).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pysam

from src.r2_2_e3_aligned_call_cap import METRIC_FILES, RAW_READ_CAPS, mask_bam, run_original_call, sha256


def trimmed_bam(source: Path, output: Path, cap_r1: int, cap_r2: int) -> dict[str, int]:
    with pysam.AlignmentFile(str(source), "rb") as handle:
        header = handle.header.to_dict()
        reads = list(handle.fetch(until_eof=True))
    new_start: dict[tuple[str, bool], int] = {}
    for read in reads:
        if read.cigartuples != [(0, read.query_length)]:
            raise ValueError(f"Non all-match CIGAR: {read.query_name}")
        cap = cap_r1 if read.is_read1 else cap_r2
        shift = max(read.query_length - cap, 0) if read.is_reverse else 0
        new_start[(read.query_name, read.is_read1)] = read.reference_start + shift
    stats = {"n_records": 0, "n_trimmed": 0, "n_pnext_changed": 0}
    with pysam.AlignmentFile(str(output), "wb", header=header) as out:
        for read in reads:
            cap = cap_r1 if read.is_read1 else cap_r2
            length = read.query_length
            stats["n_records"] += 1
            if cap < length:
                keep = slice(length - cap, length) if read.is_reverse else slice(0, cap)
                sequence, qualities = read.query_sequence[keep], read.query_qualities[keep]
                read.reference_start = new_start[(read.query_name, read.is_read1)]
                read.query_sequence = sequence
                read.query_qualities = qualities
                read.cigartuples = [(0, cap)]
                stats["n_trimmed"] += 1
            mate = new_start.get((read.query_name, not read.is_read1))
            if read.template_length > 0 and read.is_proper_pair and mate is not None and mate != read.next_reference_start:
                read.next_reference_start = mate
                stats["n_pnext_changed"] += 1
            out.write(read)
    return stats


def table_summary(path: Path) -> dict[str, float]:
    n = 0
    depth = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            n += 1
            depth += int(row["CT_count"])
    return {"n_loci": n, "sum_ct": depth, "mean_ct": depth / n if n else float("nan")}


def run(args: argparse.Namespace) -> dict[str, object]:
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cap_r1, cap_r2 = RAW_READ_CAPS[args.condition]
    caller = dict(reference=args.reference, chalm_py=args.chalm_py, samtools_dir=args.samtools_dir,
                  python_executable=args.python_executable, min_depth=args.min_depth,
                  chromosome=args.chromosome)

    trim_stats = trimmed_bam(args.source_bam, out / "trimmed.bam", cap_r1, cap_r2)
    run_original_call(bam=out / "trimmed.bam", out_dir=out / "trimmed", **caller)
    unadjusted = mask_bam(args.source_bam, out / "unadjusted.bam", cap_r1_nt=cap_r1, cap_r2_nt=cap_r2,
                          adjust_mate_start=False)
    run_original_call(bam=out / "unadjusted.bam", out_dir=out / "unadjusted", **caller)

    result: dict[str, object] = {"condition": args.condition, "cap_r1": cap_r1, "cap_r2": cap_r2,
                                 "source_bam": str(args.source_bam.resolve()),
                                 "masked_dir": str(args.masked_dir.resolve()),
                                 "trim_stats": trim_stats,
                                 "unadjusted_changed_records": unadjusted.get("n_changed_records", 0)}
    identical = True
    for target, name in METRIC_FILES.items():
        masked, trimmed, legacy = args.masked_dir / name, out / "trimmed" / name, out / "unadjusted" / name
        same = sha256(masked) == sha256(trimmed)
        identical &= same
        result[target] = {
            "masked_adjusted_equals_trimmed": same,
            "masked_adjusted": table_summary(masked),
            "trimmed": table_summary(trimmed),
            "masked_unadjusted": table_summary(legacy),
        }
    result["passed"] = identical
    (out / "trim_equivalence.json").write_text(json.dumps(result, indent=2) + "\n")
    for bam in ("trimmed.bam", "unadjusted.bam"):
        (out / bam).unlink()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=[c for c in RAW_READ_CAPS if c != "rawnoop"], required=True)
    parser.add_argument("--source-bam", type=Path, required=True)
    parser.add_argument("--masked-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--chalm-py", type=Path, required=True)
    parser.add_argument("--samtools-dir", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--chromosome")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
