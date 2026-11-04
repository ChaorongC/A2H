"""Finish a native recovery whose sorted BAM is validated but whose calls are not done.

The original recovery directory is only read.  A new directory gets symlinks to
its ``sorted.bam``, index and ``flagstat.txt``, the two historical CHALM.py
``trad`` calls run per contig (``src/r2_2_epic_window_call.call_by_contig``;
byte-identical to a whole-genome run, validated on one native atlas sample), and a
``recovery_metadata.json`` that copies the original runner record and adds the
call stage.  The completion state is the runner's own.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.r2_2_e3_aligned_call_cap import sha256
from src.r2_2_epic_window_call import call_by_contig

COMPLETE_STATE = "native_calls_complete_unverified_archive"
LINKED = ("sorted.bam", "sorted.bam.bai", "flagstat.txt")
REQUIRED_STAGES = ("sorted_quickcheck", "full_sorted_decode_count", "historical_index", "flagstat")


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def finish(source_dir: Path, out_dir: Path, contig_fasta_dir: Path, chalm_py: Path, samtools_dir: Path,
           python_executable: str, samtools: str, processes: int) -> dict[str, object]:
    source = json.loads((source_dir / "recovery_metadata.json").read_text())
    done = {c["stage"] for c in source["commands"] if c.get("exit_code") == 0}
    if missing := [stage for stage in REQUIRED_STAGES if stage not in done]:
        raise RuntimeError(f"Source BAM stages not complete: {missing}")
    if source.get("expected_filtered_records") != source.get("validated_sorted_records"):
        raise RuntimeError("Source record counts do not match")
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    before = {name: fingerprint(source_dir / name) for name in LINKED}
    out_dir.mkdir(parents=True)
    for name in LINKED:
        os.symlink((source_dir / name).resolve(), out_dir / name)
    metadata = dict(source)
    metadata.update({"state": "running", "stage": "calls_per_contig", "finished_from": str(source_dir.resolve()),
                     "finished_from_state": source["state"], "linked_inputs": before,
                     "finish_started_utc": datetime.now(timezone.utc).isoformat()})
    meta_path = out_dir / "recovery_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    calls = call_by_contig(out_dir / "sorted.bam", out_dir, contig_fasta_dir, chalm_py, samtools_dir,
                           python_executable, source["min_depth"], samtools, processes)
    if {name: fingerprint(source_dir / name) for name in LINKED} != before:
        raise RuntimeError("Source BAM, index or flagstat changed during the calls")
    metadata["commands"] = source["commands"] + [{"stage": "calls_per_contig", "processes": processes,
                                                  "call_metadata": calls, "exit_code": 0}]
    metadata["call_sha256"] = {t: sha256(out_dir / f"{t}.txt") for t in ("methRatio", "CHALM")}
    metadata["state"] = COMPLETE_STATE
    metadata["finish_completed_utc"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contig-fasta-dir", type=Path, required=True)
    parser.add_argument("--chalm-py", type=Path, required=True)
    parser.add_argument("--samtools-dir", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--samtools", required=True)
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()
    result = finish(args.source_dir, args.out_dir, args.contig_fasta_dir, args.chalm_py, args.samtools_dir,
                    args.python_executable, args.samtools, args.processes)
    print(json.dumps({k: result[k] for k in ("sample_id", "state", "call_sha256")}, indent=2))


if __name__ == "__main__":
    main()
