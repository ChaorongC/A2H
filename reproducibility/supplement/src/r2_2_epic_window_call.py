"""EPIC-window inputs and per-contig parallel runs of the unchanged CHALM.py.

Two accelerations for R2-2 E2/E3 (2026-10-15); CHALM.py itself is not changed.

1. Window BAM.  The source BAM is reduced to records overlapping EPIC CpG
   +/- ``flank`` bp (merged BED; streaming ``samtools view -L``, which keeps
   sorted order.  On one native atlas sample its record stream was identical to the
   indexed ``-M -L`` extract (md5) and took 22 instead of 56 min on DFS3b).  In CHALM.py ``trad`` the counts at a CpG
   come only from reads covering it.  With ``-r`` such a read can be dropped by
   an earlier read with the same duplicate key (alignment start for forward,
   alignment end for reverse BSMAP strands), which therefore lies within two
   read lengths of the CpG.  The overlap cut of a left mate uses its own PNEXT;
   for E3 the moved PNEXT comes from the mate, which starts inside the left
   mate's span (or up to ~10 nt before it), i.e. within three read lengths.
   With reads of at most 151 nt, ``flank`` = 600 covers all of these, so rows at
   EPIC loci equal those of a whole-BAM run.  Other rows of a window-mode table
   are not valid and must not be used.
2. Per-contig calls.  CHALM.py keeps duplicate state and counts per chromosome,
   marks the reference per chromosome, and writes chromosomes in ``sorted()``
   order.  Each contig is therefore run separately (its records streamed
   through a FIFO, a one-contig FASTA as ``-d``), and the bodies are joined
   under one header in sorted contig order.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.r2_2_e3_aligned_call_cap import METRIC_FILES, sha256


DEFAULT_FLANK = 600


def window_bed(epic_mapping: Path, flank: int, output: Path) -> dict[str, object]:
    """Write merged EPIC locus +/- ``flank`` windows as a 0-based half-open BED."""

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    loci = pd.read_csv(epic_mapping, sep="\t", usecols=["locus"])["locus"]
    parts = loci.str.rsplit("_", n=1, expand=True)
    frames = []
    for contig, group in parts.groupby(0, sort=True):
        pos = np.sort(group[1].astype(np.int64).to_numpy())  # 1-based caller positions
        start, end = np.maximum(pos - 1 - flank, 0), pos + flank
        new = np.r_[True, start[1:] > np.maximum.accumulate(end)[:-1]]
        first = np.flatnonzero(new)
        frames.append(pd.DataFrame({"contig": contig, "start": start[first],
                                    "end": np.maximum.reduceat(end, first)}))
    bed = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(output, sep="\t", header=False, index=False)
    return {"epic_mapping": str(epic_mapping.resolve()), "epic_mapping_sha256": sha256(epic_mapping),
            "n_loci": int(len(loci)), "flank_bp": flank, "n_intervals": int(len(bed)),
            "covered_bp": int((bed["end"] - bed["start"]).sum()), "bed": str(output.resolve()),
            "bed_sha256": sha256(output)}


def split_reference(reference: Path, out_dir: Path) -> dict[str, object]:
    """Copy each FASTA record verbatim into ``<out_dir>/<contig>.fa``."""

    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    out_dir.mkdir(parents=True)
    contigs, handle = [], None
    with reference.open() as source:
        for line in source:
            if line.startswith(">"):
                if handle is not None:
                    handle.close()
                name = line[1:-1].split()[0]  # CHALM.py's own contig-name rule
                contigs.append(name)
                handle = (out_dir / f"{name}.fa").open("x")
            handle.write(line)
    handle.close()
    manifest = {"reference": str(reference.resolve()), "reference_sha256": sha256(reference),
                "contigs": contigs, "created_utc": datetime.now(timezone.utc).isoformat()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def flagstat_json(samtools: str, bam: Path) -> dict[str, int]:
    output = subprocess.run([samtools, "flagstat", "-O", "json", str(bam)], check=True,
                            capture_output=True, text=True).stdout
    passed = json.loads(output)["QC-passed reads"]
    return {key: int(passed[key]) for key in ("total", "read1", "read2", "properly paired",
                                               "secondary", "supplementary", "mapped")}


def extract_window_bam(samtools: str, source_bam: Path, bed: Path, out_dir: Path,
                       threads: int) -> dict[str, object]:
    """Write ``window.bam`` (+ index, flagstat, metadata) from an indexed source."""

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    bam = out_dir / "window.bam"
    command = [samtools, "view", "-@", str(threads), "-b", "-L", str(bed), "-o", str(bam), str(source_bam)]
    subprocess.run(command, check=True)
    subprocess.run([samtools, "index", "-@", str(threads), str(bam)], check=True)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_bam": str(source_bam.resolve()),
                "bed": str(bed.resolve()), "bed_sha256": sha256(bed), "command": command,
                "counts": flagstat_json(samtools, bam), "source_counts_from_index": idxstats(samtools, source_bam),
                "valid_rows": "EPIC loci only", "state": "complete"}
    (out_dir / "window_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def idxstats(samtools: str, bam: Path) -> dict[str, int]:
    """Mapped record count per contig (contigs with no records omitted)."""

    output = subprocess.run([samtools, "idxstats", str(bam)], check=True, capture_output=True, text=True).stdout
    counts = {}
    for line in output.splitlines():
        name, _length, mapped, _unmapped = line.split("\t")
        if name != "*" and int(mapped) > 0:
            counts[name] = int(mapped)
    return counts


def caller_commands(python_executable: str, chalm_py: Path, reference: Path, samtools_dir: Path,
                    min_depth: int, bam: Path, outputs: dict[str, Path]) -> dict[str, list[str]]:
    """The historical ``trad`` commands of ``run_original_call``."""

    common = [python_executable, str(chalm_py), "trad", "-d", str(reference), "-x", "CG", "-i", "no-action",
              "-m", str(min_depth), "-p", "-r", "-u", "-g", "-q", "-s", str(samtools_dir)]
    return {"methRatio": common + ["-o", str(outputs["methRatio"]), str(bam)],
            "CHALM": common + ["-l", "1", "-o", str(outputs["CHALM"]), str(bam)]}


def _run_part(job: tuple) -> dict[str, object]:
    samtools, bam, contig, command, fifo, part_log = job
    os.mkfifo(fifo)
    try:
        with part_log.open("w") as log:
            log.write("$ " + shlex.join(command) + "\n")
            log.flush()
            producer = subprocess.Popen([samtools, "view", "-u", "-o", str(fifo), str(bam), contig],
                                        stderr=log)
            caller = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            caller_code = caller.wait()
            if caller_code != 0 and producer.poll() is None:
                producer.kill()  # the caller never opened the FIFO; do not block forever
            producer_code = producer.wait()
            log.write(f"\ncaller_exit={caller_code} producer_exit={producer_code}\n")
    finally:
        fifo.unlink()
    if caller_code != 0 or producer_code != 0:
        raise RuntimeError(f"Per-contig call failed for {contig}: see {part_log}")
    return {"contig": contig, "command": command}


def call_by_contig(bam: Path, out_dir: Path, contig_fasta_dir: Path, chalm_py: Path, samtools_dir: Path,
                   python_executable: str, min_depth: int, samtools: str, processes: int) -> dict[str, object]:
    """Run both historical calls per contig in parallel and join them in sorted order."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(f"{bam}.bai").exists():
        subprocess.run([samtools, "index", str(bam)], check=True)
    counts = idxstats(samtools, bam)
    known = set(json.loads((contig_fasta_dir / "manifest.json").read_text())["contigs"])
    if set(counts) - known:
        raise ValueError(f"Contigs without a reference FASTA: {sorted(set(counts) - known)[:5]}")
    parts_dir = out_dir / "contig_parts"
    parts_dir.mkdir()
    fifo_dir = Path(tempfile.mkdtemp(prefix="chalm_fifo_"))  # node-local ($TMPDIR); FIFOs stay off BeeGFS
    jobs, parts = [], {}
    for contig in sorted(counts, key=lambda name: -counts[name]):
        outputs = {target: parts_dir / f"{target}.{contig}.txt" for target in METRIC_FILES}
        fifo = fifo_dir / f"{contig}.%s.bam"
        commands = caller_commands(python_executable, chalm_py, contig_fasta_dir / f"{contig}.fa", samtools_dir,
                                   min_depth, Path(str(fifo)), outputs)
        for target, command in commands.items():
            target_fifo = Path(str(fifo) % target)
            command = command[:-1] + [str(target_fifo)]
            jobs.append((samtools, bam, contig, command, target_fifo, parts_dir / f"{target}.{contig}.log"))
            parts[(target, contig)] = outputs[target]
    try:
        with ThreadPoolExecutor(max_workers=processes) as pool:
            list(pool.map(_run_part, jobs))
    finally:
        shutil.rmtree(fifo_dir, ignore_errors=True)
    result = {"mode": "per_contig", "processes": processes, "contig_fasta_dir": str(contig_fasta_dir.resolve()),
              "records_per_contig": counts, "commands_template": caller_commands(
                  python_executable, chalm_py, contig_fasta_dir / "<contig>.fa", samtools_dir, min_depth,
                  Path("<contig FIFO>"), {t: Path(f"<{t} part>") for t in METRIC_FILES}),
              "outputs": {}}
    for target, name in METRIC_FILES.items():
        output = out_dir / name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        header = None
        with output.open("w") as joined:
            for contig in sorted(counts):  # CHALM.py writes sorted(depth.keys())
                with parts[(target, contig)].open() as part:
                    first = part.readline()
                    if header is None:
                        header = first
                        joined.write(header)
                    elif first != header:
                        raise RuntimeError(f"Header differs in {parts[(target, contig)]}")
                    for line in part:
                        joined.write(line)
        log = out_dir / f"{target}.log"
        with log.open("w") as handle:
            for contig in sorted(counts):
                handle.write((parts_dir / f"{target}.{contig}.log").read_text())
        result["outputs"][target] = {"output": str(output.resolve()), "sha256": sha256(output),
                                     "log": str(log.resolve())}
    for path in parts_dir.iterdir():
        path.unlink()
    parts_dir.rmdir()
    return result


def epic_rows(table: Path, loci: set[str]) -> dict[str, str]:
    """Raw caller lines at the given ``chr_pos`` loci (header excluded)."""

    rows = {}
    with table.open() as handle:
        handle.readline()
        for line in handle:
            first = line.find("\t")
            key = line[:first] + "_" + line[first + 1:line.find("\t", first + 1)]
            if key in loci:
                rows[key] = line
    return rows


def compare_epic_rows(reference_table: Path, test_table: Path, epic_mapping: Path) -> dict[str, object]:
    """Exact line comparison at EPIC loci; presence must match too."""

    loci = set(pd.read_csv(epic_mapping, sep="\t", usecols=["locus"])["locus"])
    reference, test = epic_rows(reference_table, loci), epic_rows(test_table, loci)
    shared = reference.keys() & test.keys()
    differing = sorted(key for key in shared if reference[key] != test[key])
    result = {"reference": str(reference_table), "test": str(test_table), "n_epic": len(loci),
              "n_reference_rows": len(reference), "n_test_rows": len(test),
              "n_only_reference": len(reference.keys() - test.keys()), "n_only_test": len(test.keys() - reference.keys()),
              "n_differing": len(differing), "examples": differing[:5]}
    result["identical"] = (result["n_only_reference"] == 0 and result["n_only_test"] == 0
                           and result["n_differing"] == 0)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bed = sub.add_parser("bed")
    bed.add_argument("--epic-mapping", type=Path, required=True)
    bed.add_argument("--flank", type=int, default=DEFAULT_FLANK)
    bed.add_argument("--output", type=Path, required=True)
    ref = sub.add_parser("split-reference")
    ref.add_argument("--reference", type=Path, required=True)
    ref.add_argument("--out-dir", type=Path, required=True)
    win = sub.add_parser("window")
    win.add_argument("--samtools", required=True)
    win.add_argument("--source-bam", type=Path, required=True)
    win.add_argument("--bed", type=Path, required=True)
    win.add_argument("--out-dir", type=Path, required=True)
    win.add_argument("--threads", type=int, default=1)
    call = sub.add_parser("call")
    call.add_argument("--bam", type=Path, required=True)
    call.add_argument("--out-dir", type=Path, required=True)
    call.add_argument("--contig-fasta-dir", type=Path, required=True)
    call.add_argument("--chalm-py", type=Path, required=True)
    call.add_argument("--samtools-dir", type=Path, required=True)
    call.add_argument("--python-executable", required=True)
    call.add_argument("--min-depth", type=int, default=4)
    call.add_argument("--samtools", required=True)
    call.add_argument("--processes", type=int, default=1)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--reference-table", type=Path, required=True)
    cmp_.add_argument("--test-table", type=Path, required=True)
    cmp_.add_argument("--epic-mapping", type=Path, required=True)
    cmp_.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "bed":
        result = window_bed(args.epic_mapping, args.flank, args.output)
        args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "split-reference":
        result = split_reference(args.reference, args.out_dir)
        result = {k: v for k, v in result.items() if k != "contigs"} | {"n_contigs": len(result["contigs"])}
    elif args.command == "window":
        result = extract_window_bam(args.samtools, args.source_bam, args.bed, args.out_dir, args.threads)
    elif args.command == "call":
        result = call_by_contig(args.bam, args.out_dir, args.contig_fasta_dir, args.chalm_py, args.samtools_dir,
                                args.python_executable, args.min_depth, args.samtools, args.processes)
        (args.out_dir / "call_metadata.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    else:
        result = compare_epic_rows(args.reference_table, args.test_table, args.epic_mapping)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        if not result["identical"]:
            print(json.dumps(result, indent=2))
            raise SystemExit(1)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
