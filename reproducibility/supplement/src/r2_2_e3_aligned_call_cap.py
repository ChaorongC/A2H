"""Run the E3 aligned-read observation-length cap with the original caller.

The source BAM is copied logically, not remapped: for a capped condition only
query bases outside the original sequencer-oriented 5-prime prefix are changed
to ``N``.  Flags, coordinates, CIGAR, mate fields, qualities and tags remain
unchanged.  The unchanged historical CHALM.py ``trad`` command then performs
all methylation counting, overlap handling, duplicate filtering and >=4-site
eligibility.

In the standard SAM/BAM representation a reverse-aligned read stores the
reverse-complemented sequence.  Therefore its original sequencer 5-prime
prefix is the suffix of the stored query sequence.  This assumption is
recorded in the run metadata and is checked by the pilot fixtures.

The ``raw*`` conditions (2026-10-15) cap each mate separately at the trimmed
length of a shorter raw read (historical trimming removes 20 nt from R1 and
30 nt from R2) and also move the left mate's PNEXT to the first retained base
of its capped mate.  CHALM.py drops the left mate's bases from PNEXT onward to
avoid double-counting mate overlap; with PNEXT left at the uncapped mate start,
bases that a shorter mate no longer covers would be lost from both mates.
Legacy conditions (noop, cap125, cap100) are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path


METRIC_FILES = {
    "methRatio": "methRatio.txt",
    "CHALM": "CHALM.txt",
}

# Raw (pre-trim) read length -> (R1, R2) cap on the trimmed, aligned reads.
RAW_READ_CAPS = {
    "rawnoop": (10000, 10000),
    "raw125": (105, 95),
    "raw100": (80, 70),
    "raw75": (55, 45),
}


def mask_query_sequence(sequence: str, is_reverse: bool, cap_nt: int) -> tuple[str, int]:
    """Mask post-cap stored query bases and return ``(sequence, n_masked)``."""

    if cap_nt < 0:
        raise ValueError("cap_nt must be non-negative")
    length = len(sequence)
    if cap_nt >= length:
        return sequence, 0
    chars = list(sequence)
    if is_reverse:
        # Stored SAM sequence is alignment-oriented for reverse reads; the
        # sequencer 5-prime prefix is therefore the stored suffix.
        start = 0
        stop = length - cap_nt
    else:
        start = cap_nt
        stop = length
    for index in range(start, stop):
        chars[index] = "N"
    return "".join(chars), stop - start


def _stable_identity(read, include_pnext: bool = True) -> str:
    """Return an identity string excluding sequence and quality fields."""

    tags = tuple((str(key), repr(value)) for key, value in read.get_tags())
    fields = (
        read.query_name,
        int(read.flag),
        int(read.reference_id),
        int(read.reference_start),
        int(read.mapping_quality),
        read.cigarstring,
        int(read.next_reference_id),
        int(read.next_reference_start) if include_pnext else None,
        int(read.template_length),
        tags,
    )
    return repr(fields)


def _update_digest(digest, value: str) -> None:
    digest.update(value.encode("utf-8"))
    digest.update(b"\n")


def alignment_identity_digest(path: Path, include_pnext: bool = True, threads: int = 1) -> str:
    """Hash alignment and template fields while excluding sequence/quality."""

    try:
        import pysam
    except ImportError as exc:  # pragma: no cover - exercised on cluster only
        raise RuntimeError("alignment_identity_digest requires pysam") from exc
    digest = hashlib.sha256()
    with pysam.AlignmentFile(str(path), "rb", threads=threads) as handle:
        for read in handle.fetch(until_eof=True):
            _update_digest(digest, _stable_identity(read, include_pnext))
    return digest.hexdigest()


def capped_reference_start(read, cap_nt: int) -> int | None:
    """0-based reference start of a read's retained sequencer 5-prime prefix."""

    length = read.query_length
    if cap_nt >= length or not read.is_reverse:
        return read.reference_start
    first_kept = length - cap_nt  # stored-orientation index of the first retained base
    if read.cigartuples == [(0, length)]:
        return read.reference_start + first_kept
    for query_pos, reference_pos in read.get_aligned_pairs(matches_only=True):
        if query_pos >= first_kept:
            return reference_pos
    return None


def _move_mate_start(read, mate_start: int | None, lead_cap: int, stats: Counter) -> None:
    """Set the lead's PNEXT so CHALM's overlap cut equals the cut on capped reads.

    CHALM keeps ``seq[:PNEXT0 - pos]`` of the TLEN > 0 lead.  When the capped
    mate still starts left of the lead (dovetailed short fragment) the index is
    negative and removes the lead's last bases; on a physically capped lead
    those are real bases, so the masked full-length lead gets the equivalent
    non-negative cut ``pos + max(cap - k, 0)`` instead.
    """

    if mate_start is None:
        stats["n_mate_start_unresolved"] += 1
        return
    if mate_start < read.next_reference_start:
        raise RuntimeError(f"Capped mate start moved left for {read.query_name}")
    target = mate_start
    if mate_start < read.reference_start and read.query_length > lead_cap:
        if read.is_reverse:
            stats["n_reverse_lead_negative_cut"] += 1
        else:
            target = read.reference_start + max(lead_cap - (read.reference_start - mate_start), 0)
            stats["n_dovetail_lead_cut"] += 1
    if target > read.next_reference_start:
        read.next_reference_start = target
        stats["n_mate_start_moved"] += 1


def mask_bam(
    source_bam: Path,
    output_bam: Path,
    cap_nt: int | None = None,
    *,
    cap_r1_nt: int | None = None,
    cap_r2_nt: int | None = None,
    adjust_mate_start: bool = False,
    threads: int = 1,
    contig: str | None = None,
) -> dict[str, object]:
    """Write a BAM with only post-cap query bases masked to ``N``.

    With ``adjust_mate_start`` the left mate (TLEN > 0, proper pair) gets the
    capped start of its mate as PNEXT.  Records are buffered only until their
    mate is seen, so the source record order is preserved exactly.
    """

    try:
        import pysam
    except ImportError as exc:  # pragma: no cover - exercised on cluster only
        raise RuntimeError("mask_bam requires pysam in the execution environment") from exc

    if output_bam.exists():
        raise FileExistsError(f"Refusing to overwrite {output_bam}")
    output_bam.parent.mkdir(parents=True, exist_ok=True)
    caps = {
        "R1": cap_r1_nt if cap_r1_nt is not None else cap_nt,
        "R2": cap_r2_nt if cap_r2_nt is not None else cap_nt,
        "unassigned": cap_nt,
    }
    if caps["R1"] is None or caps["R2"] is None:
        raise ValueError("Both mates need a cap")
    include_pnext = not adjust_mate_start

    original_digest = hashlib.sha256()
    stats = Counter()
    lengths: Counter[tuple[str, int]] = Counter()
    pending: dict[str, list] = {}
    # Mates (TLEN < 0) seen before their TLEN > 0 lead: equal starts, and
    # dovetailed short fragments where the reverse mate starts left of the lead.
    early_mates: dict[str, int | None] = {}
    early_order: deque[tuple[tuple[int, int], str]] = deque()
    queue: deque[list] = deque()

    with pysam.AlignmentFile(str(source_bam), "rb", threads=threads) as source, pysam.AlignmentFile(
        str(output_bam), "wb", template=source, threads=threads
    ) as destination:
        reads = source.fetch(contig) if contig is not None else source.fetch(until_eof=True)
        for read in reads:
            _update_digest(original_digest, _stable_identity(read, include_pnext))
            stats["n_records"] += 1
            if read.is_read1:
                mate = "R1"
            elif read.is_read2:
                mate = "R2"
            else:
                mate = "unassigned"
            sequence = read.query_sequence
            if sequence is None:
                stats["n_missing_sequence"] += 1
                destination.write(read)
                continue
            if caps[mate] is None:
                raise ValueError(f"No cap for {mate} record {read.query_name}")
            length = len(sequence)
            lengths[(mate, length)] += 1
            stats[f"n_{mate}_records"] += 1
            stats[f"n_{mate}_bases"] += length
            if read.is_reverse:
                stats["n_reverse_records"] += 1
            else:
                stats["n_forward_records"] += 1
            capped_start = capped_reference_start(read, caps[mate]) if adjust_mate_start else None
            capped, n_masked = mask_query_sequence(sequence, read.is_reverse, caps[mate])
            if n_masked:
                qualities = read.query_qualities
                read.query_sequence = capped
                if qualities is not None:
                    read.query_qualities = qualities
                stats["n_changed_records"] += 1
                stats["n_masked_query_bases"] += n_masked
            else:
                stats["n_unchanged_records"] += 1
            if not adjust_mate_start:
                destination.write(read)
                continue

            position = (read.reference_id, read.reference_start)
            while early_order and early_order[0][0] < position:
                # The stream has passed the expected lead start: lead absent.
                if early_mates.pop(early_order.popleft()[1], "resolved") != "resolved":
                    stats["n_lead_absent"] += 1
            entry = [read, True, None]
            if read.is_proper_pair and read.template_length > 0:
                if read.query_name in early_mates:
                    _move_mate_start(read, early_mates.pop(read.query_name), caps[mate], stats)
                    stats["n_mate_before_lead"] += 1
                else:
                    entry[1] = False
                    entry[2] = (read.next_reference_id, read.next_reference_start)
                    pending[read.query_name] = entry
            elif read.is_proper_pair and read.template_length < 0:
                lead = pending.pop(read.query_name, None)
                if lead is not None:
                    _move_mate_start(lead[0], capped_start, caps["R1" if lead[0].is_read1 else "R2"], stats)
                    lead[1] = True
                elif (read.next_reference_id, read.next_reference_start) >= position:
                    # The TLEN > 0 lead starts here or later in sorted order.
                    early_mates[read.query_name] = capped_start
                    early_order.append(((read.next_reference_id, read.next_reference_start), read.query_name))
            queue.append(entry)
            while queue:
                head = queue[0]
                if not head[1]:
                    if position <= head[2]:
                        break
                    # The coordinate-sorted stream has passed the mate's start.
                    pending.pop(head[0].query_name, None)
                    head[1] = True
                    stats["n_mate_absent"] += 1
                destination.write(queue.popleft()[0])
        while queue:
            head = queue.popleft()
            if not head[1]:
                stats["n_mate_absent"] += 1
            destination.write(head[0])
        stats["n_lead_absent"] += len(early_mates)

    stats["n_length_bins"] = len(lengths)
    stats["length_counts"] = {
        f"{mate}:{length}": count
        for (mate, length), count in sorted(lengths.items())
    }
    stats["source_bam"] = str(source_bam.resolve())
    stats["output_bam"] = str(output_bam.resolve())
    stats["cap_nt"] = cap_nt
    stats["cap_r1_nt"] = caps["R1"]
    stats["cap_r2_nt"] = caps["R2"]
    stats["adjust_mate_start"] = adjust_mate_start
    stats["identity_excludes_pnext"] = adjust_mate_start
    stats["sequence_orientation"] = (
        "standard SAM/BAM stored query orientation; reverse FLAG maps the "
        "sequencer 5-prime prefix to the stored suffix"
    )
    stats["identity_sha256_source"] = original_digest.hexdigest()
    stats["identity_sha256_output"] = alignment_identity_digest(output_bam, include_pnext, threads)
    stats["alignment_identity_preserved"] = (
        stats["identity_sha256_source"] == stats["identity_sha256_output"]
    )
    if not stats["alignment_identity_preserved"]:
        raise RuntimeError("Derived BAM alignment identity digest changed")
    return dict(stats)


def _mask_contig(job: tuple) -> tuple[str, dict[str, object]]:
    source_bam, part_bam, contig, kwargs = job
    return contig, mask_bam(source_bam, part_bam, contig=contig, **kwargs)


def mask_bam_by_contig(source_bam: Path, output_bam: Path, *, processes: int, samtools: str,
                       **kwargs) -> dict[str, object]:
    """Run ``mask_bam`` per contig in parallel and concatenate in header order.

    Proper pairs never span contigs, and a coordinate-sorted BAM stores contigs
    in header order, so ``samtools cat`` of the parts reproduces the source
    record order.  Each part verifies its own alignment-identity digest.
    """

    import multiprocessing

    import pysam

    if output_bam.exists():
        raise FileExistsError(f"Refusing to overwrite {output_bam}")
    with pysam.AlignmentFile(str(source_bam), "rb") as handle:
        if handle.header.get("HD", {}).get("SO") != "coordinate":
            raise ValueError("Per-contig masking requires a coordinate-sorted BAM")
        index = {s.contig: s.total for s in handle.get_index_statistics()}
        if handle.nocoordinate:
            raise ValueError("Unplaced records cannot be masked per contig")
        order = [name for name in handle.references if index.get(name, 0) > 0]
    parts_dir = output_bam.parent / f"{output_bam.name}.parts"
    parts_dir.mkdir()
    parts = {name: parts_dir / f"{i:04d}.bam" for i, name in enumerate(order)}
    jobs = [(source_bam, parts[name], name, kwargs) for name in sorted(order, key=lambda n: -index[n])]
    with multiprocessing.get_context("fork").Pool(processes) as pool:
        results = dict(pool.imap_unordered(_mask_contig, jobs))
    subprocess.run([samtools, "cat", "-o", str(output_bam)] + [str(parts[name]) for name in order], check=True)
    subprocess.run([samtools, "quickcheck", "-v", str(output_bam)], check=True)
    for part in parts.values():
        part.unlink()
    parts_dir.rmdir()
    stats: Counter = Counter()
    lengths: Counter = Counter()
    for name in order:
        part = results[name]
        for key, value in part.items():
            if key.startswith("n_") and isinstance(value, int):
                stats[key] += value
        lengths.update(part["length_counts"])
    if stats["n_records"] != sum(index[name] for name in order):
        raise RuntimeError("Per-contig record total differs from the BAM index")
    merged = dict(stats)
    merged.update({key: results[order[0]][key] for key in ("cap_nt", "cap_r1_nt", "cap_r2_nt",
                   "adjust_mate_start", "identity_excludes_pnext", "sequence_orientation")})
    merged["length_counts"] = dict(sorted(lengths.items()))
    merged["n_length_bins"] = len(lengths)
    merged["n_contigs"] = len(order)
    merged["per_contig_identity"] = {name: results[name]["identity_sha256_source"] for name in order}
    merged["alignment_identity_preserved"] = all(results[name]["alignment_identity_preserved"] for name in order)
    merged["source_bam"] = str(source_bam.resolve())
    merged["output_bam"] = str(output_bam.resolve())
    merged["processes"] = processes
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_command(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nexit_code={completed.returncode}\n")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def run_original_call(
    bam: Path,
    out_dir: Path,
    reference: Path,
    chalm_py: Path,
    samtools_dir: Path,
    python_executable: str,
    min_depth: int,
    chromosome: str | None,
    parallel: bool = False,
) -> dict[str, object]:
    """Run the two historical CHALM.py trad calls and return output metadata."""

    out_dir.mkdir(parents=True, exist_ok=True)
    chrom_args = ["-c", chromosome] if chromosome else []
    common = [
        python_executable,
        str(chalm_py),
        "trad",
        "-d",
        str(reference),
        "-x",
        "CG",
        "-i",
        "no-action",
        "-m",
        str(min_depth),
        "-p",
        "-r",
        "-u",
        "-g",
        "-q",
        "-s",
        str(samtools_dir),
    ] + chrom_args
    commands = {
        "methRatio": common + ["-o", str(out_dir / METRIC_FILES["methRatio"]), str(bam)],
        "CHALM": common + ["-l", "1", "-o", str(out_dir / METRIC_FILES["CHALM"]), str(bam)],
    }
    logs = {}
    if parallel:
        handles = {}
        for target, command in commands.items():
            log_handle = (out_dir / f"{target}.log").open("w", encoding="utf-8")
            log_handle.write("$ " + shlex.join(command) + "\n")
            log_handle.flush()
            handles[target] = (subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT), log_handle)
        for target, (process, log_handle) in handles.items():
            code = process.wait()
            log_handle.write(f"\nexit_code={code}\n")
            log_handle.close()
            if code != 0:
                raise subprocess.CalledProcessError(code, commands[target])
    for target, command in commands.items():
        log = out_dir / f"{target}.log"
        if not parallel:
            run_command(command, log)
        output = out_dir / METRIC_FILES[target]
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Caller produced no output: {output}")
        logs[target] = {"output": str(output.resolve()), "sha256": sha256(output), "log": str(log.resolve())}
    return {"commands": commands, "outputs": logs}


def _read_ratio_table(path: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not {"chr", "pos", "ratio"}.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected caller schema: {path}")
        rows = {}
        for row in reader:
            key = (row["chr"], row["pos"])
            if key in rows:
                raise ValueError(f"Duplicate caller locus {key} in {path}")
            rows[key] = row
    return list(reader.fieldnames), rows


def derive_camda(out_dir: Path) -> dict[str, object]:
    """Derive the historical CpG CAMDA matrix value as CHALM - methRatio."""

    _, methratio = _read_ratio_table(out_dir / METRIC_FILES["methRatio"])
    chalm_fields, chalm = _read_ratio_table(out_dir / METRIC_FILES["CHALM"])
    common = sorted(set(methratio) & set(chalm))
    only_methratio = sorted(set(methratio) - set(chalm))
    only_chalm = sorted(set(chalm) - set(methratio))
    if not common:
        raise RuntimeError("No common CpG loci between methRatio and CHALM outputs")
    output = out_dir / "CAMDA.tsv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    fields = ["chr", "pos", "strand", "context", "methRatio", "CHALM", "CAMDA"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for key in common:
            m = float(methratio[key]["ratio"])
            c = float(chalm[key]["ratio"])
            row = chalm[key]
            writer.writerow(
                {
                    "chr": row["chr"],
                    "pos": row["pos"],
                    "strand": row.get("strand", ""),
                    "context": row.get("context", ""),
                    "methRatio": f"{m:.12g}",
                    "CHALM": f"{c:.12g}",
                    "CAMDA": f"{c - m:.12g}",
                }
            )
    return {
        "output": str(output.resolve()),
        "sha256": sha256(output),
        "n_methratio_loci": len(methratio),
        "n_chalm_loci": len(chalm),
        "n_common_loci": len(common),
        "n_methratio_only": len(only_methratio),
        "n_chalm_only": len(only_chalm),
        "definition": "CAMDA = CHALM ratio - methRatio ratio, keyed by chr and pos",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.source_bam.exists():
        raise FileNotFoundError(args.source_bam)
    if args.condition == "native":
        condition_metadata = {
            "condition": "native",
            "source_bam": str(args.source_bam.resolve()),
            "cap_nt": None,
            "derived_bam": None,
            "alignment_identity_preserved": True,
        }
        call_bam = args.source_bam
    elif args.condition in RAW_READ_CAPS:
        if args.cap_nt is not None:
            raise ValueError("raw* conditions take their per-mate caps from RAW_READ_CAPS")
        cap_r1, cap_r2 = RAW_READ_CAPS[args.condition]
        derived_bam = out_dir / f"{args.sample_id}.{args.condition}.masked.bam"
        if args.processes > 1:
            condition_metadata = mask_bam_by_contig(args.source_bam, derived_bam, processes=args.processes,
                                                    samtools=args.samtools, cap_r1_nt=cap_r1, cap_r2_nt=cap_r2,
                                                    adjust_mate_start=True)
        else:
            condition_metadata = mask_bam(args.source_bam, derived_bam, cap_r1_nt=cap_r1, cap_r2_nt=cap_r2,
                                          adjust_mate_start=True, threads=args.threads)
        condition_metadata.update({"condition": args.condition, "sample_id": args.sample_id,
                                   "raw_to_trimmed": "R1 = raw - 20 nt, R2 = raw - 30 nt"})
        call_bam = derived_bam
    else:
        if args.cap_nt is None:
            raise ValueError("Capped conditions require --cap-nt")
        derived_bam = out_dir / f"{args.sample_id}.{args.condition}.masked.bam"
        condition_metadata = mask_bam(args.source_bam, derived_bam, args.cap_nt, threads=args.threads)
        condition_metadata.update({"condition": args.condition, "sample_id": args.sample_id})
        call_bam = derived_bam

    if args.window_dir is not None:
        window = json.loads((args.window_dir / "window_metadata.json").read_text())
        if window.get("state") != "complete" or (args.window_dir / "window.bam").resolve() != args.source_bam.resolve():
            raise RuntimeError(f"--source-bam is not the complete window extract in {args.window_dir}")
        condition_metadata["epic_window"] = {"window_dir": str(args.window_dir.resolve()),
                                             "native_source_bam": window["source_bam"],
                                             "bed_sha256": window["bed_sha256"], "valid_rows": "EPIC loci only"}
    if args.call_processes > 1:
        from src.r2_2_epic_window_call import call_by_contig

        calls = call_by_contig(call_bam, out_dir, args.contig_fasta_dir, args.chalm_py, args.samtools_dir,
                               args.python_executable, args.min_depth, args.samtools, args.call_processes)
    else:
        calls = run_original_call(
            bam=call_bam,
            out_dir=out_dir,
            reference=args.reference,
            chalm_py=args.chalm_py,
            samtools_dir=args.samtools_dir,
            python_executable=args.python_executable,
            min_depth=args.min_depth,
            chromosome=args.chromosome,
            parallel=args.parallel_callers,
        )
    if args.skip_camda_table:
        camda = {"skipped": True, "reason": "genome-wide table; CAMDA derived at EPIC loci downstream"}
    else:
        camda = derive_camda(out_dir)
    if args.remove_derived_bam and call_bam != args.source_bam:
        call_bam.unlink()
        Path(f"{call_bam}.bai").unlink(missing_ok=True)
        condition_metadata["derived_bam_removed"] = True
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": args.sample_id,
        "condition": args.condition,
        "source_bam": str(args.source_bam.resolve()),
        "source_bam_sha256": sha256(args.source_bam),
        "reference": str(args.reference.resolve()),
        "chalm_py": str(args.chalm_py.resolve()),
        "chalm_py_sha256": sha256(args.chalm_py),
        "samtools_dir": str(args.samtools_dir.resolve()),
        "python_executable": args.python_executable,
        "min_depth": args.min_depth,
        "context": "CG",
        "filters": ["-p", "-r", "-u", "-g", "-i no-action", "-F 268 at source BAM construction"],
        "condition_metadata": condition_metadata,
        "calls": calls,
        "camda": camda,
        "scientific_contract": {
            "cap_definition": "per-mate original sequencer 5-prime prefix; later 3-prime calls masked",
            "same_alignments": True,
            "left_mate_pnext_moved_to_capped_mate_start": args.condition in RAW_READ_CAPS,
            "same_templates": True,
            "no_remapping": True,
            "missing_values_remain_missing": True,
            "a2h_refit": False,
            "camda_definition": "CHALM - methRatio keyed by chr,pos",
        },
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--condition", choices=("native", "noop", "cap125", "cap100", *RAW_READ_CAPS),
                        required=True)
    parser.add_argument("--cap-nt", type=int)
    parser.add_argument("--threads", type=int, default=1, help="BAM (de)compression threads")
    parser.add_argument("--skip-camda-table", action="store_true")
    parser.add_argument("--processes", type=int, default=1,
                        help="raw* conditions: mask contigs in parallel and join with samtools cat")
    parser.add_argument("--samtools", default="samtools", help="samtools >=1.10 for cat/quickcheck")
    parser.add_argument("--parallel-callers", action="store_true",
                        help="Run the methRatio and CHALM calls concurrently")
    parser.add_argument("--remove-derived-bam", action="store_true",
                        help="Delete the masked BAM after the calls succeed")
    parser.add_argument("--source-bam", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--chalm-py", type=Path, required=True)
    parser.add_argument("--samtools-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--chromosome", help="Optional chromosome passed to original CHALM.py")
    parser.add_argument("--window-dir", type=Path,
                        help="EPIC-window extract used as --source-bam (r2_2_epic_window_call window); EPIC rows only")
    parser.add_argument("--call-processes", type=int, default=1,
                        help=">1: per-contig parallel calls (needs --contig-fasta-dir)")
    parser.add_argument("--contig-fasta-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
