import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest

from src.r2_2_e3_aligned_call_cap import run_original_call
from src.r2_2_epic_window_call import (call_by_contig, compare_epic_rows, extract_window_bam, split_reference,
                                       window_bed)

CHALM_PY = Path(os.environ.get("CHALM_PY", "tools/CHALM/src/CHALM.py"))
CALLER_PYTHON = os.environ.get("CHALM_PYTHON", "python")
LEGACY_SAMTOOLS = Path(os.environ.get("LEGACY_SAMTOOLS_DIR", "tools/samtools-0.1.19"))
SAMTOOLS = os.environ.get("SAMTOOLS") or shutil.which("samtools")
needs_caller = pytest.mark.skipif(
    not (CHALM_PY.exists() and Path(CALLER_PYTHON).exists() and (LEGACY_SAMTOOLS / "samtools").exists()
         and SAMTOOLS), reason="historical CHALM.py, py310, samtools-0.1.19 or samtools>=1.10 unavailable")

# Header order differs from sorted() order, which is CHALM.py's output order.
CONTIGS = {"chr1": 9000, "chr2": 7000, "chr10": 6000}


def test_window_bed_is_half_open_and_merged(tmp_path):
    mapping = tmp_path / "epic.tsv"
    pd.DataFrame({"locus": ["chr1_100", "chr1_300", "chr2_50", "chr1_900"]}).to_csv(mapping, sep="\t", index=False)
    result = window_bed(mapping, 100, tmp_path / "w.bed")
    bed = pd.read_csv(tmp_path / "w.bed", sep="\t", header=None)
    assert bed.values.tolist() == [["chr1", 0, 400], ["chr1", 799, 1000], ["chr2", 0, 150]]
    assert result["n_intervals"] == 3 and result["covered_bp"] == 400 + 201 + 150


def test_split_reference_is_verbatim(tmp_path):
    fasta = tmp_path / "ref.fa"
    fasta.write_text(">chr1 desc\nACGT\nAC\n>chr2\nGGCC\n")
    split_reference(fasta, tmp_path / "split")
    joined = (tmp_path / "split" / "chr1.fa").read_text() + (tmp_path / "split" / "chr2.fa").read_text()
    assert joined == fasta.read_text()


def _genome(rng):
    return {name: "".join(rng.choice(list("ACGT"), p=[0.25, 0.3, 0.2, 0.25], size=length))
            for name, length in CONTIGS.items()}


def _bisulfite(fragment: str, rng) -> str:
    """Top-strand bisulfite read: non-CpG C -> T, CpG C -> T with p = 0.4."""

    out = []
    for i, base in enumerate(fragment):
        if base == "C":
            cpg = i + 1 < len(fragment) and fragment[i + 1] == "G"
            out.append("T" if (not cpg or rng.random() < 0.4) else "C")
        else:
            out.append(base)
    return "".join(out)


def _write_inputs(tmp_path, rng):
    genome = _genome(rng)
    fasta = tmp_path / "ref.fa"
    with fasta.open("w") as handle:
        for name, seq in genome.items():
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                handle.write(seq[i:i + 60] + "\n")
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": name, "LN": length} for name, length in CONTIGS.items()]}
    pysam_header = pysam.AlignmentHeader.from_dict(header)
    records = []
    for tid, (name, length) in enumerate(CONTIGS.items()):
        for n in range(500):
            fragment = int(rng.integers(90, 260))
            start = int(rng.integers(0, length - fragment - 5))
            if n % 7 == 0 and records and records[-2][0] == tid:  # forward duplicate start
                start = records[-2][1]
                fragment = min(fragment, length - start - 5)
            r1_len, r2_len = min(131, fragment), min(121, fragment)
            r2_start = start + fragment - r2_len
            if n % 10 == 0:  # dovetail: reverse mate starts left of the forward lead
                r2_start = max(start - 10, 0)
            for mate, (pos, rlen, flag) in enumerate([(start, r1_len, 99), (r2_start, r2_len, 147)]):
                read = pysam.AlignedSegment(pysam_header)
                read.query_name = f"{name}_{n}"
                read.flag = flag
                read.reference_id = tid
                read.reference_start = pos
                read.mapping_quality = 255
                read.cigartuples = [(0, rlen)]
                read.next_reference_id = tid
                read.next_reference_start = r2_start if mate == 0 else start
                tlen = max(start + r1_len, r2_start + r2_len) - min(start, r2_start)
                read.template_length = tlen if mate == 0 else -tlen
                read.query_sequence = _bisulfite(genome[name][pos:pos + rlen + 1], rng)[:rlen]
                read.query_qualities = pysam.qualitystring_to_array("I" * rlen)
                read.set_tag("ZS", "++" if mate == 0 else "+-")
                records.append((tid, pos, read))
    records.sort(key=lambda item: (item[0], item[1]))
    bam = tmp_path / "in.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as handle:
        for _tid, _pos, read in records:
            handle.write(read)
    pysam.index(str(bam))
    return genome, fasta, bam


@needs_caller
def test_per_contig_calls_equal_one_whole_run_and_window_rows_at_loci(tmp_path):
    rng = np.random.default_rng(7)
    genome, fasta, bam = _write_inputs(tmp_path, rng)
    split_reference(fasta, tmp_path / "by_contig")
    whole = run_original_call(bam, tmp_path / "whole", fasta, CHALM_PY, LEGACY_SAMTOOLS, CALLER_PYTHON, 4, None)
    parallel = call_by_contig(bam, tmp_path / "parallel", tmp_path / "by_contig", CHALM_PY, LEGACY_SAMTOOLS,
                              CALLER_PYTHON, 4, SAMTOOLS, processes=4)
    for target in ("methRatio", "CHALM"):
        whole_text = Path(whole["outputs"][target]["output"]).read_text()
        assert whole_text.count("\n") > 50
        assert Path(parallel["outputs"][target]["output"]).read_text() == whole_text
    # Windowed input: rows at the window centres must be unchanged.
    cpgs = [f"{name}_{i + 1}" for name, seq in genome.items() for i in range(len(seq) - 1) if seq[i:i + 2] == "CG"]
    loci = list(rng.choice(cpgs, size=40, replace=False))
    mapping = tmp_path / "epic.tsv"
    pd.DataFrame({"locus": loci}).to_csv(mapping, sep="\t", index=False)
    window_bed(mapping, 600, tmp_path / "w.bed")
    window = extract_window_bam(SAMTOOLS, bam, tmp_path / "w.bed", tmp_path / "window", threads=1)
    assert window["counts"]["total"] < 3000  # 3,000 records in the input; the windows drop some
    call_by_contig(tmp_path / "window" / "window.bam", tmp_path / "wcall", tmp_path / "by_contig", CHALM_PY,
                   LEGACY_SAMTOOLS, CALLER_PYTHON, 4, SAMTOOLS, processes=2)
    for name in ("methRatio.txt", "CHALM.txt"):
        result = compare_epic_rows(tmp_path / "whole" / name, tmp_path / "wcall" / name, mapping)
        assert result["identical"], result
        assert result["n_reference_rows"] > 10
