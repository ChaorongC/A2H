"""Verify four raw E2 FASTQs and concatenate gzip lanes without changing read order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from r2_2_select_read_samples import OUT


def transfer_with_md5(source: Path, dest, merged_sha256) -> tuple[int, str]:
    md5 = hashlib.md5()
    n_bytes = 0
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            md5.update(block)
            merged_sha256.update(block)
            dest.write(block)
            n_bytes += len(block)
    expected = Path(f"{source}.md5").read_text().split()[0].lower()
    if md5.hexdigest() != expected:
        raise ValueError(f"Published MD5 mismatch for {source}")
    return n_bytes, md5.hexdigest()


def prepare(manifest: Path, sample_id: str, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    with manifest.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row["sample_id"] == sample_id]
    if len(rows) != 1 or rows[0]["e2_candidate"] != "yes":
        raise ValueError(f"Expected one frozen E2 sample: {sample_id}")
    row = rows[0]
    out.mkdir(parents=True)
    inputs = []
    outputs = []
    for mate in ("1", "2"):
        destination = out / f"{sample_id}_R{mate}.fastq.gz"
        sha256 = hashlib.sha256()
        with destination.open("xb") as handle:
            for lane in ("001", "002"):
                source = Path(row[f"L{lane}_R{mate}_fastq"])
                n_bytes, md5 = transfer_with_md5(source, handle, sha256)
                if n_bytes != int(row[f"L{lane}_R{mate}_bytes"]):
                    raise ValueError(f"Frozen FASTQ byte size changed: {source}")
                inputs.append({"lane": lane, "mate": mate, "source": str(source),
                               "bytes": n_bytes, "verified_md5": md5})
        outputs.append({"mate": mate, "merged_gzip_path": str(destination),
                        "merged_gzip_bytes": destination.stat().st_size,
                        "merged_gzip_sha256": sha256.hexdigest()})
    provenance = {
        "sample_id": sample_id,
        "source_manifest": str(manifest.resolve()),
        "lane_order": ["001", "002"],
        "inputs": inputs,
        "outputs": outputs,
        "all_source_md5_verified": True,
        "gzip_integrity_checked": False,
        "pair_counts_checked": False,
    }
    (out / "native_input_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=OUT / "E2_E3_sample_selection.tsv")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.manifest, args.sample_id, args.out)


if __name__ == "__main__":
    main()
