"""Freeze outcome-blind Loyfer atlas samples for R2-2 E2/E3 read pilots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


METADATA = Path("data/atlas_sample_info.csv")
ATLAS = Path(__file__).resolve().parents[1] / "results/atlas_splits/atlas_samples.tsv"
RAW = Path("data/atlas_fastq")
OUT = Path(__file__).resolve().parents[1] / "results/r2_2_sensitivity_20261011/E2_E3_access/selection_39_groups"
FASTQ_NAME = re.compile(r"^(Z[0-9A-Z]+)_L(00[12])_R([12])\.fastq\.gz$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def freeze_samples(metadata: list[dict[str, str]], atlas: list[dict[str, str]]) -> list[dict[str, str | float]]:
    if len(metadata) != 205 or len(atlas) != 205:
        raise ValueError("Expected 205 metadata and frozen-atlas rows")
    atlas_by_id = {row["sample_id"]: row for row in atlas}
    if len(atlas_by_id) != 205:
        raise ValueError("Duplicate frozen-atlas sample ID")
    candidates: list[dict[str, str | float]] = []
    for row in metadata:
        match = re.search(r"-(Z[0-9A-Z]+)$", row["Sample name"])
        if not match:
            raise ValueError(f"Cannot parse atlas ID: {row['Sample name']}")
        sample_id = match.group(1)
        frozen = atlas_by_id.get(sample_id)
        if frozen is None or frozen["Group"] != row["Group"]:
            raise ValueError(f"Atlas manifest mismatch for {sample_id}")
        if frozen["Source"] != row["Source"] or frozen["PatientID"] != row["PatientID"]:
            raise ValueError(f"Donor metadata mismatch for {sample_id}")
        candidates.append({
            "sample_id": sample_id,
            "sample_name": row["Sample name"],
            "group": row["Group"],
            "refined_group": row["Refined group"],
            "source_tissue": row["Source Tissue"],
            "cell_type": row["Cell type"],
            "germ_layer": row["Germ layer"],
            "donor_id": frozen["donor_id"],
            "published_cpg_fragment_coverage": float(row["CpG coverage (fragments)"]),
        })
    if len({row["sample_id"] for row in candidates}) != 205:
        raise ValueError("Duplicate metadata sample ID")
    by_group: dict[str, list[dict[str, str | float]]] = {}
    for row in candidates:
        by_group.setdefault(str(row["group"]), []).append(row)
    if len(by_group) != 39:
        raise ValueError(f"Expected 39 atlas groups, found {len(by_group)}")
    chosen = [min(rows, key=lambda r: (-float(r["published_cpg_fragment_coverage"]), str(r["sample_id"])))
              for _, rows in sorted(by_group.items())]
    return chosen


def select_e3(chosen: list[dict[str, str | float]]) -> set[str]:
    """Four per germ layer; greedy metadata diversity, then native coverage."""
    selected: set[str] = set()
    for layer in ("Ecto", "Endo", "Meso"):
        pool = [row for row in chosen if row["germ_layer"] == layer]
        if len(pool) < 4:
            raise ValueError(f"Fewer than four candidate groups in {layer}")
        seen_cell_types: set[str] = set()
        seen_tissues: set[str] = set()
        for _ in range(4):
            winner = min(pool, key=lambda r: (
                -(int(str(r["cell_type"]) not in seen_cell_types)
                  + int(str(r["source_tissue"]) not in seen_tissues)),
                -int(str(r["source_tissue"]) not in seen_tissues),
                -float(r["published_cpg_fragment_coverage"]),
                str(r["sample_id"]),
            ))
            selected.add(str(winner["sample_id"]))
            seen_cell_types.add(str(winner["cell_type"]))
            seen_tissues.add(str(winner["source_tissue"]))
            pool.remove(winner)
    if len(selected) != 12:
        raise ValueError("E3 sample selection is not unique")
    return selected


def raw_file_index(root: Path) -> dict[tuple[str, str, str], Path]:
    index: dict[tuple[str, str, str], Path] = {}
    for directory in root.iterdir():
        if not directory.is_dir() or not directory.name.startswith("EGAF"):
            continue
        for path in directory.glob("*.fastq.gz"):
            match = FASTQ_NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unexpected raw FASTQ name: {path}")
            key = match.groups()
            if key in index:
                raise ValueError(f"Ambiguous raw FASTQ key {key}: {path} and {index[key]}")
            index[key] = path.resolve()
    return index


def build_manifest(chosen: list[dict[str, str | float]], e3: set[str], raw_index: dict[tuple[str, str, str], Path]) -> list[dict[str, str | float]]:
    output = []
    for row in chosen:
        sample_id = str(row["sample_id"])
        record = {**row, "e2_candidate": "yes", "e3_candidate": "yes" if sample_id in e3 else "no"}
        for lane in ("001", "002"):
            for mate in ("1", "2"):
                path = raw_index.get((sample_id, lane, mate))
                if path is None or not path.is_file():
                    raise ValueError(f"Missing raw file: {sample_id} L{lane} R{mate}")
                sidecar = Path(f"{path}.md5")
                if not sidecar.is_file():
                    raise ValueError(f"Missing MD5 sidecar: {sidecar}")
                record[f"L{lane}_R{mate}_fastq"] = str(path)
                record[f"L{lane}_R{mate}_bytes"] = path.stat().st_size
        output.append(record)
    if len(output) != 39 or len({str(r["sample_id"]) for r in output}) != 39:
        raise ValueError("E2 selection did not yield 39 unique samples")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    chosen = freeze_samples(read_rows(args.metadata, ","), read_rows(args.atlas, "\t"))
    e3 = select_e3(chosen)
    manifest = build_manifest(chosen, e3, raw_file_index(args.raw))
    args.out.mkdir(parents=True)
    out_path = args.out / "E2_E3_sample_selection.tsv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": str(args.metadata.resolve()),
        "source_metadata_sha256": sha256(args.metadata),
        "frozen_atlas_manifest": str(args.atlas.resolve()),
        "frozen_atlas_manifest_sha256": sha256(args.atlas),
        "raw_fastq_root": str(args.raw.resolve()),
        "sample_selection": "highest published CpG coverage per Group, ties by sample_id ascending",
        "e3_selection": "4 per Germ layer; greedy new Cell type plus new Source Tissue, then new Source Tissue, coverage descending, sample_id ascending",
        "n_e2_candidates": len(manifest),
        "n_e3_candidates": len(e3),
        "n_unique_donors": len({str(row["donor_id"]) for row in manifest}),
        "coverage_screen_min": min(float(row["published_cpg_fragment_coverage"]) for row in manifest),
        "coverage_screen_max": max(float(row["published_cpg_fragment_coverage"]) for row in manifest),
        "n_coverage_screen_ge25": sum(float(row["published_cpg_fragment_coverage"]) >= 25 for row in manifest),
        "fastq_paths_checked": 4 * len(manifest),
        "md5_sidecars_present": 4 * len(manifest),
        "fastq_md5_or_gzip_integrity_checked": False,
        "read_length_audited": False,
        "epic_native_depth_measured": False,
        "selection_manifest_sha256": sha256(out_path),
    }
    (args.out / "selection_provenance.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
