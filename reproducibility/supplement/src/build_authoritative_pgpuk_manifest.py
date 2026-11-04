"""Audit PGP-UK array identifiers against ENA sample metadata.

The existing validation branch was assembled from ordinal PGP-UK labels. This
utility joins the ArrayExpress SDRF identifiers to ENA run metadata and emits
all runs whose ENA BioSample accession and individual identifier both match
the array record. It deliberately preserves the library strategy: an exact
WGS match authenticates an individual but is not a WGBS input. It does not
alter the existing validation namespace.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_array_samples(path: Path) -> list[dict[str, str]]:
    """Return one row per unique PGP-UK array sample from the SDRF."""
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row.get("Source Name", "")
            if not name.startswith("Sample ") or "_PGPUK_B" not in name:
                continue
            # The SDRF also contains a later-added B13 array sample without
            # a BioSample accession. It cannot enter an accession-based join.
            biosample = row.get("Comment[BioSD_SAMPLE]", "").strip()
            if not biosample:
                continue
            rows.setdefault(
                name,
                {
                    "array_sdrf_name": name,
                    "array_id": name.split("_", 1)[1],
                    "individual": row["Characteristics[individual]"],
                    "biosample": biosample,
                    "tissue": row["Characteristics[organism part]"],
                },
            )
    return [rows[key] for key in sorted(rows, key=lambda value: int(value.split()[1].split("_")[0]))]


def read_ena_runs(path: Path) -> dict[str, list[dict[str, str]]]:
    """Index all ENA runs by BioSample and individual alias."""
    result: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            run = row.get("run_accession", "")
            if run:
                for key in (run, row.get("sample_accession", ""), row.get("sample_alias", "")):
                    if key:
                        result.setdefault(key, []).append(row)
    return result


def build_linkage(array_rows: list[dict[str, str]], ena_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for array in array_rows:
        candidates = ena_rows.get(array["biosample"], []) + ena_rows.get(array["individual"], [])
        unique_candidates = {row["run_accession"]: row for row in candidates}
        exact = [
            row for row in unique_candidates.values()
            if row.get("sample_accession") == array["biosample"]
            and row.get("sample_alias") == array["individual"]
        ]
        if not exact:
            raise ValueError(f"No ENA run matched array sample {array['array_id']}")
        for ena in sorted(exact, key=lambda row: row["run_accession"]):
            output.append({
                **array,
                "ena_run": ena["run_accession"],
                "ena_biosample": ena["sample_accession"],
                "ena_individual": ena["sample_alias"],
                "secondary_sample_accession": ena.get("secondary_sample_accession", ""),
                "library_strategy": ena.get("library_strategy", ""),
            })
    return output


def write_linkage(linkage: list[dict[str, str]], output: Path) -> None:
    fields = ["array_sdrf_name", "array_id", "individual", "biosample", "tissue", "ena_run", "ena_biosample", "ena_individual", "secondary_sample_accession", "library_strategy"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in fields} for item in linkage)


def write_wgbs_candidates(linkage: list[dict[str, str]], output: Path) -> None:
    """Write only exact bisulfite-sequencing matches for downstream review."""
    fields = ["array_sdrf_name", "array_id", "individual", "biosample", "tissue", "ena_run", "ena_biosample", "ena_individual", "secondary_sample_accession", "library_strategy"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: item.get(field, "") for field in fields}
            for item in linkage
            if item.get("library_strategy") == "Bisulfite-Seq"
        )


def compare_current_runs(
    array_rows: list[dict[str, str]],
    ena_rows: dict[str, list[dict[str, str]]],
    current_manifest: Path,
) -> list[dict[str, str]]:
    """Compare ordinal B1-B10 pairs to ENA's actual WGBS sample metadata."""
    arrays = {f"PGPUK{row['array_id'].split('_B')[1]}": row for row in array_rows}
    if len(arrays) != len(array_rows):
        raise ValueError("Duplicate PGP-UK array ID")
    output = []
    with current_manifest.open(newline="") as handle:
        for current in csv.DictReader(handle, delimiter="\t"):
            if current.get("study") != "pgpuk":
                continue
            wgbs_id = current["wgbs_id"]
            if wgbs_id not in arrays:
                raise ValueError(f"Unexpected PGP-UK WGBS ID: {wgbs_id}")
            run = current["run_accession"]
            matches = ena_rows.get(run, [])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one ENA record for {run}")
            array = arrays[wgbs_id]
            ena = matches[0]
            output.append({
                "wgbs_id": wgbs_id,
                "array_id": array["array_id"],
                "array_biosample": array["biosample"],
                "array_individual": array["individual"],
                "wgbs_run": run,
                "wgbs_biosample": ena.get("sample_accession", ""),
                "wgbs_alias": ena.get("sample_alias", ""),
                "wgbs_library_strategy": ena.get("library_strategy", ""),
                "same_biosample": str(ena.get("sample_accession") == array["biosample"]).lower(),
                "same_individual_alias": str(ena.get("sample_alias") == array["individual"]).lower(),
            })
    if len(output) != len(arrays) or len({row["wgbs_id"] for row in output}) != len(arrays):
        raise ValueError("Current PGP-UK manifest does not contain one run per array sample")
    return output


def write_comparison(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-sdrf", type=Path, required=True)
    parser.add_argument("--ena-runs", type=Path, required=True)
    parser.add_argument("--linkage-output", type=Path, required=True)
    parser.add_argument("--wgbs-candidates-output", type=Path, required=True)
    parser.add_argument("--current-wgbs-manifest", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.current_wgbs_manifest is None) != (args.comparison_output is None):
        raise ValueError("--current-wgbs-manifest and --comparison-output must be supplied together")
    array_rows = read_array_samples(args.array_sdrf)
    ena_rows = read_ena_runs(args.ena_runs)
    linkage = build_linkage(array_rows, ena_rows)
    if args.current_wgbs_manifest:
        comparison = compare_current_runs(array_rows, ena_rows, args.current_wgbs_manifest)
        write_comparison(comparison, args.comparison_output)
        print(f"wrote {args.comparison_output}")
    write_linkage(linkage, args.linkage_output)
    write_wgbs_candidates(linkage, args.wgbs_candidates_output)
    print(f"wrote {args.linkage_output}")
    print(f"wrote {args.wgbs_candidates_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
