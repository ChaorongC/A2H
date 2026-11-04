"""Materialize aligned CpG shards once for low-memory atlas CV split jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.audit_model import ATLAS
from src.evaluate_pgpuk import file_hash


MATRIX_NAMES = {
    "original": {
        "m": "20241023_merged_methratio_all_sample_beadchip_cpg.pkl",
        "CHALM": "20241023_merged_CHALM_all_sample_beadchip_cpg.pkl",
        "CAMDA": "20260514_merged_CAMDA_all_sample_beadchip_cpg.pkl",
    },
    "depth10": {
        "m": "20260605_merged_methratio_all_sample_beadchip_cpg_depth10.pkl",
        "CHALM": "20260605_merged_CHALM_all_sample_beadchip_cpg_depth10.pkl",
        "CAMDA": "20260605_merged_CAMDA_all_sample_beadchip_cpg_depth10.pkl",
    },
}


def prepare(atlas_root: Path, sample_manifest: Path, output: Path,
            branch: str, shard_size: int, limit_loci: int | None = None) -> None:
    if branch not in MATRIX_NAMES or shard_size < 1 or (limit_loci is not None and limit_loci < 1):
        raise ValueError("Invalid matrix branch, shard size, or locus limit")
    samples = pd.read_csv(sample_manifest, sep="\t").sample_id.tolist()
    paths = {name: atlas_root / "data" / filename for name, filename in MATRIX_NAMES[branch].items()}
    frames = {name: pd.read_pickle(path) for name, path in paths.items()}
    index = frames["m"].index
    if index.has_duplicates or len(samples) != 205 or len(set(samples)) != 205:
        raise ValueError("Expected unique loci and 205 atlas samples")
    for name, frame in frames.items():
        if not frame.index.equals(index) or frame.columns.tolist() != samples:
            raise ValueError(f"{name} matrix does not match loci and sample order")
    n_loci = len(index) if limit_loci is None else min(len(index), limit_loci)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard_index, start in enumerate(range(0, n_loci, shard_size)):
        stop = min(start + shard_size, n_loci)
        columns = {"locus": index[start:stop].to_numpy()}
        for name, frame in frames.items():
            values = frame.iloc[start:stop]
            for sample in samples:
                columns[f"{name}__{sample}"] = values[sample].to_numpy(dtype=float)
        block = pd.DataFrame(columns)
        path = (output / f"shard_{shard_index:04d}.parquet").resolve()
        block.to_parquet(path, index=False, compression="zstd")
        rows.append({"shard_index": shard_index, "path": str(path),
                     "start": start, "stop": stop, "n_loci": stop - start,
                     "first_locus": index[start], "last_locus": index[stop - 1],
                     "sha256": file_hash(path)})
    pd.DataFrame(rows).to_csv(output / "shards.tsv", sep="\t", index=False)
    provenance = {
        "branch": branch, "source_root": str(atlas_root), "n_loci": n_loci,
        "complete_matrix_loci": len(index), "n_samples": len(samples),
        "shard_size": shard_size, "limit_loci": limit_loci,
        "sample_manifest_sha256": file_hash(sample_manifest),
        "source_sha256": {name: file_hash(path) for name, path in paths.items()},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-root", type=Path, default=ATLAS)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--branch", choices=MATRIX_NAMES, required=True)
    parser.add_argument("--shard-size", type=int, default=20000)
    parser.add_argument("--limit-loci", type=int)
    args = parser.parse_args()
    prepare(args.atlas_root, args.sample_manifest, args.output_dir,
            args.branch, args.shard_size, args.limit_loci)


if __name__ == "__main__":
    main()
