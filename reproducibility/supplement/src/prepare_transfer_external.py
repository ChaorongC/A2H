"""Partition fixed PGP-UK observations by original-atlas CpG shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluate_pgpuk import file_hash


def prepare(shard_dir: Path, observation_dir: Path, probe_map: Path, output: Path) -> None:
    shards_path = shard_dir / "shards.tsv"
    shards = pd.read_csv(shards_path, sep="\t")
    if shards.shard_index.duplicated().any():
        raise ValueError("Duplicate shard index")
    locus_shard = []
    for row in shards.itertuples(index=False):
        loci = pd.read_parquet(row.path, columns=["locus"]).locus
        if len(loci) != row.n_loci or loci.iloc[0] != row.first_locus or loci.iloc[-1] != row.last_locus:
            raise ValueError(f"Shard metadata mismatch: {row.path}")
        locus_shard.append(pd.DataFrame({"locus": loci, "shard_index": row.shard_index}))
    locus_shard = pd.concat(locus_shard, ignore_index=True)
    if locus_shard.locus.duplicated().any():
        raise ValueError("Duplicate atlas locus")
    mapping = pd.read_csv(probe_map, sep="\t", usecols=["loci.hg38", "HM450.hg38"])
    mapping = mapping.dropna().drop_duplicates().rename(
        columns={"loci.hg38": "locus", "HM450.hg38": "probe_id"}
    )
    if mapping.probe_id.duplicated().any():
        raise ValueError("One external probe maps to multiple loci")
    mapping = mapping.merge(locus_shard, on="locus", how="inner", validate="many_to_one")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    source_hashes = {"shards": file_hash(shards_path), "probe_map": file_hash(probe_map)}
    for target in ("CHALM", "CAMDA"):
        path = observation_dir / f"{target}_observations.parquet"
        source_hashes[target] = file_hash(path)
        obs = pd.read_parquet(path)
        if obs.duplicated(["probe_id", "sample_id"]).any():
            raise ValueError(f"Duplicate {target} probe-subject observation")
        joined = obs.merge(mapping, on="probe_id", how="left", validate="many_to_one")
        n_unmapped = int(joined.shard_index.isna().sum())
        for shard_index, part in joined.dropna(subset=["shard_index"]).groupby("shard_index", sort=True):
            shard_index = int(shard_index)
            destination = (output / f"{target}_shard{shard_index:04d}.parquet").resolve()
            part.drop(columns="shard_index").to_parquet(destination, index=False, compression="zstd")
            rows.append({"target": target, "shard_index": shard_index, "path": str(destination),
                         "n_observations": len(part), "n_probes": part.probe_id.nunique(),
                         "sha256": file_hash(destination)})
        source_hashes[f"{target}_n_unmapped_observations"] = n_unmapped
        source_hashes[f"{target}_n_input_observations"] = len(obs)
    pd.DataFrame(rows).to_csv(output / "external_shards.tsv", sep="\t", index=False)
    (output / "provenance.json").write_text(json.dumps({
        "shard_dir": str(shard_dir), "observation_dir": str(observation_dir),
        "probe_map": str(probe_map), "source_sha256": source_hashes,
        "comparison_population": "fixed PGP-UK depth10 calibrated eligible observations",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--observation-dir", type=Path, required=True)
    parser.add_argument("--probe-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.shard_dir, args.observation_dir, args.probe_map, args.output_dir)


if __name__ == "__main__":
    main()
