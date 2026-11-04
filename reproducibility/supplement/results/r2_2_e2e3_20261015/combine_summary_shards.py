"""Combine per-shard outputs of src/r2_2_e2e3_summarize.py (see build_summary_shards.sh).

Concatenates each shard's paired_epic_deltas.tsv / realized_epic_depth.tsv into --output-dir,
links every shard parquet into <namespace>/epic (the --epic-dir for r2_2_e2e3_external), and
checks condition and native counts.  Run from R/:
  python results/r2_2_e2e3_20261015/combine_summary_shards.py --namespace <ROOT>/<ns> \
      --output-dir results/r2_2_e2e3_20261015/<summary branch> [--expect e2=351 e3=117 pgpuk_c=30]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect", nargs="*", default=[], help="experiment=n_conditions")
    args = parser.parse_args()

    shards = pd.read_csv(args.namespace / "shards.tsv", sep="\t")
    deltas, depth = [], []
    for root in map(Path, shards["root"]):
        deltas.append(pd.read_csv(root / "out" / "paired_epic_deltas.tsv", sep="\t"))
        depth.append(pd.read_csv(root / "out" / "realized_epic_depth.tsv", sep="\t"))
    deltas, depth = pd.concat(deltas, ignore_index=True), pd.concat(depth, ignore_index=True)

    keys = ["experiment", "sample_id", "condition"]
    if depth.duplicated(keys).any() or deltas.duplicated(keys + ["target"]).any():
        raise ValueError("Duplicate condition rows across shards")
    per_condition = deltas.groupby(keys)["target"].nunique()
    if not (per_condition == 3).all():
        raise ValueError("A condition lacks one of the m/chalm/camda rows")
    counts = depth.groupby("experiment").size().to_dict()
    for item in args.expect:
        experiment, n = item.split("=")
        if counts.get(experiment, 0) != int(n):
            raise ValueError(f"{experiment}: {counts.get(experiment, 0)} conditions, expected {n}")

    epic = args.namespace / "epic"
    epic.mkdir(exist_ok=False)
    linked = {}
    for root in map(Path, shards["root"]):
        for parquet in sorted((root / "epic").glob("*.parquet")):
            if parquet.name in linked:
                raise ValueError(f"{parquet.name} produced by two shards")
            (epic / parquet.name).symlink_to(parquet)
            linked[parquet.name] = str(parquet)
    natives = {e: sum(n.startswith(f"{e}__") and n.endswith("__native.parquet") for n in linked)
               for e in ("e2", "e3", "pgpuk_c")}
    expected_parquets = len(depth) + sum(natives.values())
    if len(linked) != expected_parquets:
        raise ValueError(f"{len(linked)} parquets linked, expected {expected_parquets}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    deltas.to_csv(args.output_dir / "paired_epic_deltas.tsv", sep="\t", index=False)
    depth.to_csv(args.output_dir / "realized_epic_depth.tsv", sep="\t", index=False)
    here = Path(__file__).resolve()
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "namespace": str(args.namespace), "shards": len(shards), "epic_dir": str(epic),
        "conditions_per_experiment": counts, "natives_per_experiment": natives,
        "parquets_linked": len(linked),
        "code_sha256": {"src/r2_2_e2e3_summarize.py": sha256(here.parents[2] / "src" / "r2_2_e2e3_summarize.py"),
                        here.name: sha256(here),
                        "build_summary_shards.sh": sha256(here.parent / "build_summary_shards.sh")},
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
