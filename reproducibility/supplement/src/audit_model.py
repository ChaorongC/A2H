"""Verify deployed A2H coefficients against the original atlas fit artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash


MODEL = Path("legacy_model")
ATLAS = Path("atlas")


def compare_coefficients(deployed: pd.DataFrame, atlas: pd.DataFrame, column: str) -> dict[str, int | bool]:
    if deployed.featureID.duplicated().any() or atlas.index.has_duplicates:
        raise ValueError("Duplicate locus in coefficient artifact")
    indexed = deployed.set_index("featureID")
    if set(indexed.index) != set(atlas.index):
        raise ValueError("Deployed and atlas coefficient loci differ")
    values = np.stack(indexed.loc[atlas.index, column].to_numpy())
    if values.shape != atlas.shape or values.shape[1] != 3:
        raise ValueError("Expected one quadratic coefficient triplet per locus")
    mismatch = int(np.count_nonzero(~np.all(values == atlas.to_numpy(), axis=1)))
    return {"loci": len(atlas), "coefficient_triplet_mismatches": mismatch, "exact_match": mismatch == 0}


def audit(output: Path, model_root: Path, atlas_root: Path) -> None:
    paths = {
        "deployed": model_root / "A2H_coef.pkl",
        "cpg_info": model_root / "A2H_cpg_info.pkl",
        "probe_map": model_root / "A2H_cpg_probe.hg38.tsv",
        "atlas_CHALM": atlas_root / "result/20241024_CHALM_array_atlas_coef_degree2.pkl",
        "atlas_CAMDA": atlas_root / "result/20260514_CAMDA_array_atlas_coef_degree2.pkl",
    }
    deployed = pd.read_pickle(paths["deployed"])
    if list(deployed.columns) != ["featureID", "chalm_coef", "camda_coef"]:
        raise ValueError(f"Unexpected deployed coefficient schema: {list(deployed.columns)}")
    comparisons = {
        target: compare_coefficients(deployed, pd.read_pickle(paths[f"atlas_{target}"]), column)
        for target, column in (("CHALM", "chalm_coef"), ("CAMDA", "camda_coef"))
    }
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "model_identity": comparisons,
        "training_depth": {
            "field": "CT_count",
            "minimum": 4,
            "evidence": "Historical atlas processing used CHALM.py trad -m 4 for the methRatio and CHALM calls.",
            "deployed_reference_coefficients": "The deployed coefficients exactly match the historical atlas degree-2 fit artifacts.",
            "external_primary_truth_minimum": 10,
            "external_sensitivity_truth_minimum": 4,
            "interpretation": "The external WGBS truth threshold is separate from the depth used to generate the deployed reference coefficients.",
        },
        "file_sha256": {name: file_hash(path) for name, path in paths.items()},
    }
    (output / "model_provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    if not all(item["exact_match"] for item in comparisons.values()):
        raise ValueError("Deployed coefficients differ from the atlas artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=MODEL)
    parser.add_argument("--atlas-root", type=Path, default=ATLAS)
    args = parser.parse_args()
    audit(args.output_dir, args.model_root, args.atlas_root)


if __name__ == "__main__":
    main()
