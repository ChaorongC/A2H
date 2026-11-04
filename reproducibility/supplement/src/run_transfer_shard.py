"""Fit one frozen atlas transfer model on one CpG shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash, row_stats, spearman_rows
from src.fit_atlas_fold import fit_sites
from src.fit_reference_controls import fit_chunk
from src.run_atlas_cv_split import shard_arrays


def model_members(manifest: pd.DataFrame, model: str, sample_order: list[str]) -> tuple[list[str], list[str]]:
    part = manifest.loc[manifest.model == model]
    if len(part) != len(sample_order) or part.sample_id.duplicated().any() or set(part.sample_id) != set(sample_order):
        raise ValueError(f"Incomplete transfer membership: {model}")
    if not part.role.isin(["train", "heldout", "excluded"]).all():
        raise ValueError("Unexpected membership role")
    train = part.loc[part.role == "train", "sample_id"].tolist()
    heldout = part.loc[part.role == "heldout", "sample_id"].tolist()
    train_donors = set(part.loc[part.role == "train", "donor_id"])
    heldout_donors = set(part.loc[part.role == "heldout", "donor_id"])
    if train_donors & heldout_donors:
        raise ValueError(f"Donor overlap for {model}")
    return train, heldout


def external_arrays(obs: pd.DataFrame, loci: pd.Index, sample_order: list[str]
                    ) -> tuple[pd.Index, pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    if obs.duplicated(["probe_id", "sample_id"]).any() or obs.groupby("probe_id").locus.nunique().gt(1).any():
        raise ValueError("Ambiguous external probe observations")
    probes = pd.Index(obs.probe_id.drop_duplicates(), name="probe_id")
    probe_loci = obs.drop_duplicates("probe_id").set_index("probe_id").loc[probes, "locus"]
    positions = loci.get_indexer(probe_loci)
    if (positions < 0).any():
        raise ValueError("External probe locus absent from shard")
    def pivot(column: str) -> np.ndarray:
        return obs.pivot(index="probe_id", columns="sample_id", values=column).reindex(
            index=probes, columns=sample_order
        ).to_numpy(dtype=float)
    return probes, positions, pivot("m"), pivot("observed"), pivot("A2H")


def external_scores(obs: pd.DataFrame, loci: pd.Index, x: np.ndarray, y: np.ndarray,
                    sample_order: list[str], min_train: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    probes, positions, beta, truth, baseline = external_arrays(obs, loci, sample_order)
    quality, predictions = fit_sites(x[positions], y[positions], beta, min_train=min_train)
    rows = []
    for method, pred in {"historical_A2H": baseline, **predictions}.items():
        stats = row_stats(truth, pred)
        rows.append(pd.DataFrame({
            "probe_id": probes, "locus": loci[positions], "method": method,
            "n": stats["n"], "rmse": stats["rmse"],
            "pearson_r": stats["pearson_r"], "spearman_rho": spearman_rows(truth, pred),
            "fit_status": quality.fit_status.to_numpy(),
        }))
    site_metrics = pd.concat(rows, ignore_index=True)
    valid = np.isfinite(truth) & np.isfinite(baseline) & np.isfinite(predictions["A2H"])
    long = pd.DataFrame({
        "probe_id": np.repeat(probes.to_numpy(), len(sample_order)),
        "sample_id": np.tile(sample_order, len(probes)),
        "observed": truth.ravel(), "historical_A2H": baseline.ravel(),
        "transfer_A2H": predictions["A2H"].ravel(),
        "transfer_reference_linear": predictions["reference_linear"].ravel(),
        "m": beta.ravel(),
    }).loc[valid.ravel()].reset_index(drop=True)
    return site_metrics, long


def heldout_scores(loci: pd.Index, x: np.ndarray, y: np.ndarray,
                   x_test: np.ndarray, truth: np.ndarray, min_train: int) -> pd.DataFrame:
    quality, predictions = fit_sites(x, y, x_test, min_train=min_train)
    rows = []
    for method, pred in predictions.items():
        stats = row_stats(truth, pred)
        rows.append(pd.DataFrame({
            "locus": loci, "method": method, "n": stats["n"],
            "rmse": stats["rmse"], "pearson_r": stats["pearson_r"],
            "spearman_rho": spearman_rows(truth, pred),
            "fit_status": quality.fit_status.to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True)


def run(shard_dir: Path, external_dir: Path, membership_path: Path, sample_path: Path,
        output: Path, model: str, shard_index: int, min_train: int) -> None:
    if min_train < 3:
        raise ValueError("Minimum joint training count must be >=3")
    shards_path = shard_dir / "shards.tsv"
    external_index_path = external_dir / "external_shards.tsv"
    shards = pd.read_csv(shards_path, sep="\t").set_index("shard_index")
    external_index = pd.read_csv(external_index_path, sep="\t")
    sample_order = pd.read_csv(sample_path, sep="\t").sample_id.tolist()
    train, heldout = model_members(pd.read_csv(membership_path, sep="\t"), model, sample_order)
    if len(train) < min_train:
        raise ValueError(f"{model} has only {len(train)} training samples (<{min_train})")
    row = shards.loc[shard_index]
    frame = pd.read_parquet(row.path)
    if len(frame) != row.n_loci or frame.locus.iloc[0] != row.first_locus or frame.locus.iloc[-1] != row.last_locus:
        raise ValueError("Atlas shard metadata mismatch")
    loci = pd.Index(frame.locus)
    if loci.has_duplicates:
        raise ValueError("Duplicate atlas locus")
    output.mkdir(parents=True, exist_ok=True)
    result = {"model": model, "shard_index": shard_index, "n_loci": len(loci),
              "min_train": min_train, "train_ids": train, "heldout_ids": heldout,
              "source_sha256": {"atlas_shards": file_hash(shards_path),
                                "external_shards": file_hash(external_index_path),
                                "membership": file_hash(membership_path),
                                "samples": file_hash(sample_path), "atlas_shard": file_hash(Path(row.path))},
              "targets": {}}
    for target in ("CHALM", "CAMDA"):
        x, y, x_test, truth = shard_arrays(frame, target, train, heldout)
        quality, gram, rhs = fit_chunk(x, y, loci, min_train)
        detail = {"n_joint_eligible": int(quality.reference_mean.notna().sum()),
                  "universal_gram": gram.tolist(), "universal_rhs": rhs.tolist(),
                  "n_external_input": 0, "n_external_fitted": 0}
        if heldout:
            heldout_table = heldout_scores(loci, x, y, x_test, truth, min_train)
            path = (output / f"{target}_heldout_site_metrics.parquet").resolve()
            heldout_table.to_parquet(path, index=False, compression="zstd")
            detail["heldout_site_metrics"] = str(path)
            detail["heldout_sha256"] = file_hash(path)
        external_row = external_index.loc[(external_index.target == target) &
                                          (external_index.shard_index == shard_index)]
        if len(external_row) > 1:
            raise ValueError("Duplicate external shard")
        if len(external_row):
            external_path = Path(external_row.path.iloc[0])
            if file_hash(external_path) != external_row.sha256.iloc[0]:
                raise ValueError("External shard hash mismatch")
            obs = pd.read_parquet(external_path)
            sample_ids = sorted(obs.sample_id.unique().tolist())
            sites, predictions = external_scores(obs, loci, x, y, sample_ids, min_train)
            sites.insert(2, "target", target)
            detail["n_external_input"] = int(obs.probe_id.nunique())
            detail["n_external_fitted"] = int(sites.loc[
                (sites.method == "A2H") & (sites.n >= 8), "probe_id"].nunique())
            for name, table in (("external_site_metrics", sites), ("external_predictions", predictions)):
                path = (output / f"{target}_{name}.parquet").resolve()
                table.to_parquet(path, index=False, compression="zstd")
                detail[name] = str(path)
                detail[f"{name}_sha256"] = file_hash(path)
            result["source_sha256"][f"external_{target}"] = file_hash(external_path)
        result["targets"][target] = detail
    (output / "completion.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--min-train", type=int, default=50)
    args = parser.parse_args()
    run(args.shard_dir, args.external_dir, args.membership, args.samples,
        args.output_dir, args.model, args.shard_index, args.min_train)


if __name__ == "__main__":
    main()
