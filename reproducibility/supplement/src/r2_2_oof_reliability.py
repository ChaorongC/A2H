"""R2-2 E4: EPIC site reliability from completed sample-resolved atlas OOF CV.

This reads the frozen 100-split reviewer CV branch. That branch follows the
historical sample splits but used a vectorized fitter, not literal scalar
numpy.polyfit at every site; it must be labeled as a numerical reproduction.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import file_hash
from src.r2_2_common import pooled_metrics, site_metrics
from src.r2_2_reference_size import DEFAULT_ROOT, RESULTS, SAMPLES, SHARDS, TARGETS


SPLITS = RESULTS / "atlas_splits/splits.tsv"
CV = RESULTS / "atlas_cv_full/original"


def run_shard(root: Path, shard_index: int) -> None:
    e4 = root / "E4"
    out = e4 / f"shard{shard_index:04d}"
    out.mkdir(parents=True, exist_ok=False)
    mapping = pd.read_csv(root / "E1/epic_mapping.tsv", sep="\t", usecols=["locus", "shard_index"])
    mapping = mapping.loc[mapping.shard_index == shard_index]
    if mapping.locus.duplicated().any() or mapping.empty:
        raise ValueError("E4 EPIC locus mapping must be unique and nonempty")
    samples = pd.read_csv(SAMPLES, sep="\t").sample_id.tolist()
    if len(samples) != 205 or len(set(samples)) != 205:
        raise ValueError("Invalid atlas sample manifest")
    sample_index = {sample: i for i, sample in enumerate(samples)}
    splits = pd.read_csv(SPLITS, sep="\t")
    splits = splits.loc[splits.branch == "historical_sample"]
    shard = pd.read_csv(SHARDS, sep="\t").set_index("shard_index").loc[shard_index]
    frame = pd.read_parquet(shard.path)
    if (len(frame) != shard.n_loci or frame.locus.iloc[0] != shard.first_locus or
            frame.locus.iloc[-1] != shard.last_locus):
        raise ValueError("Changed atlas shard boundaries")
    frame = frame.loc[frame.locus.isin(mapping.locus)].reset_index(drop=True)
    if len(frame) != len(mapping) or frame.locus.duplicated().any():
        raise ValueError("EPIC mapping and atlas data disagree")
    loci = frame.locus.to_numpy()
    x = frame[[f"m__{sample}" for sample in samples]].to_numpy(dtype=float)
    outputs = {}
    for target in TARGETS:
        truth = frame[[f"{target}__{sample}" for sample in samples]].to_numpy(dtype=float)
        joint = np.isfinite(x) & np.isfinite(truth)
        n_reference_joint = joint.sum(axis=1)
        x_variance = np.nanvar(np.where(joint, x, np.nan), axis=1)
        prediction_sum = np.zeros_like(truth)
        prediction_count = np.zeros_like(truth, dtype=np.uint8)
        per_repeat = []
        overall_repeat = []
        train_min = np.full(len(loci), 205)
        train_max = np.zeros(len(loci), dtype=int)
        train_sum = np.zeros(len(loci), dtype=int)
        n_training_folds = 0
        for repeat in range(10):
            observed = np.full_like(truth, np.nan)
            predicted = np.full_like(truth, np.nan)
            fit_status = np.full((len(loci), len(samples)), "not_assigned", dtype=object)
            filled = np.zeros(len(samples), dtype=bool)
            for fold in range(10):
                split_index = repeat * 10 + fold
                part = splits.loc[splits.split_index == split_index]
                train_ids = part.loc[part.role == "train", "sample_id"].tolist()
                test_ids = part.loc[part.role == "test", "sample_id"].tolist()
                if (len(train_ids) + len(test_ids) != 205 or set(train_ids) & set(test_ids) or
                        set(train_ids) | set(test_ids) != set(samples)):
                    raise ValueError(f"Invalid split {split_index}")
                positions = [sample_index[name] for name in test_ids]
                if filled[positions].any():
                    raise ValueError("Sample held out twice within a repeat")
                cv_dir = CV / f"historical_sample_split{split_index:03d}"
                metadata = json.loads((cv_dir / "completion.json").read_text())
                if (metadata["repeat"] != repeat or metadata["fold"] != fold or
                        metadata["test_ids"] != test_ids or metadata["train_ids"] != train_ids or
                        metadata["min_joint_train"] != 50):
                    raise ValueError(f"CV completion/split mismatch: {cv_dir}")
                path = cv_dir / f"{target}_shard{shard_index:04d}_predictions.parquet"
                record = next((item for item in metadata["outputs"] if item["target"] == target
                               and item["shard_index"] == shard_index), None)
                if record is None or str(path.resolve()) != record["prediction_path"]:
                    raise ValueError("CV output not declared in completion manifest")
                data = pd.read_parquet(path, columns=["locus", "sample_id", "target", "repeat", "fold",
                                                      "observed", "A2H", "fit_status"])
                data = data.loc[data.locus.isin(mapping.locus)]
                expected_loci = np.repeat(loci, len(test_ids))
                expected_samples = np.tile(test_ids, len(loci))
                if (len(data) != len(expected_loci) or
                        not np.array_equal(data.locus.to_numpy(), expected_loci) or
                        not np.array_equal(data.sample_id.to_numpy(), expected_samples) or
                        not data.target.eq(target).all() or not data.repeat.eq(repeat).all() or
                        not data.fold.eq(fold).all()):
                    raise ValueError(f"OOF sample/site key mismatch: {path}")
                observed[:, positions] = data.observed.to_numpy(dtype=float).reshape(len(loci), len(test_ids))
                predicted[:, positions] = data.A2H.to_numpy(dtype=float).reshape(len(loci), len(test_ids))
                fit_status[:, positions] = data.fit_status.to_numpy().reshape(len(loci), len(test_ids))
                filled[positions] = True
                n_train = n_reference_joint - joint[:, positions].sum(axis=1)
                train_min = np.minimum(train_min, n_train)
                train_max = np.maximum(train_max, n_train)
                train_sum += n_train
                n_training_folds += 1
            if not filled.all() or not np.allclose(observed, truth, equal_nan=True, atol=0, rtol=0):
                raise ValueError("OOF repeat incomplete or truth differs from frozen atlas matrix")
            metrics = site_metrics(observed, predicted)
            metrics.insert(0, "locus", loci)
            metrics.insert(1, "target", target)
            metrics.insert(2, "repeat", repeat)
            metrics["n_truth_measurements"] = np.isfinite(truth).sum(axis=1)
            metrics["n_rank_deficient_test_records"] = (fit_status == "rank_deficient").sum(axis=1)
            metrics["n_constant_kde_test_records"] = (fit_status == "constant_kde").sum(axis=1)
            metrics["n_insufficient_test_records"] = (fit_status == "insufficient").sum(axis=1)
            per_repeat.append(metrics)
            summary = pooled_metrics(metrics)
            centered = metrics.loc[(metrics.n >= 2) & (metrics.syy > 1e-24) & (metrics.spp > 1e-24)]
            overall_repeat.append({"target": target, "repeat": repeat, "shard_index": shard_index,
                                   "n_epic_sites": len(loci), **summary,
                                   "sse": float(metrics.sse.sum()),
                                   "centered_sxy": float(centered.sxy.sum()),
                                   "centered_syy": float(centered.syy.sum()),
                                   "centered_spp": float(centered.spp.sum())})
            valid = np.isfinite(predicted) & np.isfinite(truth)
            prediction_sum += np.where(valid, predicted, 0.0)
            prediction_count += valid.astype(np.uint8)
        averaged = np.divide(prediction_sum, np.maximum(prediction_count, 1),
                             out=np.full_like(prediction_sum, np.nan), where=prediction_count > 0)
        metrics = site_metrics(truth, averaged)
        metrics.insert(0, "locus", loci)
        metrics.insert(1, "target", target)
        metrics["n_truth_measurements"] = np.isfinite(truth).sum(axis=1)
        metrics["valid_prediction_fraction"] = metrics.n / np.maximum(metrics.n_truth_measurements, 1)
        metrics["n_reference_joint"] = n_reference_joint
        metrics["reference_m_variance"] = x_variance
        metrics["min_fold_train_joint"] = train_min
        metrics["max_fold_train_joint"] = train_max
        metrics["mean_fold_train_joint"] = train_sum / n_training_folds
        metrics["n_technical_oof_repeats"] = 10
        named = {
            "per_site_oof": metrics,
            "per_site_by_repeat": pd.concat(per_repeat, ignore_index=True),
            "overall_by_repeat_shard": pd.DataFrame(overall_repeat),
        }
        outputs[target] = {}
        for name, table in named.items():
            path = out / f"{target}_{name}.parquet"
            table.to_parquet(path, index=False, compression="zstd")
            outputs[target][name] = {"path": str(path.resolve()), "sha256": file_hash(path), "n_rows": len(table)}
    (out / "completion.json").write_text(json.dumps({
        "run_status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
        "shard_index": shard_index, "n_epic_loci": len(loci), "n_reference_samples": len(samples),
        "n_repeats": 10, "n_splits": 100, "training_depth": ">=4",
        "fit_branch": "existing vectorized sample-resolved reviewer CV; not the historical fold-mean pickle",
        "outputs": outputs, "source_shard": {"path": shard.path, "sha256": shard.sha256},
        "sample_manifest_sha256": file_hash(SAMPLES), "split_manifest_sha256": file_hash(SPLITS),
        "epic_mapping_sha256": file_hash(root / "E1/epic_mapping.tsv"),
        "code_sha256": file_hash(Path(__file__)), "command": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }, indent=2) + "\n")
    print(f"E4 shard {shard_index} verified: {len(loci)} EPIC loci, 205 samples, 10 repeats", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    run_shard(args.root, args.shard_index)


if __name__ == "__main__":
    main()
