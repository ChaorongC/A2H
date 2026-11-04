"""Add beta mean/variance summaries to existing complete-ten group definitions."""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(output):
    table_path = output / "group_beta_summary.tsv"
    record_path = output / "group_beta_summary_provenance.json"
    if table_path.exists() or record_path.exists():
        raise FileExistsError("Refusing to overwrite group beta summary outputs")
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    inputs = {
        "groups": output / "groups.tsv",
        "quality": output / "group_quality.tsv",
        "parent_provenance": provenance_path,
        "observations": Path(provenance["inputs"]["CHALM"]["path"]),
    }
    hashes = {name: digest(path) for name, path in inputs.items()}
    for filename, expected in provenance["output_sha256"].items():
        assert digest(output / filename) == expected, filename
    assert hashes["observations"] == provenance["inputs"]["CHALM"]["sha256"]
    groups = pd.read_csv(inputs["groups"], sep="\t", float_precision="round_trip")
    groups = groups.sort_values(["group_id", "member_rank"])
    quality = pd.read_csv(inputs["quality"], sep="\t", float_precision="round_trip")
    samples = provenance["samples_in_manifest_order"]
    assert len(samples) == len(set(samples)) == 10
    assert not groups.probe_id.duplicated().any()
    assert groups.groupby("group_id").size().eq(10).all()
    frame = pd.read_parquet(inputs["observations"], columns=["probe_id", "sample_id", "m"])
    frame = frame.loc[frame.probe_id.isin(groups.probe_id)]
    assert not frame.duplicated(["probe_id", "sample_id"]).any()
    beta = frame.pivot(index="probe_id", columns="sample_id", values="m")
    beta = beta.reindex(index=groups.probe_id, columns=samples).to_numpy(dtype=np.float64)
    assert beta.shape == (len(groups), 10) and np.isfinite(beta).all()
    np.testing.assert_allclose(beta.mean(axis=1), groups.mean_m, atol=1e-15, rtol=0)
    np.testing.assert_allclose(beta.var(axis=1, ddof=0), groups.var_m, atol=1e-15, rtol=0)
    values = beta.reshape(-1, 10, 10)
    means = values.mean(axis=2)
    variances = values.var(axis=2, ddof=0)
    pooled = values.reshape(-1, 100)
    statistics = pd.DataFrame({
        "group_id": groups.group_id.to_numpy()[::10],
        "n_observations": 100,
        "group_mean_beta": pooled.mean(axis=1),
        "group_variance_beta": pooled.var(axis=1, ddof=0),
        "mean_cpg_variance_beta": variances.mean(axis=1),
        "variance_cpg_mean_beta": means.var(axis=1, ddof=0),
    })
    np.testing.assert_allclose(statistics.group_mean_beta, means.mean(axis=1), atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        statistics.group_variance_beta,
        statistics.mean_cpg_variance_beta + statistics.variance_cpg_mean_beta,
        atol=1e-15, rtol=0,
    )
    assert set(statistics.group_id) == set(quality.group_id)
    result = quality.merge(statistics, on="group_id", validate="one_to_one").sort_values("group_id")
    for name, path in inputs.items():
        assert digest(path) == hashes[name], name
    result.to_csv(table_path, sep="\t", index=False)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {name: {"path": str(path.resolve()), "sha256": hashes[name]} for name, path in inputs.items()},
        "code": {"path": str(Path(__file__).resolve()), "sha256": digest(__file__)},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "group_count": len(result), "variance_ddof": 0,
        "definitions": {
            "group_mean_beta": "Mean of 100 beta values (10 CpGs x 10 participants); also mean of the ten CpG means.",
            "group_variance_beta": "Population variance of the 100 beta values around group_mean_beta.",
            "mean_cpg_variance_beta": "Average of ten per-CpG population variances across participants; describes the grouping variance statistic.",
            "variance_cpg_mean_beta": "Population variance of ten CpG mean beta values; between-CpG contribution to pooled variance.",
            "other_columns": "Existing group_quality.tsv columns retained without changing group membership.",
        },
        "validation": {
            "all_parent_output_hashes_verified": True,
            "source_beta_statistics_reproduced": True,
            "pooled_variance_equals_mean_within_cpg_variance_plus_variance_of_cpg_means": True,
            "all_groups_complete_10_by_10": True,
            "inputs_unchanged": True,
            "no_new_filter_or_regrouping": True,
        },
        "output": {"path": str(table_path.resolve()), "sha256": digest(table_path)},
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {len(result)} group summaries to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir.resolve())
