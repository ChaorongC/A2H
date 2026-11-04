import numpy as np
import pandas as pd

from src.r2_2_e2e3_external import mean_site_rmse, paired_bootstrap, predict


def test_predict_masks_unfitted_sites():
    quality = pd.DataFrame({"fit_status": ["quadratic", "insufficient_train"],
                            "a2h_intercept": [0.1, 0.0], "a2h_linear": [1.0, 0.0], "a2h_quadratic": [0.5, 0.0]})
    pred = predict(quality, np.array([[0.2, 0.4], [0.2, 0.4]]))
    assert np.allclose(pred[0], [0.1 + 0.2 + 0.02, 0.1 + 0.4 + 0.08])
    assert np.isnan(pred[1]).all()


def test_mean_site_rmse_is_equal_weight_over_sites():
    obs = np.array([[0.0, 0.0], [0.0, np.nan]])
    pred = np.array([[0.3, 0.4], [0.1, 0.9]])
    # site 1: sqrt((0.09+0.16)/2); site 2: 0.1 (one valid pair)
    assert np.isclose(mean_site_rmse(obs, pred), (np.sqrt(0.125) + 0.1) / 2)


def test_paired_bootstrap_identical_predictions_give_zero():
    rng = np.random.default_rng(0)
    obs = rng.random((50, 10))
    pred = obs + rng.normal(0, 0.1, obs.shape)
    result = paired_bootstrap(obs, pred, pred.copy(), n_boot=50, seed=1)
    assert result["difference_a_minus_b"] == 0 and result["ci95_low"] == 0 and result["ci95_high"] == 0


def test_paired_bootstrap_uses_common_pairs_only():
    obs = np.array([[0.0, 0.0, 0.0]])
    a = np.array([[0.1, 0.1, 5.0]])
    b = np.array([[0.2, 0.2, np.nan]])
    result = paired_bootstrap(obs, a, b, n_boot=10, seed=1)
    assert result["n_pairs"] == 2
    assert np.isclose(result["difference_a_minus_b"], 0.1 - 0.2)


def test_pgpuk_c_arms_are_deployed_vs_capped_and_vs_locked_truth(tmp_path):
    import argparse
    from src.r2_2_e2e3_external import pgpuk_c

    rng = np.random.default_rng(3)
    loci = [f"chr1_{i}" for i in range(40)]
    arrays, wgbs = ["A1", "A2", "A3"], ["W1", "W2", "W3"]
    observed = rng.random((40, 3))
    deployed = observed + rng.normal(0.05, 0.1, observed.shape)
    capped = observed - 0.03 + rng.normal(0, 0.02, observed.shape)
    for target in ("CHALM", "CAMDA"):
        table = pd.DataFrame({"probe_id": [f"cg{i}" for i in range(40)], "locus": loci})
        for j, a in enumerate(arrays):
            table[f"m__{a}"] = 0.5
            table[f"observed__{a}"] = observed[:, j] if target == "CHALM" else observed[:, j] - 0.5
            table[f"baseline__{a}"] = deployed[:, j] if target == "CHALM" else deployed[:, j] - 0.5
        table.to_parquet(tmp_path / f"external_{target}.parquet", index=False)
    for j, w in enumerate(wgbs):
        for condition, chalm in (("native", observed[:, j]), ("raw125", capped[:, j])):
            pd.DataFrame({"locus": loci, "m": 0.5, "chalm": chalm, "ct": 12.0}).to_parquet(
                tmp_path / f"pgpuk_c__{w}__{condition}.parquet", index=False)
    pd.DataFrame({"study": "pgpuk", "tissue": "blood", "array_id": arrays, "wgbs_id": wgbs}).to_csv(
        tmp_path / "pairs.tsv", sep="\t", index=False)
    args = argparse.Namespace(epic_dir=tmp_path, external_dir=tmp_path, pairs=tmp_path / "pairs.tsv",
                              conditions=["raw125"], min_depth=10, n_boot=20, seed=1, output=tmp_path / "c.tsv")
    pgpuk_c(args)
    row = pd.read_csv(tmp_path / "c.tsv", sep="\t").query("target == 'CHALM'").iloc[0]
    assert np.isclose(row["mean_site_rmse_a"], mean_site_rmse(capped, deployed))
    assert np.isclose(row["mean_site_rmse_b"], mean_site_rmse(observed, deployed))
    assert np.isclose(row["mean_truth_shift_capped_minus_native"], (capped - observed).mean())
