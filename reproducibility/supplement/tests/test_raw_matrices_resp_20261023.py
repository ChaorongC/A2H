import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_pgpuk import row_stats
from src.run_pgpuk_disjoint_complete10 import make_groups as published_make_groups

SCRIPT = Path(__file__).resolve().parents[1] / "collaborator_figure_data_20261017/raw_matrices_resp_20261023/make_groups.py"


def load_script():
    spec = importlib.util.spec_from_file_location("make_groups", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_grouping_matches_published_rule_with_ties_and_remainder():
    mg = load_script()
    rng = np.random.default_rng(1)
    n = 1037  # partial last block (37) -> 3 full groups + 7 remainder
    beta = rng.random((n, 10)).round(2)  # rounding creates mean and variance ties
    probes = np.array([f"cg{i:08d}" for i in rng.permutation(n)])
    mine, remainder = mg.make_groups(probes, beta)
    stats = pd.DataFrame({"probe_id": probes, "n_participants": 10, "mean_m": beta.mean(axis=1), "var_m": beta.var(axis=1, ddof=0)})
    published, published_remainder = published_make_groups(stats)
    pd.testing.assert_frame_equal(mine[published.columns], published)
    assert remainder == len(published_remainder) == 7


def test_shipped_group_metrics_match_published_row_stats():
    mg = load_script()
    rng = np.random.default_rng(2)
    y, p = rng.random((50, 100)), rng.random((50, 100))
    y[3] = 0.5  # constant truth: r undefined, RMSE kept
    r, rmse = mg.pearson_rmse(y, p)
    ref = row_stats(y, p)
    np.testing.assert_allclose(rmse, ref["rmse"], rtol=0, atol=1e-14)
    np.testing.assert_allclose(r, ref["pearson_r"], rtol=0, atol=1e-14, equal_nan=True)
    assert np.isnan(r[3]) and np.isfinite(rmse[3])
