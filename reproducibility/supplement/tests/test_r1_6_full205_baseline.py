import numpy as np

from src.evaluate_pgpuk import row_stats
from src.r1_6_full205_baseline import boot_rmse_sum, donor_weights


def test_donor_weights_resample_whole_donors():
    donors = np.array(["d1", "d1", "d2", "d3", "d3", "d3"])
    w = donor_weights(donors, n_boot=50, seed=[2026, 0])
    assert w.shape == (6, 50)
    assert (w[0] == w[1]).all() and (w[3] == w[4]).all() and (w[4] == w[5]).all()
    assert (w[[0, 2, 3]].sum(axis=0) == 3).all()      # three donors drawn per replicate
    assert np.array_equal(w, donor_weights(donors, n_boot=50, seed=[2026, 0]))


def test_unit_weights_reproduce_row_rmse():
    obs = np.array([[0.1, 0.5, np.nan], [0.2, 0.2, 0.9]])
    pred = np.array([[0.2, 0.25, 0.3], [0.2, 0.7, 0.4]])
    valid = np.isfinite(obs)
    total, count = boot_rmse_sum(np.where(valid, (pred - obs) ** 2, 0.0), valid, np.ones((3, 1)))
    expected = row_stats(np.where(valid, obs, np.nan), pred)["rmse"].sum()
    assert np.isclose(total[0], expected) and count[0] == 2


def test_zero_weight_cpg_is_dropped_not_zeroed():
    obs = np.array([[0.0, 0.0], [0.0, 0.0]])
    pred = np.array([[0.5, 0.5], [0.25, 0.25]])
    valid = np.array([[True, False], [False, True]])
    total, count = boot_rmse_sum((pred - obs) ** 2, valid, np.array([[2.0], [0.0]]))
    assert count[0] == 1 and np.isclose(total[0], 0.5)
