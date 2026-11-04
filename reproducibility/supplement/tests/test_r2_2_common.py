"""Synthetic invariants for R2-2 site and sampling computations."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.r2_2_common import fit_compatible, nested_group_draws, pooled_metrics, site_metrics


def test_site_metrics_do_not_pool_between_site_means():
    y = np.array([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]])
    p = np.array([[0.3, 0.2, 0.1], [0.9, 0.8, 0.7]])
    metric = site_metrics(y, p)
    assert np.allclose(metric.pearson_r, -1)
    assert np.isclose(pooled_metrics(metric)["centered_pearson_r"], -1)
    assert np.corrcoef(y.ravel(), p.ravel())[0, 1] > 0


def test_missing_and_undefined_site_correlations():
    y = np.array([[0.2, np.nan, 0.6], [0.3, 0.3, 0.3], [0.1, 0.4, 0.7]])
    p = np.array([[0.2, 0.5, np.nan], [0.4, 0.5, 0.6], [0.5, 0.5, 0.5]])
    metric = site_metrics(y, p)
    assert metric.n.tolist() == [1, 3, 3]
    assert metric.correlation_reason.tolist() == ["insufficient_pairs", "constant_observed", "constant_prediction"]
    assert metric.pearson_r.isna().all()
    assert pooled_metrics(metric)["n_valid_pairs"] == 7


def test_fit_quadratic_and_kde_fallback():
    x = np.vstack([np.linspace(0, 1, 55), np.zeros(55)])
    y = np.vstack([0.1 + 0.2 * x[0] + 0.3 * x[0] ** 2, np.full(55, 0.82)])
    result = fit_compatible(x, y)
    assert result.fit_status.tolist() == ["quadratic", "constant_kde"]
    assert np.allclose(result.loc[0, ["a2h_intercept", "a2h_linear", "a2h_quadratic"]].astype(float), [0.1, 0.2, 0.3])
    assert np.isclose(result.a2h_intercept.iloc[1], 7 / 9)


def test_fit_rejects_unavailable_training_sites():
    x = np.tile(np.linspace(0, 1, 55), (2, 1))
    y = x.copy()
    y[0, :6] = np.nan
    quality = fit_compatible(x, y)
    assert quality.fit_status.iloc[0] == "insufficient"
    assert np.isnan(quality.a2h_intercept.iloc[0])
    assert quality.n_train_joint.tolist() == [49, 55]


def test_nested_draws_are_unique_and_stratified():
    names = [f"s{i}" for i in range(205)]
    group = [f"g{i % 39}" for i in range(205)]
    samples = pd.DataFrame({"sample_id": names, "donor_id": names, "Group": group})
    first = nested_group_draws(samples, seeds=[42, 43])
    second = nested_group_draws(samples, seeds=[42, 43])
    pd.testing.assert_frame_equal(first, second)
    for _, part in first.groupby("replicate"):
        for size in (75, 100, 150, 205):
            subset = part.loc[part["rank"] <= size]
            assert len(subset) == subset.sample_id.nunique() == size
            assert subset.Group.nunique() == 39
    with pytest.raises(ValueError):
        nested_group_draws(pd.concat([samples, samples.iloc[[0]]]), seeds=[42])


def test_metrics_invariant_to_common_sample_reorder_and_nan_predictions():
    y = np.array([[0.1, 0.3, 0.7, np.nan], [0.2, 0.4, 0.6, 0.8]])
    p = np.array([[0.2, 0.5, 0.6, 0.9], [0.3, np.nan, 0.5, 0.7]])
    base = site_metrics(y, p)
    reordered = site_metrics(y[:, [2, 0, 3, 1]], p[:, [2, 0, 3, 1]])
    pd.testing.assert_frame_equal(base, reordered, check_exact=False, rtol=1e-14, atol=1e-14)
    assert base.n.tolist() == [3, 3]
    assert np.isfinite(base.rmse).all()


def test_rank_deficient_scalar_polyfit_path():
    x = np.array([0.2] * 27 + [0.8] * 28, dtype=float)[None, :]
    y = (0.4 + 0.3 * x)
    quality = fit_compatible(x, y)
    assert quality.fit_status.iloc[0] == "quadratic_rank_warning"
    values = np.array([0.2, 0.5, 0.8])
    coefficients = quality.loc[0, ["a2h_quadratic", "a2h_linear", "a2h_intercept"]].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = np.polyval(np.polyfit(x.ravel(), y.ravel(), 2), values)
    assert np.allclose(np.polyval(coefficients, values), expected)
