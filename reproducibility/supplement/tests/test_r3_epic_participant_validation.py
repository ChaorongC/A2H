import numpy as np
import pytest

from src.r3_epic_participant_validation import participant_metrics


def test_observed_range_not_prediction_range():
    y = np.array([.2, .4, .6])
    p = np.array([.1, .3, .5])
    r = participant_metrics(y, p)
    assert r['rmse'] == pytest.approx(.1)
    assert r['observed_range'] == pytest.approx(.4)
    assert r['nrmse'] == pytest.approx(.25)
    assert r['pearson_r'] == pytest.approx(1)


def test_constant_truth_does_not_become_zero_nrmse():
    r = participant_metrics(np.array([.5, .5]), np.array([.4, .6]))
    assert r['rmse'] == pytest.approx(.1)
    assert np.isnan(r['nrmse']) and np.isnan(r['pearson_r'])
    assert r['nrmse_status'] == 'zero_observed_range'


def test_range_one_and_perfect_predictions():
    r = participant_metrics(np.array([0., .5, 1.]), np.array([0., .5, 1.]))
    assert r['n'] == 3 and r['observed_range'] == 1
    assert r['rmse'] == 0 and r['nrmse'] == 0


def test_reject_nonfinite_or_mismatched_values():
    with pytest.raises(ValueError):
        participant_metrics(np.array([np.nan]), np.array([.2]))
    with pytest.raises(ValueError):
        participant_metrics(np.array([.2, .3]), np.array([.2]))
