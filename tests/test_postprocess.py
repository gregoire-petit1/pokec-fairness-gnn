"""Smoke tests for the equal-opportunity / demographic-parity post-processor."""

import numpy as np
import pytest

from src.postprocess.equal_opportunity import (
    apply_thresholds,
    calibrate_predictions,
    fit_thresholds,
)


@pytest.mark.smoke
def test_threshold_equalises_positive_rate():
    """When proba differs by group, fit_thresholds(DP) should equalise the
    positive-prediction rate across groups."""
    rng = np.random.default_rng(0)
    n_per = 500
    # Group 0: probas centred on 0.7 (most predicted +), group 1: centred on 0.3
    p0 = np.clip(rng.normal(0.7, 0.1, n_per), 0.01, 0.99)
    p1 = np.clip(rng.normal(0.3, 0.1, n_per), 0.01, 0.99)
    proba = np.concatenate([p0, p1])
    sensitive = np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(np.int64)
    y_true = (proba > 0.5).astype(np.int64)  # well-calibrated dummy

    thresholds = fit_thresholds(proba, y_true, sensitive, strategy="demographic_parity")
    pred = apply_thresholds(proba, sensitive, thresholds)

    rate_0 = pred[sensitive == 0].mean()
    rate_1 = pred[sensitive == 1].mean()
    # Equalised within tolerance — sample noise of ~0.02 is acceptable
    assert abs(rate_0 - rate_1) < 0.05


@pytest.mark.smoke
def test_apply_thresholds_is_vectorised_and_correct():
    proba = np.array([0.4, 0.6, 0.8, 0.2])
    sensitive = np.array([0, 1, 0, 1])
    thresholds = {0: 0.5, 1: 0.7}
    out = apply_thresholds(proba, sensitive, thresholds)
    # row 0: g=0 t=0.5 p=0.4 → 0
    # row 1: g=1 t=0.7 p=0.6 → 0
    # row 2: g=0 t=0.5 p=0.8 → 1
    # row 3: g=1 t=0.7 p=0.2 → 0
    np.testing.assert_array_equal(out, [0, 0, 1, 0])


@pytest.mark.smoke
def test_equal_opportunity_uses_only_y_true_positives():
    """In equal_opportunity strategy, the calibration target is the TPR among
    y_true=1, not the marginal positive rate."""
    proba = np.array([0.9, 0.9, 0.1, 0.1, 0.6, 0.4])
    y_true = np.array([1, 1, 1, 1, 0, 0])
    sensitive = np.array([0, 1, 0, 1, 0, 1])
    thresholds = fit_thresholds(proba, y_true, sensitive, strategy="equal_opportunity")
    # Target rate = mean((proba >= 0.5)[y=1]) = mean([1,1,0,0]) = 0.5
    # Group 0 (y=1): probas [0.9, 0.1], target 1 of 2 → threshold = 0.9
    # Group 1 (y=1): probas [0.9, 0.1], target 1 of 2 → threshold = 0.9
    assert thresholds[0] == 0.9
    assert thresholds[1] == 0.9


@pytest.mark.smoke
def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        fit_thresholds(np.array([0.5]), np.array([0]), np.array([0]), strategy="bogus")


@pytest.mark.smoke
def test_calibrate_predictions_round_trip():
    """End-to-end: fit on calib, apply on eval, returns predictions and thresholds."""
    rng = np.random.default_rng(1)
    n = 200
    proba_calib = rng.uniform(0, 1, n)
    y_calib = (proba_calib > 0.5).astype(np.int64)
    s_calib = rng.integers(0, 2, n).astype(np.int64)
    proba_eval = rng.uniform(0, 1, n)
    s_eval = rng.integers(0, 2, n).astype(np.int64)

    pred, thr = calibrate_predictions(
        proba_calib, y_calib, s_calib, proba_eval, s_eval, strategy="equal_opportunity"
    )
    assert pred.shape == (n,)
    assert pred.dtype == np.int64
    assert set(thr.keys()) == {0, 1}
    for t in thr.values():
        assert 0.0 <= t <= 1.0
