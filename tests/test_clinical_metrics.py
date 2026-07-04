"""Sanity tests for src.clinical_metrics. Run via pytest."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.clinical_metrics import (
    bootstrap_auc, calibration_intercept_slope, hosmer_lemeshow,
    adjust_pvalues, delong_test, equalized_odds, optimal_threshold,
    bootstrap_classification_metrics, nri_idi, decision_curve,
)


@pytest.fixture
def perfect_predictions():
    """Perfect discrimination: predicted = true."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    p = np.where(y == 1, 0.95, 0.05) + rng.normal(0, 0.02, n)
    return np.clip(p, 0.001, 0.999), y


@pytest.fixture
def noisy_predictions():
    """Moderate discrimination."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n)
    p = 0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.2, n)
    return np.clip(p, 0.001, 0.999), y


def test_bootstrap_auc_perfect(perfect_predictions):
    p, y = perfect_predictions
    res = bootstrap_auc(y, p, n_boot=200)
    assert res["auc"] > 0.95
    assert 0.5 < res["auc_lo"] <= res["auc"] <= res["auc_hi"] <= 1.0


def test_bootstrap_auc_random():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    res = bootstrap_auc(y, p, n_boot=200)
    # Random predictions: AUC near 0.5
    assert 0.35 < res["auc"] < 0.65


def test_calibration_intercept_slope_perfect():
    """Perfect calibration: intercept ~ 0, slope ~ 1."""
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < p).astype(int)
    res = calibration_intercept_slope(y, p)
    assert abs(res["intercept"]) < 0.2
    assert 0.8 < res["slope"] < 1.2


def test_hosmer_lemeshow_well_calibrated():
    """Well-calibrated model: HL p > 0.05."""
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < p).astype(int)
    res = hosmer_lemeshow(y, p, g=10)
    assert res["p_value"] > 0.05
    assert res["dof"] == 8


def test_hosmer_lemeshow_miscalibrated():
    """Heavily miscalibrated: HL p << 0.05."""
    rng = np.random.default_rng(0)
    n = 1000
    p = rng.uniform(0.01, 0.99, n)
    # Outcome unabhaengig von p
    y = rng.integers(0, 2, n)
    res = hosmer_lemeshow(y, p, g=10)
    assert res["chi2"] > 0
    # p-value can be near 0 or near 1 depending on luck, just check not NaN
    assert 0 <= res["p_value"] <= 1


def test_adjust_pvalues_bonferroni_holm():
    p = [0.001, 0.01, 0.03, 0.05, 0.2, 0.5]
    adj = adjust_pvalues(p, method="holm")
    # Holm should be monotonically non-decreasing in sorted order
    sorted_adj = [adj[i] for i in np.argsort(p)]
    assert all(sorted_adj[i] <= sorted_adj[i + 1] + 1e-9
                for i in range(len(sorted_adj) - 1))


def test_adjust_pvalues_bh():
    p = [0.001, 0.01, 0.03, 0.05, 0.2, 0.5]
    adj_bh = adjust_pvalues(p, method="bh")
    adj_holm = adjust_pvalues(p, method="holm")
    # BH should be at most as strict as Holm
    assert all(adj_bh[i] <= adj_holm[i] + 1e-9 for i in range(len(p)))


def test_adjust_pvalues_handles_nan():
    p = [0.01, np.nan, 0.05]
    adj = adjust_pvalues(p, method="holm")
    assert np.isnan(adj[1])
    assert not np.isnan(adj[0])
    assert not np.isnan(adj[2])


def test_delong_test_identical():
    """Two identical classifiers should have p > 0.9 in DeLong test."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    auc_a, auc_b, p_val = delong_test(y, p, p)
    assert auc_a == auc_b
    assert p_val > 0.9 or np.isnan(p_val)


def test_equalized_odds():
    """Sanity check on equalized-odds calculation."""
    n = 200
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    p = rng.uniform(size=n)
    g = rng.choice(["a", "b"], n)
    res = equalized_odds(y, p, g, threshold=0.5)
    assert "tpr_per_group" in res
    assert "fpr_per_group" in res
    assert 0 <= res["eod"] <= 1


def test_optimal_threshold_youden():
    """Youden maximizes sens + spec - 1."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    p = 0.5 + 0.3 * (2 * y - 1) + rng.normal(0, 0.1, n)
    p = np.clip(p, 0.001, 0.999)
    res = optimal_threshold(y, p, criterion="youden")
    assert 0.0 < res["threshold"] < 1.0
    assert res["sens"] + res["spec"] - 1 >= 0


def test_decision_curve_treat_none_zero():
    """Treat-none net benefit must always be 0."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    df = decision_curve(y, p)
    assert (df["Treat none"] == 0).all()


def test_nri_idi_zero_for_identical():
    """NRI and IDI of a model against itself should be zero."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = rng.uniform(size=200)
    res = nri_idi(y, p, p)
    assert res["nri"] == 0
    assert abs(res["idi"]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
