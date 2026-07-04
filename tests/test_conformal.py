"""Sanity tests for src.conformal -- MAPIE wrapper roundtrip with a mini classifier."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.conformal import fit_conformal, predict_sets, CLASS_LABELS, HAS_MAPIE


pytestmark = pytest.mark.skipif(
    not HAS_MAPIE, reason="MAPIE not installed in this environment",
)


@pytest.fixture
def trained_logreg_and_split():
    """Simple binary classifier with 3-feature input + held-out calibration split."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=400, n_features=3, n_informative=2,
        n_redundant=0, random_state=0,
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.4, random_state=0,
    )
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    return clf, X_calib, y_calib


def test_fit_conformal_returns_classifier(trained_logreg_and_split):
    clf, X_calib, y_calib = trained_logreg_and_split
    scc = fit_conformal(clf, X_calib, y_calib, confidence_level=0.9)
    assert scc is not None


def test_predict_sets_shape_and_labels(trained_logreg_and_split):
    clf, X_calib, y_calib = trained_logreg_and_split
    scc = fit_conformal(clf, X_calib, y_calib, confidence_level=0.9)
    sets = predict_sets(scc, X_calib[:10])
    assert len(sets) == 10
    for s in sets:
        # Each set is a non-empty subset of {"Slow", "Fast"}
        assert 1 <= len(s) <= 2
        assert all(label in CLASS_LABELS for label in s)


def test_predict_sets_empirical_coverage(trained_logreg_and_split):
    """Empirical coverage on the calibration set should be close to 90%."""
    clf, X_calib, y_calib = trained_logreg_and_split
    scc = fit_conformal(clf, X_calib, y_calib, confidence_level=0.9)
    sets = predict_sets(scc, X_calib)
    true_labels = [CLASS_LABELS[int(y)] for y in y_calib]
    covered = sum(1 for s, t in zip(sets, true_labels) if t in s)
    coverage = covered / len(true_labels)
    # MAPIE guarantees coverage marginally -- in-sample it should be >0.85
    assert coverage > 0.85, f"Coverage {coverage:.3f} below 0.85 sanity floor"


def test_predict_sets_none_returns_none():
    """predict_sets should handle None gracefully, not crash."""
    result = predict_sets(None, np.zeros((3, 2)))
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
