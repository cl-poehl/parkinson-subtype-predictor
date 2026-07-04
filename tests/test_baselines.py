"""Sanity tests for src.baselines -- constants and API contract."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import BASELINE_DEFINITIONS, predict_baselines, _MODELS_DIR

# Trained models are not distributed publicly (DUA); tests that need them skip.
_MODELS_AVAILABLE = all(
    os.path.exists(os.path.join(_MODELS_DIR, f)) for _, f, _ in BASELINE_DEFINITIONS)
_needs_models = pytest.mark.skipif(
    not _MODELS_AVAILABLE, reason="trained models not distributed (DUA); see README")


def test_baseline_definitions_well_formed():
    """Each baseline definition is a triple (label, filename, auc string)."""
    assert len(BASELINE_DEFINITIONS) >= 1
    for entry in BASELINE_DEFINITIONS:
        assert len(entry) == 3
        label, fname, auc = entry
        assert isinstance(label, str) and label
        assert fname.endswith(".joblib")
        float(auc)


@_needs_models
def test_baseline_models_exist_on_disk():
    """The referenced joblibs actually exist in the models/ directory."""
    for _, fname, _ in BASELINE_DEFINITIONS:
        path = os.path.join(_MODELS_DIR, fname)
        assert os.path.exists(path), f"Missing baseline model file: {path}"


@_needs_models
def test_predict_baselines_row_shape():
    """When models are present, predict_baselines returns a list of
    well-formed dict entries. We take the feature names from the loaded
    bundles -- this keeps the test robust if the training-side feature
    names change."""
    import joblib

    all_cols = []
    for _, fname, _ in BASELINE_DEFINITIONS:
        bundle = joblib.load(os.path.join(_MODELS_DIR, fname))
        all_cols.extend(bundle["features"])
    all_cols = sorted(set(all_cols))

    features = pd.DataFrame([[1.0] * len(all_cols)], columns=all_cols)
    train_means = pd.Series([1.0] * len(all_cols), index=all_cols)

    rows = predict_baselines(features, train_means)
    assert isinstance(rows, list)
    assert len(rows) == len(BASELINE_DEFINITIONS)
    for r in rows:
        assert set(r.keys()) == {
            "Method", "P(Fast)", "Class at 0.5",
            "Discriminative AUC on PPMI",
        }
        assert r["P(Fast)"].endswith("%")
        assert r["Class at 0.5"] in ("Fast", "Slow")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
