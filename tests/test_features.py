"""Sanity tests fuer src.features. Run via pytest."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    extract_slope_intercept, extract_baseline,
    imputation_flags, feature_reliability,
)


@pytest.fixture
def two_patient_visits():
    return pd.DataFrame({
        "patno": [1, 1, 1, 2, 2],
        "disease_duration": [0, 12, 24, 0, 12],
        "updrs3": [10, 14, 18, 20, 22],
        "moca": [28, 27, 26, np.nan, 24],
    })


def test_extract_slope_intercept_known_values(two_patient_visits):
    """Patient 1: updrs3 von 10 auf 18 ueber 24 Monate -> slope ~0.333, intercept 10."""
    feats = extract_slope_intercept(two_patient_visits, ["updrs3", "moca"])
    assert set(feats.columns) == {
        "updrs3_slope", "updrs3_intercept", "moca_slope", "moca_intercept",
    }
    assert abs(feats.loc[1, "updrs3_slope"] - 0.3333) < 1e-3
    assert abs(feats.loc[1, "updrs3_intercept"] - 10.0) < 1e-3


def test_extract_slope_intercept_nan_below_two_obs():
    """Patient mit nur einer Messung pro Score -> NaN-Slope und NaN-Intercept."""
    df = pd.DataFrame({
        "patno": [1], "disease_duration": [0], "updrs3": [10],
    })
    feats = extract_slope_intercept(df, ["updrs3"])
    assert np.isnan(feats.loc[1, "updrs3_slope"])
    assert np.isnan(feats.loc[1, "updrs3_intercept"])


def test_extract_baseline_first_value(two_patient_visits):
    """Baseline = erster nicht-NaN Wert pro Score."""
    feats = extract_baseline(two_patient_visits, ["updrs3", "moca"])
    assert feats.loc[1, "updrs3"] == 10
    assert feats.loc[2, "updrs3"] == 20
    # Patient 2 hat NaN bei Visit 0 fuer moca, der erste echte Wert ist 24
    assert feats.loc[2, "moca"] == 24


def test_imputation_flags_slope_mode(two_patient_visits):
    flags = imputation_flags(two_patient_visits, ["updrs3", "moca"], mode="slope")
    # Patient 1 hat 3 updrs3-Werte: nicht imputiert
    assert flags[1]["updrs3_slope"] is False
    # Patient 2 hat nur 1 moca-Wert: imputiert
    assert flags[2]["moca_slope"] is True


def test_feature_reliability_three_levels(two_patient_visits):
    rel = feature_reliability(two_patient_visits, ["updrs3", "moca"], mode="slope")
    # Patient 1: 3 updrs3-Werte -> ok, 3 moca-Werte -> ok
    assert rel[1]["updrs3_slope"] == "ok"
    # Patient 2: 2 updrs3-Werte -> low, 1 moca-Wert -> imputed
    assert rel[2]["updrs3_slope"] == "low"
    assert rel[2]["moca_slope"] == "imputed"


def test_feature_reliability_baseline_mode(two_patient_visits):
    rel = feature_reliability(two_patient_visits, ["updrs3", "moca"], mode="baseline")
    # Im Baseline-Modus: 'ok' wenn >=1 Messung, 'imputed' wenn 0
    assert rel[1]["updrs3"] == "ok"
    assert rel[2]["moca"] == "ok"  # Patient 2 hat eine moca-Messung (24)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
