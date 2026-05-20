"""Survival-Prediction-Inferenz: laedt das Cox-PH-Modell und gibt
geschaetzte Time-to-HY-3 fuer einen Patienten zurueck."""
import os

import joblib
import pandas as pd


_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models")
_COX_BUNDLE = None


def _load_cox():
    global _COX_BUNDLE
    if _COX_BUNDLE is None:
        path = os.path.join(_MODELS_DIR, "cox_survival.joblib")
        if not os.path.exists(path):
            return None
        _COX_BUNDLE = joblib.load(path)
    return _COX_BUNDLE


def predict_time_to_hy3(patient_features):
    """Predict median, 25%, 75% quantile time-to-HY-3 in months.

    patient_features: pd.DataFrame, single row, slope+intercept columns
        matching the Cox-Modell-Featureraum.

    Returns dict with keys: median, q25, q75 (each float or None) or
    None if the Cox model is not available.
    """
    bundle = _load_cox()
    if bundle is None:
        return None
    cox = bundle["cox"]
    cox_feats = bundle["features"]
    med = bundle["median_imp"]
    row = patient_features.copy()
    for c in cox_feats:
        if c not in row.columns:
            row[c] = med.get(c, 0.0)
        elif pd.isna(row[c].iloc[0]):
            row[c] = med.get(c, 0.0)
    row = row[cox_feats]
    try:
        sf = cox.predict_survival_function(row)
        col = sf.columns[0]
        t_med = sf.index[sf[col] <= 0.5]
        median = float(t_med[0]) if len(t_med) > 0 else None
        t_q25 = sf.index[sf[col] <= 0.75]
        q25 = float(t_q25[0]) if len(t_q25) > 0 else None
        t_q75 = sf.index[sf[col] <= 0.25]
        q75 = float(t_q75[0]) if len(t_q75) > 0 else None
        return {"median": median, "q25": q25, "q75": q75}
    except Exception:
        return None
