"""Single-Feature Baseline-Modelle (UPDRS3-only LogReg, MoCA-only LogReg).

Trainiert in scripts/train_full_models.py auf der gesamten PPMI-Kohorte,
hier nur Inferenz fuer eingehende Patient-Features."""
import os

import joblib
import pandas as pd


_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models")
_BASELINES = None

BASELINE_DEFINITIONS = (
    ("UPDRS3-on only (LogReg)", "baseline_updrs3_only.joblib", "0.73"),
    ("MoCA only (LogReg)", "baseline_moca_only.joblib", "0.76"),
)


def _load_baselines():
    global _BASELINES
    if _BASELINES is None:
        _BASELINES = []
        for label, fname, auc_label in BASELINE_DEFINITIONS:
            path = os.path.join(_MODELS_DIR, fname)
            if not os.path.exists(path):
                continue
            bundle = joblib.load(path)
            _BASELINES.append({
                "label": label, "pipeline": bundle["pipeline"],
                "features": bundle["features"], "auc": auc_label,
            })
    return _BASELINES


def predict_baselines(patient_features, train_means):
    """Liefert P(Fast) fuer beide Single-Feature Baselines.

    patient_features: DataFrame mit 1 Reihe, Spalten umfassen die jeweils
        benoetigten slope+intercept-Features.
    train_means: pd.Series der Trainings-Spalten-Means (fuer NaN-Replace).

    Returns Liste an dicts mit Method, P(Fast), Class at 0.5, AUC.
    """
    baselines = _load_baselines()
    rows = []
    for bd in baselines:
        sub = patient_features[bd["features"]].copy()
        if sub.isna().any().any():
            sub = sub.fillna(train_means)
        try:
            p = float(bd["pipeline"].predict_proba(sub.values)[0, 1])
        except Exception:
            continue
        rows.append({
            "Method": bd["label"],
            "P(Fast)": f"{p*100:.1f}%",
            "Class at 0.5": "Fast" if p >= 0.5 else "Slow",
            "Discriminative AUC on PPMI": bd["auc"],
        })
    return rows
