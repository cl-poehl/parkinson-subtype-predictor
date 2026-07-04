"""Single-feature baseline models (UPDRS3-only LogReg, MoCA-only LogReg).

Trained in scripts/train_full_models.py on the full PPMI cohort;
this module only runs inference for incoming patient features."""
import os

import joblib
import pandas as pd


_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "auxiliary")
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
    """Return P(Fast) for both single-feature baselines.

    patient_features: DataFrame with 1 row; columns include the
        respective required slope+intercept features.
    train_means: pd.Series of the training column means (for NaN replacement).

    Returns a list of dicts with Method, P(Fast), Class at 0.5, AUC.
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
