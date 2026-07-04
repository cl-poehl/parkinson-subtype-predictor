"""Modell-Loading und Predictions inklusive Per-Fold-Vorhersagen
fuer Konfidenzintervalle."""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_models(model_files):
    """Pickled Modelle einmal beim App-Start laden."""
    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def predict_all(models, X):
    """Mean predict_proba pro Modell als DataFrame."""
    out = pd.DataFrame(index=X.index)
    for name, model in models.items():
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            proba = [float("nan")] * len(X)
        out[name] = proba
    return out


def predict_all_with_folds(models, X):
    """Wie predict_all, liefert aber zusaetzlich pro Patient pro Modell
    die K Vorhersagen aus den CalibratedClassifierCV-Folds. Daraus laesst
    sich eine min/max-Spanne als Modell-Konfidenzintervall ableiten.

    Returns (mean_df, folds_dict)
    - mean_df: DataFrame mit Spalten = Modellname, Werten = mittleres P(Fast)
    - folds_dict: {model_name: np.array shape (n_patients, K)}
    """
    mean_df = pd.DataFrame(index=X.index)
    folds_dict = {}
    Xv = X.values if hasattr(X, "values") else X

    for name, model in models.items():
        try:
            mean_p = model.predict_proba(X)[:, 1]
        except Exception:
            mean_p = [float("nan")] * len(X)
            folds_dict[name] = np.full((len(X), 1), np.nan)
            mean_df[name] = mean_p
            continue
        mean_df[name] = mean_p

        # Per-Fold-Vorhersagen aus calibrated_classifiers_
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            per_fold = []
            for inner in model.calibrated_classifiers_:
                try:
                    per_fold.append(inner.predict_proba(Xv)[:, 1])
                except Exception:
                    per_fold.append(np.full(len(X), np.nan))
            folds_dict[name] = np.stack(per_fold, axis=1)
        else:
            folds_dict[name] = np.array(mean_p).reshape(-1, 1)

    return mean_df, folds_dict
