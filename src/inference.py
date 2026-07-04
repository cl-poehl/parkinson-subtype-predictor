"""Model loading and predictions, including per-fold predictions
for confidence intervals."""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_models(model_files):
    """Load the pickled models once at app startup."""
    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def predict_all(models, X):
    """Mean predict_proba per model as a DataFrame."""
    out = pd.DataFrame(index=X.index)
    for name, model in models.items():
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            proba = [float("nan")] * len(X)
        out[name] = proba
    return out


def predict_all_with_folds(models, X):
    """Like predict_all, but additionally returns, per patient per model,
    the K predictions from the CalibratedClassifierCV folds. From these a
    min/max range can be derived as a model confidence interval.

    Returns (mean_df, folds_dict)
    - mean_df: DataFrame with columns = model name, values = mean P(Fast)
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

        # per-fold predictions from calibrated_classifiers_
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
