"""Conformal Prediction Wrapper fuer die kalibrierten Klassifikatoren.

Nutzt MAPIE 1.4+ mit SplitConformalClassifier (prefit-Modus, LAC-Score).
Per Patient ein Prediction Set ({Fast}, {Slow} oder {Fast, Slow}) mit
nachweislicher Coverage 1-alpha (90% bei confidence_level=0.9).

Workflow:
- Training: CalibratedClassifierCV auf 80% PPMI
- Conformal-Kalibrierung auf den 20% Holdout
- Inference: kalibrierter Klassifikator + Conformal-Schwellen -> Sets
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from mapie.classification import SplitConformalClassifier
    HAS_MAPIE = True
except ImportError:
    HAS_MAPIE = False

CLASS_LABELS = ["Slow", "Fast"]  # 0=Slow, 1=Fast


def fit_conformal(base_estimator, X_calib, y_calib, confidence_level=0.9):
    """SplitConformalClassifier (prefit) auf Held-Out-Daten kalibrieren."""
    if not HAS_MAPIE:
        return None
    sc = SplitConformalClassifier(
        estimator=base_estimator,
        confidence_level=confidence_level,
        conformity_score="lac",
        prefit=True,
        random_state=42,
    )
    sc.conformalize(X_calib, y_calib)
    return sc


def predict_sets(scc, X):
    """Returns list[list[str]] mit prediction sets pro Patient."""
    if scc is None:
        return None
    Xv = X.values if hasattr(X, "values") else X
    y_pred, y_set = scc.predict_set(Xv)
    # y_set shape: (n_samples, n_classes) bei single confidence level
    if y_set.ndim == 3:
        y_set = y_set[..., 0]
    results = []
    for i in range(len(Xv)):
        mask = y_set[i]
        labels = [CLASS_LABELS[c] for c, m in enumerate(mask) if m]
        if not labels:
            labels = [CLASS_LABELS[int(y_pred[i])]]
        results.append(labels)
    return results


@st.cache_resource
def load_conformal_set(paths_dict):
    """Conformal-Joblibs einmal beim App-Start laden."""
    out = {}
    for name, path in paths_dict.items():
        if os.path.exists(path):
            out[name] = joblib.load(path)
    return out
