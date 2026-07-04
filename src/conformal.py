"""Conformal prediction wrapper for the calibrated classifiers.

Uses MAPIE 1.4+ with SplitConformalClassifier (prefit mode, LAC score).
One prediction set per patient ({Fast}, {Slow} or {Fast, Slow}) with
provable coverage 1-alpha (90% at confidence_level=0.9).

Workflow:
- Training: CalibratedClassifierCV on 80% of PPMI
- Conformal calibration on the 20% holdout
- Inference: calibrated classifier + conformal thresholds -> sets
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
    """Calibrate a SplitConformalClassifier (prefit) on held-out data."""
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
    """Returns list[list[str]] with prediction sets per patient."""
    if scc is None:
        return None
    Xv = X.values if hasattr(X, "values") else X
    y_pred, y_set = scc.predict_set(Xv)
    # y_set shape: (n_samples, n_classes) for a single confidence level
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
    """Load the conformal joblibs once at app startup."""
    out = {}
    for name, path in paths_dict.items():
        if os.path.exists(path):
            out[name] = joblib.load(path)
    return out
