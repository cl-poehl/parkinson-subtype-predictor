"""Modell-Loading und Predictions."""
import os
import joblib
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
    """Predict_proba fuer alle geladenen Modelle.
    Rueckgabe: DataFrame mit Spalten <model>_prob_fast pro Patient."""
    out = pd.DataFrame(index=X.index)
    for name, model in models.items():
        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception as e:
            proba = [float("nan")] * len(X)
        out[name] = proba
    return out
