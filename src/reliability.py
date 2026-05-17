"""Vorhersage-Verlaesslichkeit basierend auf den Missingness-Simulationen
aus dem Hauptprojekt. Wir nutzen die AUC bei aehnlicher Missingness und
Follow-Up-Dauer als Schaetzung dafuer, wie gut die Klassifikation bei den
Daten des aktuellen Patienten zu vertrauen ist."""
import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Anzeige-Name -> Code in der Simulation
CLF_CODE = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Logistic Regression": "logistic_regression",
}


@st.cache_data
def _load_tables():
    miss = pd.read_csv(os.path.join(DATA_DIR, "ml_missingness_simulation.csv"))
    miss_fu = pd.read_csv(os.path.join(DATA_DIR, "ml_missingness_followup_simulation.csv"))
    return miss, miss_fu


def expected_auc(classifier_name, model_type, missingness, follow_up=None):
    """AUC bei aehnlichen Bedingungen aus der Simulation interpolieren.
    classifier_name als Anzeigename ('Random Forest' etc.), model_type 'slopes' oder
    'slopes+intercepts', missingness in [0,1], follow_up in Monaten.
    Returns float oder None."""
    code = CLF_CODE.get(classifier_name)
    if code is None:
        return None
    miss, miss_fu = _load_tables()

    # 2D Lookup wenn follow_up bekannt und >0
    if follow_up is not None and follow_up > 0:
        sub = miss_fu[(miss_fu["classifier"] == code) &
                      (miss_fu["model_type"] == model_type)]
        if not sub.empty:
            # Nearest neighbor in normalisierter Distanz
            d = ((sub["missingness"] - missingness) ** 2 +
                 ((sub["follow_up"] - follow_up) / 120) ** 2)
            return float(sub.loc[d.idxmin(), "roc_auc"])

    # 1D Fallback nur ueber Missingness
    sub = miss[(miss["classifier"] == code) & (miss["model_type"] == model_type)]
    if sub.empty:
        return None
    d = (sub["missingness"] - missingness).abs()
    return float(sub.loc[d.idxmin(), "roc_auc"])


def reliability_label(auc):
    """Qualitatives Label und Farbe fuer eine AUC."""
    if auc is None or np.isnan(auc):
        return "unbekannt", "gray"
    if auc >= 0.90:
        return "hoch", "#1f8a3a"
    if auc >= 0.80:
        return "mittel", "#d39e00"
    return "niedrig", "#c0392b"
