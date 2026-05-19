"""Vorhersage-Verlaesslichkeit basierend auf Missingness x Follow-Up Simulationen.
Lookup nach Score-Modus (luxpark vs. full), Klassifikator, Modelltyp, Missingness
und Follow-Up-Dauer."""
import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CLF_CODE = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Logistic Regression": "logistic_regression",
}


@st.cache_data
def _load_table(score_mode):
    """Tabelle fuer den gewuenschten Score-Modus laden.
    Fallback auf die alte 5-Score-Simulation, wenn webapp-spezifische
    Simulation noch nicht gelaufen ist."""
    specific = f"ml_missingness_followup_simulation_{score_mode}.csv"
    fallback = "ml_missingness_followup_simulation.csv"
    for fname in [specific, fallback]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.attrs["source"] = fname
            return df
    return None


def expected_auc(classifier_name, model_type, missingness, follow_up=None,
                  score_mode="luxpark"):
    """Erwartete AUC bei aehnlichen Bedingungen.
    Returns (auc, source) Tupel. source ist der Dateiname der genutzten Tabelle
    (fuer Transparenz, ob webapp-spezifisch oder Fallback)."""
    code = CLF_CODE.get(classifier_name)
    if code is None:
        return None, None
    df = _load_table(score_mode)
    if df is None:
        return None, None

    sub = df[(df["classifier"] == code) & (df["model_type"] == model_type)]
    # NaN-AUCs aus der Simulation rausfiltern (zu kurzes Follow-Up plus
    # Slope-Modell hat in manchen Bedingungen <2 Messungen pro Patient)
    sub = sub[sub["roc_auc"].notna()]
    if sub.empty:
        return None, df.attrs.get("source")

    if follow_up is None or follow_up <= 0:
        # Nur Missingness, nimm den kuerzesten Follow-Up
        sub = sub[sub["follow_up"] == sub["follow_up"].min()]
        d = (sub["missingness"] - missingness).abs()
    else:
        d = ((sub["missingness"] - missingness) ** 2 +
             ((sub["follow_up"] - follow_up) / 120) ** 2)

    val = float(sub.loc[d.idxmin(), "roc_auc"])
    if np.isnan(val):
        return None, df.attrs.get("source")
    return val, df.attrs.get("source")


def reliability_label(auc):
    if auc is None or np.isnan(auc):
        return "unbekannt", "gray"
    if auc >= 0.90:
        return "hoch", "#1f8a3a"
    if auc >= 0.80:
        return "mittel", "#d39e00"
    return "niedrig", "#c0392b"
