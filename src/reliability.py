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

    Primaere Quelle ist die Bootstrap-Tabelle (auc_mean nach Missingness),
    weil die 2D-Tabelle (missingness x follow_up) in vielen Bedingungen NaN
    enthielt und die wenigen validen Zeilen oft AUC=1.0 ergaben (zu kleine
    Cohorts bei min_follow_up=120 plus shorten_follow_up). Fallback auf die
    2D-Tabelle falls Bootstrap nicht verfuegbar.
    Returns (auc, source)."""
    code = CLF_CODE.get(classifier_name)
    if code is None:
        return None, None

    # Primaere Quelle: Bootstrap-Tabelle pro Score-Set
    boot = _load_bootstrap_table(score_mode)
    if boot is not None:
        sub = boot[boot["classifier"] == code].copy()
        sub = sub[sub["auc_mean"].notna()]
        if not sub.empty:
            d = (sub["missingness"] - missingness).abs()
            return (float(sub.loc[d.idxmin(), "auc_mean"]),
                    f"ml_missingness_bootstrap_{score_mode}.csv")

    # Fallback: alte 2D-Tabelle
    df = _load_table(score_mode)
    if df is None:
        return None, None
    sub = df[(df["classifier"] == code) & (df["model_type"] == model_type)]
    sub = sub[sub["roc_auc"].notna()]
    if sub.empty:
        return None, df.attrs.get("source")

    if follow_up is None or follow_up <= 0:
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


@st.cache_data
def _load_bootstrap_table(score_mode):
    """Score-set-spezifische Bootstrap-AUC + CI als Funktion der Missingness.
    Returns DataFrame oder None falls die Datei nicht existiert."""
    fname = f"ml_missingness_bootstrap_{score_mode}.csv"
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def expected_auc_ci(classifier_name, missingness, score_mode="luxpark"):
    """95%-Bootstrap-CI fuer die erwartete AUC bei dem gegebenen Missingness-
    Anteil und Score-Set. Liefert (auc_mean, auc_lo, auc_hi) oder
    (None, None, None) falls keine Daten verfuegbar.

    Lookup nearest neighbor in missingness. Die Folge-Variable Follow-Up ist
    hier nicht modelliert, weil die 1D-Bootstrap-Tabelle ueber Follow-Up
    mittelt; fuer eine konservative Schaetzung der erwarteten Genauigkeit
    bei der aktuellen Datenqualitaet reicht das."""
    code = CLF_CODE.get(classifier_name)
    if code is None:
        return None, None, None
    df = _load_bootstrap_table(score_mode)
    if df is None:
        return None, None, None
    sub = df[df["classifier"] == code].copy()
    sub = sub[sub["auc_mean"].notna()]
    if sub.empty:
        return None, None, None
    d = (sub["missingness"] - missingness).abs()
    row = sub.loc[d.idxmin()]
    return (float(row["auc_mean"]), float(row["auc_lo"]), float(row["auc_hi"]))
