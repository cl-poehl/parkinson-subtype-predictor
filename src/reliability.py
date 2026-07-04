"""Prediction reliability based on missingness x follow-up simulations.
Looked up by score mode (luxpark vs. full), classifier, model type, missingness
and follow-up duration."""
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
    """Load the table for the requested score mode.
    Falls back to the old 5-score simulation if the web-app-specific
    simulation has not run yet."""
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
    """Expected AUC under similar conditions.

    The primary source is the bootstrap table (auc_mean by missingness),
    because the 2D table (missingness x follow_up) contained NaN under many
    conditions and the few valid rows often yielded AUC=1.0 (cohorts too small
    at min_follow_up=120 plus shorten_follow_up). Falls back to the 2D table if
    the bootstrap is not available.
    Returns (auc, source)."""
    code = CLF_CODE.get(classifier_name)
    if code is None:
        return None, None

    # Primary source: bootstrap table per score set
    boot = _load_bootstrap_table(score_mode)
    if boot is not None:
        sub = boot[boot["classifier"] == code].copy()
        sub = sub[sub["auc_mean"].notna()]
        if not sub.empty:
            d = (sub["missingness"] - missingness).abs()
            return (float(sub.loc[d.idxmin(), "auc_mean"]),
                    f"ml_missingness_bootstrap_{score_mode}.csv")

    # Fallback: old 2D table
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
    """Score-set-specific bootstrap AUC + CI as a function of missingness.
    Returns a DataFrame or None if the file does not exist."""
    fname = f"ml_missingness_bootstrap_{score_mode}.csv"
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def expected_auc_ci(classifier_name, missingness, score_mode="luxpark"):
    """95% bootstrap CI for the expected AUC at the given missingness fraction
    and score set. Returns (auc_mean, auc_lo, auc_hi) or (None, None, None) if
    no data is available.

    Nearest-neighbor lookup in missingness. The follow-up variable is not
    modeled here, because the 1D bootstrap table averages over follow-up; for a
    conservative estimate of the expected accuracy at the current data quality
    that is sufficient."""
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
