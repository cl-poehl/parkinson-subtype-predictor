"""Feature-Extraktion aus Visit-Daten."""
import numpy as np
import pandas as pd
from scipy import stats


def extract_slope_intercept(visits, scores, time_col="disease_duration"):
    """Aus einem DataFrame mit Visits pro Patient die Slope+Intercept-Features ziehen.

    visits: DataFrame mit Spalten patno, disease_duration, plus den scores
    Rueckgabe: DataFrame index=patno, columns=<score>_slope, <score>_intercept
    """
    rows = {}
    for patno, grp in visits.groupby("patno"):
        row = {}
        for score in scores:
            vals = grp[[time_col, score]].dropna()
            if len(vals) < 2:
                row[f"{score}_slope"] = np.nan
                row[f"{score}_intercept"] = np.nan
            else:
                reg = stats.linregress(vals[time_col], vals[score])
                row[f"{score}_slope"] = reg.slope
                row[f"{score}_intercept"] = reg.intercept
        rows[patno] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def extract_baseline(visits, scores):
    """Nur den ersten verfuegbaren Wert pro Score (Single-Visit-Modell)."""
    rows = {}
    for patno, grp in visits.groupby("patno"):
        grp_sorted = grp.sort_values("disease_duration")
        row = {score: grp_sorted[score].dropna().iloc[0] if grp_sorted[score].notna().any()
               else np.nan for score in scores}
        rows[patno] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def imputation_flags(visits, scores, mode):
    """Pro Patient pro Feature: wird der Wert downstream imputiert (weil
    nicht genug echte Daten da sind), oder kommt er aus realen Messungen?
    mode: 'slope' (>=2 Visits noetig) oder 'baseline' (>=1 Visit noetig).
    Returns dict {patno: {feature_name: True wenn imputiert, sonst False}}."""
    flags = {}
    for patno, grp in visits.groupby("patno"):
        pat_flags = {}
        for score in scores:
            n_valid = grp[score].notna().sum()
            if mode == "slope":
                is_imputed = n_valid < 2
                pat_flags[f"{score}_slope"] = bool(is_imputed)
                pat_flags[f"{score}_intercept"] = bool(is_imputed)
            else:
                pat_flags[score] = bool(n_valid == 0)
        flags[patno] = pat_flags
    return flags


def feature_reliability(visits, scores, mode):
    """Pro Patient pro Feature ein 3-stufiges Datenqualitaets-Label:

    - 'imputed': 0 oder 1 Messung -> kNN-imputiert (kein realer OLS-Fit moeglich)
    - 'low':     genau 2 Messungen -> OLS-Slope berechenbar, aber statistisch
                 wackelig (degenerate fit ohne Residuen-Information)
    - 'ok':      >=3 Messungen -> belastbarer OLS-Fit

    Baseline-Modus: 'imputed' wenn 0 Messungen, sonst 'ok'.

    Returns dict {patno: {feature_name: 'imputed' | 'low' | 'ok'}}.
    """
    labels = {}
    for patno, grp in visits.groupby("patno"):
        pat_labels = {}
        for score in scores:
            n_valid = int(grp[score].notna().sum())
            if mode == "slope":
                if n_valid < 2:
                    lab = "imputed"
                elif n_valid == 2:
                    lab = "low"
                else:
                    lab = "ok"
                pat_labels[f"{score}_slope"] = lab
                pat_labels[f"{score}_intercept"] = lab
            else:
                pat_labels[score] = "imputed" if n_valid == 0 else "ok"
        labels[patno] = pat_labels
    return labels
