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
