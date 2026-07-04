"""Feature extraction from visit data."""
import numpy as np
import pandas as pd
from scipy import stats


def extract_slope_intercept(visits, scores, time_col="disease_duration"):
    """Extract slope+intercept features from a DataFrame of per-patient visits.

    visits: DataFrame with columns patno, disease_duration, plus the scores
    Returns: DataFrame index=patno, columns=<score>_slope, <score>_intercept
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
    """Only the first available value per score (single-visit model)."""
    rows = {}
    for patno, grp in visits.groupby("patno"):
        grp_sorted = grp.sort_values("disease_duration")
        row = {score: grp_sorted[score].dropna().iloc[0] if grp_sorted[score].notna().any()
               else np.nan for score in scores}
        rows[patno] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def imputation_flags(visits, scores, mode):
    """Per patient, per feature: will the value be imputed downstream (because
    there is not enough real data), or does it come from actual measurements?
    mode: 'slope' (>=2 visits required) or 'baseline' (>=1 visit required).
    Returns dict {patno: {feature_name: True if imputed, else False}}."""
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
    """A 3-level data-quality label per patient, per feature:

    - 'imputed': 0 or 1 measurement -> kNN-imputed (no real OLS fit possible)
    - 'low':     exactly 2 measurements -> OLS slope computable, but
                 statistically shaky (degenerate fit without residual information)
    - 'ok':      >=3 measurements -> robust OLS fit

    Baseline mode: 'imputed' if 0 measurements, otherwise 'ok'.

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
