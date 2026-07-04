"""Per-patient slope/intercept features from longitudinal score data."""

import numpy as np
import pandas as pd
from scipy import stats

from constants import SCORE_LABELS


def _fit_per_score(group, scores, time_col, include_intercept):
    """Fit OLS per score for a single patient, return feature dict."""
    row = {}
    for score in scores:
        vals = group[[time_col, score]].dropna()
        if len(vals) < 2:
            if include_intercept:
                row[f"{score}_slope"] = np.nan
                row[f"{score}_intercept"] = np.nan
            else:
                row[score] = np.nan
            continue

        reg = stats.linregress(vals[time_col], vals[score])
        if include_intercept:
            row[f"{score}_slope"] = reg.slope
            row[f"{score}_intercept"] = reg.intercept
        else:
            row[score] = reg.slope
    return row


def extract_features(data, model_type, scores=None, patno_col="PATNO", time_col="Disease_duration"):
    if scores is None:
        scores = list(SCORE_LABELS.keys())

    include_intercept = (model_type == "slopes+intercepts")
    if model_type not in ("slopes", "slopes+intercepts"):
        raise ValueError(f"Unknown model_type '{model_type}'")

    features = {
        patno: _fit_per_score(group, scores, time_col, include_intercept)
        for patno, group in data.groupby(patno_col)
    }
    return pd.DataFrame(features).T


def get_labels(data, patno_col="PATNO", subtype_col="Subtype"):
    return data.groupby(patno_col)[subtype_col].first()
