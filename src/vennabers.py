"""Inductive Venn-Abers predictor (Vovk & Petej 2014) for per-patient calibrated
probability intervals.

A Venn-Abers predictor turns a scoring classifier into a *multiprobability*
prediction: for each instance it returns an interval [p0, p1] that is guaranteed
to be well-calibrated under exchangeability, with the width reflecting how much
the calibration data constrains the probability locally. We fit it on the same
held-out calibration split the split-conformal predictor uses, so one calibration
set powers both the prediction set (which label) and the probability interval
(how sure), giving a single distribution-free uncertainty framework.

This replaces the earlier per-patient confidence interval taken from the five
internal isotonic-calibration folds, which was only an informal stability cue.

Implementation: inductive VAP (IVAP). For calibration scores s_i with labels
y_i and a test score s, p0 is the isotonic fit of the calibration set augmented
with (s, 0) evaluated at s, and p1 the same with (s, 1); p0 <= p1 always. The
merged point probability is p1 / (1 - p0 + p1) (Vovk's minimax formula).
"""
import os

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_states(paths):
    """Load VA calibrators from {name: path}; skip any that are missing."""
    out = {}
    for name, rel in paths.items():
        full = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
        if os.path.exists(full):
            out[name] = joblib.load(full)
    return out


def fit(cal_scores, cal_labels):
    """Return the VA calibrator state (just the sorted calibration pairs)."""
    s = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_labels, dtype=float)
    order = np.argsort(s)
    return {"cal_scores": s[order], "cal_labels": y[order]}


def _interval_one(cal_s, cal_y, s):
    s0 = np.append(cal_s, s)
    y0 = np.append(cal_y, 0.0)
    p0 = IsotonicRegression(out_of_bounds="clip").fit(s0, y0).predict([s])[0]
    s1 = np.append(cal_s, s)
    y1 = np.append(cal_y, 1.0)
    p1 = IsotonicRegression(out_of_bounds="clip").fit(s1, y1).predict([s])[0]
    lo, hi = (p0, p1) if p0 <= p1 else (p1, p0)
    return float(lo), float(hi)


def predict_intervals(state, scores):
    """For each test score return (p_lo, p_hi, p_point) where [p_lo, p_hi] is the
    Venn-Abers calibrated probability interval for class 1 (Fast) and p_point is
    the merged point probability."""
    cal_s, cal_y = state["cal_scores"], state["cal_labels"]
    out = []
    for s in np.asarray(scores, dtype=float):
        lo, hi = _interval_one(cal_s, cal_y, s)
        denom = 1.0 - lo + hi
        point = hi / denom if denom > 0 else hi
        out.append((lo, hi, float(point)))
    return out
