"""The likelihood-ratio method for the web app.

Inference: given a patient's OLS slopes per score, compute the likelihoods under
the PPMI subtype distributions, derive the log10 LR per score, and sum them into
log10_lr_total. From log10_lr_total we obtain a pseudo-probability via logistic
calibration on the PPMI training set.

Percentiles: given a value, look it up against PPMI-fast and PPMI-slow.
"""
import os

import joblib
import numpy as np
import streamlit as st
from scipy import stats

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@st.cache_resource
def _load_reference(score_mode):
    path = os.path.join(MODELS_DIR, "auxiliary", f"lr_reference_{score_mode}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _likelihood_zscore(distribution, value):
    """Two-tailed Gaussian likelihood under the distribution. NaN if N is too small."""
    distribution = np.asarray(distribution)
    if distribution.size < 10:
        return np.nan
    mean = distribution.mean()
    std = distribution.std()
    if std == 0:
        return np.nan
    z = (value - mean) / std
    # two-tailed normal likelihood
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(p)


def _log10_lr(distribution_fast, distribution_slow, value):
    """log10(P(value | fast) / P(value | slow)).
    Clamped per score to +-1.3 (corresponds to an LR range of 0.05 - 20,
    analogous to the reference implementation `calc_likelihood_ratio` in the
    SubtypePredictions repo), so that no single score with a value near
    floating-point 0 dominates the total sum."""
    lf = _likelihood_zscore(distribution_fast, value)
    ls = _likelihood_zscore(distribution_slow, value)
    if np.isnan(lf) or np.isnan(ls):
        return np.nan
    if lf == 0 and ls == 0:
        return np.nan
    if lf == 0:
        return -1.3
    if ls == 0:
        return 1.3
    raw = float(np.log10(lf / ls))
    return max(-1.3, min(1.3, raw))


def lr_predict_from_slopes(slopes_dict, score_mode):
    """slopes_dict: {score_code: slope_value}. Returns dict with total_log10_lr,
    p_fast (sigmoid-calibrated), and detailed LR per score.
    Uses the per-subtype OLS slope distributions, because that is the unit in
    which we also compute the patient slopes."""
    ref = _load_reference(score_mode)
    if ref is None:
        return None

    # OLS slopes as the reference, because they share the same unit as the
    # patient-side slopes from extract_slope_intercept.
    slope_dists = ref.get("ols_slope_distributions", ref.get("slope_distributions", {}))
    per_score = {}
    total = 0.0
    contributed = 0
    for score, value in slopes_dict.items():
        if value is None or np.isnan(value) or score not in slope_dists:
            continue
        lr = _log10_lr(slope_dists[score].get(1, np.array([])),
                        slope_dists[score].get(2, np.array([])),
                        value)
        per_score[score] = lr
        if not np.isnan(lr):
            total += lr
            contributed += 1

    # Mathematically correct LR-to-probability conversion with a uniform
    # prior P(fast)=P(slow)=0.5:
    #   LR = 10^log10_LR_total = P(data|fast)/P(data|slow)
    #   P(fast|data) = LR / (LR + 1) = 1 / (1 + 10^(-log10_LR_total))
    # Previously I accidentally used np.exp(-total) (base e), which yielded
    # overly conservative probabilities (at log10_LR=1: 73% instead of 91%).
    p_fast = 1.0 / (1.0 + 10.0 ** (-total)) if contributed > 0 else 0.5

    return {
        "total_log10_lr": total,
        "p_fast": float(p_fast),
        "per_score": per_score,
        "contributed": contributed,
    }


def percentile_in_subtype(reference, score, value, subtype, dist_kind="slope"):
    """Percentile of a value within the subtype's PPMI distribution.
    dist_kind: 'slope' or 'intercept'. Uses the OLS slope distributions for
    consistency with the patient-side feature_extraction."""
    if dist_kind == "slope":
        key = "ols_slope_distributions"
        fallback = "slope_distributions"
    else:
        key = "intercept_distributions"
        fallback = None
    if not reference:
        return None
    dists = reference.get(key) or (reference.get(fallback) if fallback else None)
    if not dists or score not in dists:
        return None
    dist = dists[score].get(subtype, np.array([]))
    if dist.size < 10:
        return None
    pct = (dist < value).mean() * 100
    return float(pct)


def get_reference(score_mode):
    return _load_reference(score_mode)
