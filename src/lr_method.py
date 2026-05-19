"""Toms Likelihood-Ratio-Methode fuer die Webapp.

Inference: gegeben die OLS-Slopes eines Patienten pro Score, berechne unter den
PPMI-Subtyp-Verteilungen die Likelihoods, daraus den log10-LR pro Score und
summiere zum log10_lr_total. Aus log10_lr_total ergibt sich eine Pseudo-
Wahrscheinlichkeit via Logistic-Calibration auf dem PPMI-Trainingsset.

Perzentile: gegeben einen Wert, lookup unter PPMI-fast und PPMI-slow.
"""
import os

import joblib
import numpy as np
import streamlit as st
from scipy import stats

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


@st.cache_resource
def _load_reference(score_mode):
    path = os.path.join(MODELS_DIR, f"lr_reference_{score_mode}.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _likelihood_zscore(distribution, value):
    """Two-tailed Gaussian likelihood unter der Verteilung. NaN bei zu kleinem N."""
    distribution = np.asarray(distribution)
    if distribution.size < 10:
        return np.nan
    mean = distribution.mean()
    std = distribution.std()
    if std == 0:
        return np.nan
    z = (value - mean) / std
    # two-tailed Normal-Likelihood
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(p)


def _log10_lr(distribution_fast, distribution_slow, value):
    """log10(P(value | fast) / P(value | slow)).
    Per-Score auf +-1.3 begrenzt (entspricht LR-Bereich 0.05 - 20, analog zu
    Toms calc_likelihood_ratio), damit kein Einzelscore mit floating-point-
    nahem 0 die Gesamtsumme dominiert."""
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
    """slopes_dict: {score_code: slope_value}. Returns dict mit total_log10_lr,
    p_fast (sigmoid-kalibriert), und Detail-LR pro Score.
    Nutzt die OLS-Slope-Verteilungen pro Subtyp, weil das die Einheit ist, in
    der wir auch die Patient-Slopes berechnen."""
    ref = _load_reference(score_mode)
    if ref is None:
        return None

    # OLS-Slopes als Referenz, weil sie die gleiche Einheit haben wie die
    # patient-side slopes aus extract_slope_intercept.
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

    # Sehr einfache Wahrscheinlichkeits-Kalibrierung: Sigmoid auf total
    # (matched empirisch grob die PPMI-Konsens-Distribution).
    # Steilheit grob auf log10_lr_total von [-5, +5] -> [~0, ~1] geeicht.
    p_fast = 1.0 / (1.0 + np.exp(-total)) if contributed > 0 else 0.5

    return {
        "total_log10_lr": total,
        "p_fast": float(p_fast),
        "per_score": per_score,
        "contributed": contributed,
    }


def percentile_in_subtype(reference, score, value, subtype, dist_kind="slope"):
    """Perzentil eines Werts in der PPMI-Verteilung des Subtyps.
    dist_kind: 'slope' oder 'intercept'. Nutzt die OLS-Slope-Verteilungen
    fuer Konsistenz mit der patient-side feature_extraction."""
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
