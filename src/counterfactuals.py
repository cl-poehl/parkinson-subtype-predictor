"""Counterfactual Explanations via DiCE + sparse single-feature analysis.

Wir liefern zwei Sichten:

1) **Single-feature counterfactuals**: pro Feature einzeln binaer-suchen,
   bei welchem Wert die Vorhersage flippt. Sortiert nach minimaler
   |delta|. Klinisch direkt interpretierbar ('wenn nur UPDRS3-Slope
   um 0.1 hoeher waere, waere die Vorhersage Fast').

2) **DiCE multi-feature counterfactuals** (optional, fuer Vollstaendigkeit):
   findet gemeinsame Mehr-Feature-Aenderungen, die zur Klassen-Flip
   fuehren. Genetic-Methode mit proximity_weight bevorzugt sparsame
   Loesungen.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import dice_ml
    HAS_DICE = True
except ImportError:
    HAS_DICE = False


def _pretty_feature(code, score_labels):
    if code.endswith("_slope"):
        return f"{score_labels.get(code[:-6], code[:-6])} (slope)"
    if code.endswith("_intercept"):
        return f"{score_labels.get(code[:-10], code[:-10])} (intercept)"
    return score_labels.get(code, code)


def single_feature_counterfactuals(model, query, X_train, n_top=5, score_labels=None):
    """Pro Feature suchen, bei welchem Wert die Klasse fuer den Patient flippt.

    Binaersuche entlang der Trainings-Feature-Verteilung (1.-99. Perzentil).
    Liefert die Top-N Features mit kleinster relativer Aenderung.

    Returns: list of dicts with feature, original, target, delta, direction
    """
    if score_labels is None:
        score_labels = {}
    feature_cols = list(X_train.columns)
    q = query[feature_cols].iloc[0].copy()
    p_query = float(model.predict_proba(query[feature_cols].values)[0, 1])
    target_class = 0 if p_query >= 0.5 else 1
    target_direction = "lower" if target_class == 0 else "higher"

    results = []
    for feat in feature_cols:
        orig = float(q[feat])
        # Feature-Verteilungs-Bereich
        lo = float(np.nanpercentile(X_train[feat].values, 1))
        hi = float(np.nanpercentile(X_train[feat].values, 99))
        if lo == hi:
            continue

        # Binaersuche in die Zielrichtung
        candidate = orig
        best_delta = None
        # Probiere 25 Werte zwischen orig und Extrem in beide Richtungen
        for endpoint in (lo, hi):
            if (endpoint - orig) * (1 if target_class == 1 else -1) <= 0 and target_class == 1:
                # endpoint geht in falsche Richtung wenn p_query > 0.5
                pass
            # Binaersuche
            a, b = orig, endpoint
            test_q = query[feature_cols].copy()
            test_q.iloc[0, feature_cols.index(feat)] = b
            p_b = float(model.predict_proba(test_q.values)[0, 1])
            if (p_b >= 0.5) == (target_class == 1):
                # Flip moeglich, jetzt binaersuche
                for _ in range(20):
                    mid = 0.5 * (a + b)
                    test_q.iloc[0, feature_cols.index(feat)] = mid
                    p_mid = float(model.predict_proba(test_q.values)[0, 1])
                    if (p_mid >= 0.5) == (target_class == 1):
                        b = mid
                    else:
                        a = mid
                delta = b - orig
                if best_delta is None or abs(delta) < abs(best_delta):
                    best_delta = delta
                    candidate = b
        if best_delta is None:
            continue
        results.append({
            "feature_code": feat,
            "feature": _pretty_feature(feat, score_labels),
            "original": orig,
            "target_value": candidate,
            "delta": best_delta,
            "rel_delta_pct": abs(best_delta) / (hi - lo) * 100,
        })

    if not results:
        return None
    df = pd.DataFrame(results).sort_values("rel_delta_pct").head(n_top).reset_index(drop=True)
    return df


@st.cache_resource
def _build_dice_genetic(model_key, _train_df, _model):
    if not HAS_DICE:
        return None
    feature_cols = [c for c in _train_df.columns if c != "target"]
    data = dice_ml.Data(dataframe=_train_df,
                         continuous_features=feature_cols,
                         outcome_name="target")
    dice_model = dice_ml.Model(model=_model, backend="sklearn",
                                model_type="classifier")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = dice_ml.Dice(data, dice_model, method="genetic")
    return exp


def dice_counterfactuals(model, query, X_train, y_train, n_cfs=3,
                          score_labels=None, model_key="default"):
    """DiCE Counterfactuals mit Genetic-Algorithmus + proximity weight.
    Returns DataFrame mit (Counterfactual, Feature, Patient, CF-Value, Delta)
    oder None bei Fehler."""
    if not HAS_DICE:
        return None, "DiCE not installed"
    if score_labels is None:
        score_labels = {}
    train_df = X_train.copy()
    train_df["target"] = np.asarray(y_train).astype(int)
    try:
        exp = _build_dice_genetic(model_key, train_df, model)
        if exp is None:
            return None, "DiCE setup failed"
        feature_cols = [c for c in train_df.columns if c != "target"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfs = exp.generate_counterfactuals(
                query[feature_cols], total_CFs=n_cfs,
                desired_class="opposite", verbose=False,
                proximity_weight=2.0,
                diversity_weight=1.0,
            )
        cf_df = cfs.cf_examples_list[0].final_cfs_df_sparse
        if cf_df is None or len(cf_df) == 0:
            return None, "No counterfactuals found"
        rows = []
        original = query[feature_cols].iloc[0]
        for i, (_, cf) in enumerate(cf_df.iterrows()):
            for feat in feature_cols:
                if feat not in cf.index:
                    continue
                orig_val = float(original[feat])
                new_val = float(cf[feat])
                if abs(new_val - orig_val) < 1e-4:
                    continue
                rows.append({
                    "Counterfactual": f"CF {i + 1}",
                    "Feature": _pretty_feature(feat, score_labels),
                    "Patient value": orig_val,
                    "Counterfactual value": new_val,
                    "Delta": new_val - orig_val,
                })
        if not rows:
            return None, "No changes needed"
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
