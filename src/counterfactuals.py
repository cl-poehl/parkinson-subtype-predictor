"""Counterfactual Explanations via DiCE (Mothilal et al. 2020).

Beantwortet: 'Welche kleinen Feature-Aenderungen wuerden die Klasse fuer
diesen Patient flippen?' Antwort: Top-N alternative Feature-Sets mit
minimaler Distanz zum aktuellen Input, die zu der jeweils anderen Klasse
fuehren.

Wir nutzen die kdtree-Methode (Mahalanobis-Distanz, schnell), trainiert
auf PPMI-Slope-Features. Diversitaet ueber bis zu 3 Counterfactuals.
"""
import os
import sys
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


SCORE_LABELS_LOCAL = None  # injected from constants at runtime


def _pretty_feature(code, score_labels):
    if code.endswith("_slope"):
        return f"{score_labels.get(code[:-6], code[:-6])} (slope)"
    if code.endswith("_intercept"):
        return f"{score_labels.get(code[:-10], code[:-10])} (intercept)"
    return score_labels.get(code, code)


@st.cache_resource
def _build_dice(model_path_key, train_df_csv_hash, _train_df, _model):
    """DiCE-Wrapper bauen. Wird einmal pro (model, training data) gecached."""
    if not HAS_DICE:
        return None
    feature_cols = [c for c in _train_df.columns if c != "target"]
    data = dice_ml.Data(
        dataframe=_train_df,
        continuous_features=feature_cols,
        outcome_name="target",
    )
    dice_model = dice_ml.Model(model=_model, backend="sklearn",
                                model_type="classifier")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp = dice_ml.Dice(data, dice_model, method="kdtree")
    return exp, feature_cols


def find_counterfactuals(model, X_train, y_train, query_df, n_cfs=3,
                          score_labels=None, model_key="default"):
    """Findet n_cfs Counterfactuals fuer den ersten Patient in query_df.
    Returns DataFrame mit den geaenderten Feature-Werten und einer 'change'-Spalte
    pro Counterfactual."""
    if not HAS_DICE:
        return None, "DiCE library not installed"
    if score_labels is None:
        score_labels = {}

    train_df = X_train.copy()
    train_df["target"] = np.asarray(y_train).astype(int)

    try:
        result = _build_dice(model_key, hash(str(X_train.values.tobytes())),
                              train_df, model)
        if result is None:
            return None, "DiCE wrapper could not be built"
        exp, feature_cols = result
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfs = exp.generate_counterfactuals(
                query_df[feature_cols], total_CFs=n_cfs,
                desired_class="opposite", verbose=False,
            )
        cf_df = cfs.cf_examples_list[0].final_cfs_df_sparse
        if cf_df is None or len(cf_df) == 0:
            return None, "No counterfactuals found"
        original = query_df[feature_cols].iloc[0]
        rows = []
        for i, (_, cf) in enumerate(cf_df.iterrows()):
            for feat in feature_cols:
                if feat not in cf.index:
                    continue
                orig_val = float(original[feat])
                new_val = float(cf[feat])
                if abs(new_val - orig_val) < 1e-6:
                    continue
                rows.append({
                    "Counterfactual": f"CF {i + 1}",
                    "Feature": _pretty_feature(feat, score_labels),
                    "Patient value": orig_val,
                    "Counterfactual value": new_val,
                    "Delta": new_val - orig_val,
                })
        if not rows:
            return None, "No feature changes needed (already at boundary)"
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
