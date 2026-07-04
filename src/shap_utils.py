"""SHAP values for the full CalibratedClassifierCV ensemble.

For each classifier, the output probabilities are the mean of K
isotonically-calibrated folds. So that the SHAP attribution is mathematically
consistent with the calibrated prediction, we compute the SHAP values
separately for each fold in the ensemble and average them.

The result is a `shap.Explanation` whose values represent the ensemble's mean
feature attribution profile."""
import numpy as np
import shap
import streamlit as st

from src.constants import SCORE_LABELS


def _pretty(feat_name):
    """`UPDRS3_off_slope` -> `MDS-UPDRS III (Off) (slope)`."""
    for suffix, tag in [("_slope", "slope"), ("_intercept", "intercept")]:
        if feat_name.endswith(suffix):
            base = feat_name[: -len(suffix)]
            return f"{SCORE_LABELS.get(base, base)} ({tag})"
    return SCORE_LABELS.get(feat_name, feat_name)


def _preprocess(pipeline, X):
    """Pass X through all pipeline steps except the final classifier."""
    Xp = X.values if hasattr(X, "values") else X
    if not hasattr(pipeline, "named_steps"):
        return Xp, pipeline
    clf = None
    for name, step in pipeline.named_steps.items():
        if name == "clf":
            clf = step
            break
        Xp = step.transform(Xp)
    return Xp, clf


def _shap_for_pipeline(pipeline, X):
    """SHAP for a single pipeline. Returns (values, base_values, data) or None."""
    Xp, clf = _preprocess(pipeline, X)
    if clf is None:
        return None

    clf_name = clf.__class__.__name__
    is_tree = any(t in clf_name for t in ("XGB", "Forest", "Tree", "Boost"))
    if is_tree:
        explainer = shap.TreeExplainer(clf)
    elif hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, Xp)
    else:
        bg = Xp[: min(20, len(Xp))]
        explainer = shap.KernelExplainer(clf.predict_proba, bg)

    sv = explainer(Xp)
    # Binary classification: class 1 (= Fast)
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[..., 1]
    return sv.values, sv.base_values, sv.data


def _compute_explanation(_model, X, feature_cols):
    """Average SHAP values over ALL folds of the CalibratedClassifierCV
    ensemble, so that the attribution matches the actually output
    (ensemble) probability.

    For non-calibration models (single fit), it falls back to a single
    SHAP computation."""
    if hasattr(_model, "calibrated_classifiers_") and _model.calibrated_classifiers_:
        all_values = []
        all_base = []
        ref_data = None
        for inner in _model.calibrated_classifiers_:
            pipeline = (inner.estimator if hasattr(inner, "estimator")
                         else getattr(inner, "base_estimator", None))
            if pipeline is None:
                continue
            res = _shap_for_pipeline(pipeline, X)
            if res is None:
                continue
            v, b, d = res
            all_values.append(v)
            all_base.append(b)
            if ref_data is None:
                ref_data = d
        if not all_values:
            return None
        mean_values = np.mean(np.stack(all_values), axis=0)
        mean_base = np.mean(np.stack(all_base), axis=0)
        sv = shap.Explanation(
            values=mean_values,
            base_values=mean_base,
            data=ref_data,
            feature_names=[_pretty(c) for c in feature_cols],
        )
        return sv

    # Fallback for models without calibrated_classifiers_
    res = _shap_for_pipeline(_model, X)
    if res is None:
        return None
    v, b, d = res
    return shap.Explanation(values=v, base_values=b, data=d,
                             feature_names=[_pretty(c) for c in feature_cols])


_SHAP_MEMO = {}


def get_shap(model, X, model_key):
    """Public API. Returns shap.Explanation or None on error.
    The returned values are the fold average over the
    CalibratedClassifierCV ensemble, so that the SHAP attribution is
    consistent with the displayed ensemble probability."""
    feature_cols = list(X.columns)
    try:
        data_hash = hash((tuple(feature_cols), tuple(map(tuple, X.values.tolist()))))
        cache_key = (model_key, data_hash)
        if cache_key in _SHAP_MEMO:
            return _SHAP_MEMO[cache_key]
        sv = _compute_explanation(model, X, feature_cols)
        _SHAP_MEMO[cache_key] = sv
        return sv
    except Exception as e:
        st.warning(
            f"SHAP plot unavailable for `{model_key}`: "
            f"{type(e).__name__}: {e}",
            icon=":material/warning:",
        )
        return None
