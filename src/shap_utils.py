"""SHAP helpers fuer Einzelpatient- und Batch-Plots.
Unwrappt CalibratedClassifierCV und Pipeline und liefert eine Explanation,
die mit shap.plots.* gerendert werden kann."""
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


def _unwrap_pipeline(model):
    """CalibratedClassifierCV -> Pipeline. Erster Fold."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        inner = model.calibrated_classifiers_[0]
        if hasattr(inner, "estimator"):
            return inner.estimator
        if hasattr(inner, "base_estimator"):
            return inner.base_estimator
    return model


def _preprocess(pipeline, X):
    """X durch alle Schritte ausser dem finalen Klassifikator schicken."""
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


def _compute_explanation(_model, X, feature_cols):
    """SHAP-Werte berechnen. _model wird nicht gehasht (Streamlit-Konvention)."""
    pipeline = _unwrap_pipeline(_model)
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

    # Bei RF mit binary kann das Ergebnis (n, n_features, 2) sein, dann Klasse 1
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = sv[..., 1]

    sv.feature_names = [_pretty(c) for c in feature_cols]
    return sv


# Cache nur auf (model_key, feature_tuple, data_hash) basis, damit verschiedene
# Klassifikatoren auch wirklich verschiedene Ergebnisse liefern.
@st.cache_data(show_spinner=False)
def _cached_shap(model_key, feature_cols_tuple, data_bytes):
    # Diese Funktion wird nie direkt aufgerufen, sie dient nur als Cache-Eintrag.
    # Stattdessen rufen wir get_shap() auf, die die echte Logik kapselt.
    raise NotImplementedError("Use get_shap() instead.")


_SHAP_MEMO = {}


def get_shap(model, X, model_key):
    """Public API. Returns shap.Explanation oder None bei Fehler.
    model_key sollte unique sein pro (score_mode, classifier, model_type)."""
    feature_cols = list(X.columns)
    try:
        # Eigener Memo-Cache: model_key ist eindeutig, X identifiziert wir per
        # Shape + Hash der Werte. So vermeiden wir Streamlit-Cache-Probleme
        # mit dem unhashbaren Model.
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
