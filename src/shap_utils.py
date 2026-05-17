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
    if not hasattr(pipeline, "named_steps"):
        return X.values if hasattr(X, "values") else X, pipeline
    Xp = X.values if hasattr(X, "values") else X
    clf = None
    for name, step in pipeline.named_steps.items():
        if name == "clf":
            clf = step
            break
        Xp = step.transform(Xp)
    return Xp, clf


@st.cache_data(show_spinner=False, hash_funcs={object: id})
def _get_explanation(_model, X, _model_key):
    """Berechne SHAP-Werte. _model_key dient nur dem Cache-Key, _model wird per id() gehasht."""
    pipeline = _unwrap_pipeline(_model)
    Xp, clf = _preprocess(pipeline, X)
    if clf is None:
        return None

    clf_name = clf.__class__.__name__
    if "XGB" in clf_name or "Forest" in clf_name or "Tree" in clf_name or "Boost" in clf_name:
        explainer = shap.TreeExplainer(clf)
    else:
        # Fallback: KernelExplainer auf einem kleinen Subsample des Inputs
        explainer = shap.LinearExplainer(clf, Xp) if hasattr(clf, "coef_") \
            else shap.KernelExplainer(clf.predict_proba, Xp[: min(20, len(Xp))])

    sv = explainer(Xp) if callable(explainer) else explainer.shap_values(Xp)
    # Bei RF kommt manchmal eine Liste oder ein 3D-Array (binary). Wir wollen Klasse 1.
    if isinstance(sv, list):
        sv = shap.Explanation(values=sv[1], base_values=np.full(len(Xp), explainer.expected_value[1])
                              if isinstance(explainer.expected_value, (list, np.ndarray))
                              else np.full(len(Xp), explainer.expected_value),
                              data=Xp)
    if hasattr(sv, "values") and sv.values.ndim == 3:
        # (n_samples, n_features, n_classes), Klasse 1
        sv = sv[..., 1]

    sv.feature_names = [_pretty(c) for c in X.columns]
    return sv


def get_shap(model, X, model_key):
    """Public API. Returns shap.Explanation oder None bei Fehler."""
    try:
        return _get_explanation(model, X, model_key)
    except Exception as e:
        st.caption(f"SHAP nicht verfuegbar fuer {model_key}: {type(e).__name__}: {e}")
        return None
