"""SHAP-Plots fuer Erklaerbarkeit."""
import matplotlib.pyplot as plt
import numpy as np
import shap


def waterfall_for_patient(model, X, patient_idx=0, max_display=12):
    """Waterfall-Plot fuer einen einzelnen Patienten.
    model: trainierte sklearn/xgboost-Pipeline mit .predict_proba
    X: DataFrame mit den Features"""
    # versuche Pipeline, sonst direkt
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    # Falls Pipeline: Daten erst durch Imputer und Scaler laufen lassen
    if hasattr(model, "named_steps"):
        X_proc = X.copy()
        for name, step in model.named_steps.items():
            if name == "clf":
                break
            X_proc = step.transform(X_proc)
    else:
        X_proc = X.values if hasattr(X, "values") else X

    explainer = shap.TreeExplainer(clf) if hasattr(clf, "estimators_") or "xgb" in type(clf).__name__.lower() \
                else shap.LinearExplainer(clf, X_proc)
    shap_values = explainer(X_proc)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[patient_idx], max_display=max_display, show=False)
    plt.tight_layout()
    return fig
