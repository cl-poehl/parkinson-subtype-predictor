"""Speichert die PPMI-Trainings-Features als joblib fuer DiCE-Counterfactuals.

Ohne die Trainings-Referenzpunkte kann DiCE keine plausible Counterfactual-
Suche durchfuehren. Wir speichern X (Features) und y (Labels) fuer beide
Score-Sets (luxpark, full) und beide Modelltypen (slope, baseline)."""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)

from data_loading import load_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import SCORE_LABELS, SCORES_LUXPARK
from src.features import extract_slope_intercept, extract_baseline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

SCORE_SETS = {"luxpark": list(SCORES_LUXPARK), "full": list(SCORE_LABELS.keys())}

print("Loading PPMI ...")
data = load_data()
data = data.rename(columns={"PATNO": "patno", "Disease_duration": "disease_duration"})
labels = data.groupby("patno")["Subtype"].first()
y_full = (labels == 1).astype(int)

for set_name, scores in SCORE_SETS.items():
    for kind, extract in [("slope", extract_slope_intercept), ("baseline", extract_baseline)]:
        X = extract(data[["patno", "disease_duration"] + scores], scores)
        y = y_full.loc[X.index]
        # DiCE braucht NaN-freie Daten fuer den KD-Tree.
        # Wir imputieren mit demselben kNN das auch im Modell genutzt wird.
        imp = KNNImputer(n_neighbors=5)
        X_imputed_arr = imp.fit_transform(X.values)
        X_imputed = pd.DataFrame(X_imputed_arr, index=X.index, columns=X.columns)
        path = os.path.join(DATA_DIR, f"training_features_{set_name}_{kind}.joblib")
        joblib.dump({"X": X_imputed, "y": y.values}, path)
        print(f"  -> {path} ({len(X_imputed)} rows, {X_imputed.shape[1]} features, imputed)")
