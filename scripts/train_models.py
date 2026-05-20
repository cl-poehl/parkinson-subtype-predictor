"""Trainiert 12 Modelle (3 Klassifikatoren x 2 Score-Sets x 2 Modelltypen).

Wissenschaftliches Setup:
- kNN-Imputation (k=5) statt Median: vermeidet Klassen-Bias bei Imputation
- CalibratedClassifierCV (isotonic, 5-fold) auf 80% Training
- MapieClassifier (LAC, cv='prefit') auf 20% Holdout: Conformal-Schwellen
  fuer distribution-free Coverage-Garantien (alpha=0.10 -> 90% Coverage)

Output pro Score-Set:
- 6 Modell-Joblibs (calibrated): <clf>_<set>_<slope|baseline>.joblib
- 6 Conformal-Joblibs: <clf>_<set>_<slope|baseline>_conformal.joblib
"""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)

from data_loading import load_data
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import SCORE_LABELS, SCORES_LUXPARK
from src.features import extract_slope_intercept, extract_baseline

# MAPIE 1.4+
try:
    from mapie.classification import SplitConformalClassifier
except ImportError:
    print("MAPIE nicht installiert. pip install mapie")
    sys.exit(1)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_SETS = {
    "luxpark": list(SCORES_LUXPARK),
    "full": list(SCORE_LABELS.keys()),
}


def make_pipe(clf):
    # kNN-Imputation statt Median: vermeidet den Klassen-Bias bei der
    # 4.5:1-Imbalance der PPMI-Kohorte
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


base_clfs = {
    "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                          class_weight="balanced", random_state=42, n_jobs=-1),
    "xgb": lambda: XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42, n_jobs=-1),
    "logreg": lambda: LogisticRegression(max_iter=5000, class_weight="balanced",
                                          random_state=42, solver="saga", penalty="l1"),
}


print("PPMI-Daten laden ...")
data = load_data()
data = data.rename(columns={"PATNO": "patno", "Disease_duration": "disease_duration"})
labels = data.groupby("patno")["Subtype"].first()
# Konvention: 1 = Fast -> Klasse 1 (positiv), 2 = Slow -> Klasse 0
y_full = (labels == 1).astype(int)

for set_name, scores in SCORE_SETS.items():
    print(f"\n=== Score-Set '{set_name}' ({len(scores)} Scores) ===")
    print("Slope-Features ...")
    X_slope = extract_slope_intercept(data[["patno", "disease_duration"] + scores], scores)
    y_slope = y_full.loc[X_slope.index]

    print("Baseline-Features ...")
    X_base = extract_baseline(data[["patno", "disease_duration"] + scores], scores)
    y_base = y_full.loc[X_base.index]

    for short, factory in base_clfs.items():
        for suffix, X, y_ in [("slope", X_slope, y_slope), ("baseline", X_base, y_base)]:
            print(f"  Training {short}_{set_name}_{suffix} (n={len(X)}) ...")
            # 80% Training / 20% Conformal-Calibration
            X_train, X_calib, y_train, y_calib = train_test_split(
                X.values, y_.values, test_size=0.2,
                random_state=42, stratify=y_.values,
            )
            pipe = make_pipe(factory())
            cal = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cal.fit(X_train, y_train)
            path = os.path.join(OUT_DIR, f"{short}_{set_name}_{suffix}.joblib")
            joblib.dump(cal, path)

            # Conformal-Kalibrierung mit MAPIE 1.4+ Split-Conformal
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scc = SplitConformalClassifier(
                    estimator=cal, confidence_level=0.9,
                    conformity_score="lac", prefit=True, random_state=42,
                )
                scc.conformalize(X_calib, y_calib)
            conformal_path = os.path.join(
                OUT_DIR, f"{short}_{set_name}_{suffix}_conformal.joblib"
            )
            joblib.dump(scc, conformal_path)
            print(f"    -> {path} + conformal")

print("\nFertig.")
