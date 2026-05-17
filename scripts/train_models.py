"""Trainiert 12 Modelle: 3 Klassifikatoren x 2 Score-Sets x 2 Modelltypen.

Score-Sets:
- 'luxpark' (17 Scores, LuxPARK-kompatibel) -> Suffix _luxpark_
- 'full' (25 PPMI-Scores)                   -> Suffix _full_

Beide Varianten via CalibratedClassifierCV (isotonic, 5-fold CV) kalibriert."""
import os
import sys
import joblib
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)

from data_loading import load_data
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import SCORE_LABELS, SCORES_LUXPARK
from src.features import extract_slope_intercept, extract_baseline

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_SETS = {
    "luxpark": list(SCORES_LUXPARK),
    "full": list(SCORE_LABELS.keys()),
}


def make_pipe(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
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
y = (labels == 1).astype(int)

for set_name, scores in SCORE_SETS.items():
    print(f"\n=== Score-Set '{set_name}' ({len(scores)} Scores) ===")
    print("Slope-Features ...")
    X_slope = extract_slope_intercept(data[["patno", "disease_duration"] + scores], scores)
    y_slope = y.loc[X_slope.index]

    print("Baseline-Features ...")
    X_base = extract_baseline(data[["patno", "disease_duration"] + scores], scores)
    y_base = y.loc[X_base.index]

    for short, factory in base_clfs.items():
        for suffix, X, y_ in [("slope", X_slope, y_slope), ("baseline", X_base, y_base)]:
            print(f"  Training {short}_{set_name}_{suffix} (n={len(X)}) ...")
            pipe = make_pipe(factory())
            cal = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
            cal.fit(X.values, y_.values)
            path = os.path.join(OUT_DIR, f"{short}_{set_name}_{suffix}.joblib")
            joblib.dump(cal, path)
            print(f"    -> {path}")

print("\nFertig.")
