"""Trainiert die deployten 3 Klassifikatoren (RF, XGBoost, LogReg) zusaetzlich
mit alternativen Imputern (median, mice) - parallel zu den existierenden
kNN-Modellen. Damit kann die Webapp im Sidebar einen Imputer-Selector
anbieten und zeigen wie sich Predictions zwischen den Imputern unterscheiden.

Ausgangspunkt: scripts/train_models.py mit hardcoded KNNImputer.
Hier dasselbe Setup, aber mit:
- SimpleImputer(strategy='median')
- IterativeImputer (MICE)

Pro Variante 3 Modelle (slope+intercepts) * 2 Score-Sets (luxpark, full)
* 1 Modelltyp = 6 Modelle. Plus Conformal-Wrapper.

Output: models/{rf|xgb|logreg}_{luxpark|full}_slope_{median|mice}.joblib
plus _conformal.joblib.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import (
    SCORE_LABELS, SCORES_LUXPARK, MODEL_FILES_LUXPARK, MODEL_FILES_FULL,
)
from src.features import extract_slope_intercept

try:
    from mapie.classification import SplitConformalClassifier
    HAS_MAPIE = True
except ImportError:
    HAS_MAPIE = False


def make_imputer(name):
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "mice":
        return IterativeImputer(max_iter=10, random_state=42,
                                  sample_posterior=False)
    if name == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(name)


def make_classifier(name, n_pos, n_neg):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
    if name == "xgb":
        return XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss", random_state=42, n_jobs=-1)
    if name == "logreg":
        return LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="saga", penalty="l1", random_state=42)
    raise ValueError(name)


def train_one(X, y, classifier_name, imputer_name):
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pipe = Pipeline([
        ("imp", make_imputer(imputer_name)),
        ("sc", StandardScaler()),
        ("clf", make_classifier(classifier_name, n_pos, n_neg)),
    ])
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    cal = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
    cal.fit(X_tr, y_tr)
    conformal = None
    if HAS_MAPIE:
        try:
            conformal = SplitConformalClassifier(
                cal, prefit=True, conformity_score="lac",
                confidence_level=0.9)
            conformal.conformalize(X_cal, y_cal)
        except Exception as e:
            print(f"  conformal failed: {e}")
            conformal = None
    return cal, conformal


def main():
    data = load_data()
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])
    subtype = df.groupby("patno")["Subtype"].first()
    y_full = (subtype == 1).astype(int)

    score_sets = {
        "luxpark": SCORES_LUXPARK,
        "full": list(SCORE_LABELS.keys()),
    }
    imputers = ("median", "mice")
    classifiers = ("rf", "xgb", "logreg")

    for set_name, scores in score_sets.items():
        feats = extract_slope_intercept(df, scores)
        common = feats.index.intersection(y_full.index)
        X = feats.loc[common].values
        y = y_full.loc[common].values
        print(f"\n=== {set_name} (n={len(common)}, {X.shape[1]} features) ===")
        for imp_name in imputers:
            for clf_name in classifiers:
                fname_clf = {"rf": "rf", "xgb": "xgb",
                              "logreg": "logreg"}[clf_name]
                base_path = f"models/{fname_clf}_{set_name}_slope.joblib"
                out_path = base_path.replace(".joblib", f"_{imp_name}.joblib")
                conf_path = out_path.replace(".joblib", "_conformal.joblib")
                full_out = os.path.join(ROOT, out_path)
                full_conf = os.path.join(ROOT, conf_path)
                print(f"  {clf_name} + {imp_name} ...", end=" ")
                cal, conformal = train_one(X, y, clf_name, imp_name)
                joblib.dump(cal, full_out)
                if conformal is not None:
                    joblib.dump(conformal, full_conf)
                print("ok")
    print("\nDone.")


if __name__ == "__main__":
    main()
