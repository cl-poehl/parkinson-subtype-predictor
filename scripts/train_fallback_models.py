"""Fallback models for the missing-core regime (see paper 3.4.2).

The leave-core-out analysis showed that when two or more MDS-UPDRS core parts
(I, II, III) are missing, the full-17 model imputing them is significantly
beaten by a model trained natively on the scores that remain. This script
pre-trains that small fallback library, one model per *core-presence* pattern
so the deployed app can route a patient to the model trained for exactly the
core scores they have:

  coreI    -- only MDS-UPDRS I present  (II and III missing)
  coreII   -- only MDS-UPDRS II present (I and III missing)
  coreIII  -- only MDS-UPDRS III present (I and II missing)
  coreNone -- no core part present      (all of I, II, III missing)

Each fallback is trained natively on {available core subset + all peripheral
scores}; its own kNN imputer still fills any missing *peripheral* score. Built
for both score sets (luxpark Standard, full Extended), both feature types
(slope, baseline) and all three classifiers, with the same isotonic calibration
+ split-conformal recipe as the deployed models (train_models.py), so routing
preserves calibrated probabilities and 90% prediction sets.

Output per pattern: <clf>_<set>_<slope|baseline>_<pattern>.joblib (+ _conformal).
"""
import os
import sys
import warnings

import joblib

PPMI_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PPMI_REPO)

from data_loading import load_data
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from mapie.classification import SplitConformalClassifier

from src.constants import SCORE_LABELS, SCORES_LUXPARK
from src.features import extract_slope_intercept, extract_baseline

OUT_DIR = os.path.join(PPMI_REPO, "models")

SCORE_SETS = {
    "luxpark": list(SCORES_LUXPARK),
    "full": list(SCORE_LABELS.keys()),
}

# Each pattern drops the core scores that are MISSING (so the kept set is the
# native feature set). MDS-UPDRS III spans both medication states.
DROP = {
    "coreI":    ["UPDRS2", "UPDRS3_off", "UPDRS3_on"],
    "coreII":   ["UPDRS1", "UPDRS3_off", "UPDRS3_on"],
    "coreIII":  ["UPDRS1", "UPDRS2"],
    "coreNone": ["UPDRS1", "UPDRS2", "UPDRS3_off", "UPDRS3_on"],
}

BASE_CLFS = {
    "rf": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                          class_weight="balanced", random_state=42,
                                          n_jobs=-1),
    "xgb": lambda: XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=42, n_jobs=-1),
    "logreg": lambda: LogisticRegression(max_iter=5000, class_weight="balanced",
                                          random_state=42, solver="saga",
                                          penalty="l1"),
}


def make_pipe(clf):
    return Pipeline([("imputer", KNNImputer(n_neighbors=5)),
                     ("scaler", StandardScaler()), ("clf", clf)])


def main():
    print("Loading PPMI data ...", flush=True)
    data = load_data().rename(columns={"PATNO": "patno",
                                        "Disease_duration": "disease_duration"})
    labels = data.groupby("patno")["Subtype"].first()
    y_full = (labels == 1).astype(int)  # 1 = Fast -> positive class

    n_done = 0
    for set_name, scores in SCORE_SETS.items():
        for pattern, drop in DROP.items():
            kept = [s for s in scores if s not in drop]
            cols = ["patno", "disease_duration"] + kept
            X_slope = extract_slope_intercept(data[cols], kept)
            X_base = extract_baseline(data[cols], kept)
            print(f"\n=== {set_name} / {pattern} "
                  f"({len(kept)} scores, {X_slope.shape[1]} slope feats) ===",
                  flush=True)
            for short, factory in BASE_CLFS.items():
                for suffix, X in [("slope", X_slope), ("baseline", X_base)]:
                    y_ = y_full.loc[X.index]
                    X_train, X_calib, y_train, y_calib = train_test_split(
                        X.values, y_.values, test_size=0.2, random_state=42,
                        stratify=y_.values)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cal = CalibratedClassifierCV(make_pipe(factory()),
                                                      method="isotonic", cv=5)
                        cal.fit(X_train, y_train)
                        scc = SplitConformalClassifier(
                            estimator=cal, confidence_level=0.9,
                            conformity_score="lac", prefit=True, random_state=42)
                        scc.conformalize(X_calib, y_calib)
                    stem = f"{short}_{set_name}_{suffix}_{pattern}"
                    joblib.dump(cal, os.path.join(OUT_DIR, f"{stem}.joblib"))
                    joblib.dump(scc, os.path.join(OUT_DIR, f"{stem}_conformal.joblib"))
                    n_done += 1
                    print(f"  -> {stem} (+conformal)  [{n_done}]", flush=True)
    print(f"\nDone. {n_done} fallback models + {n_done} conformal.", flush=True)


if __name__ == "__main__":
    main()
