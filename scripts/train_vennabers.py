"""Fit Venn-Abers calibrators for every deployed model (primary + fallback),
parallel to the split-conformal artefacts.

Each calibrator is fit on the *same* 20% held-out calibration split the conformal
predictor uses (reproducible: random_state=42, stratified), using the deployed
calibrated model's scores on that holdout. One calibration set therefore powers
both the conformal prediction set and the Venn-Abers probability interval.

Output per model: <stem>_vennabers.joblib  ({cal_scores, cal_labels}).
"""
import os
import sys
import warnings

import joblib

PPMI_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PPMI_REPO)

from data_loading import load_data
from sklearn.model_selection import train_test_split

from src import vennabers
from src.constants import SCORE_LABELS, SCORES_LUXPARK, CORE_DROP, fallback_scores
from src.features import extract_slope_intercept, extract_baseline

OUT_DIR = os.path.join(PPMI_REPO, "models")
SCORE_SETS = {"luxpark": list(SCORES_LUXPARK), "full": list(SCORE_LABELS.keys())}
SHORT = {"rf": "Random Forest", "xgb": "XGBoost", "logreg": "Logistic Regression"}


def _fit_one(stem, X, y):
    model_path = os.path.join(OUT_DIR, f"{stem}.joblib")
    if not os.path.exists(model_path):
        return False
    X_train, X_calib, y_train, y_calib = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y.values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
        cal_scores = model.predict_proba(X_calib)[:, 1]
    state = vennabers.fit(cal_scores, y_calib)
    joblib.dump(state, os.path.join(OUT_DIR, f"{stem}_vennabers.joblib"))
    return True


def main():
    data = load_data().rename(columns={"PATNO": "patno",
                                        "Disease_duration": "disease_duration"})
    y_full = (data.groupby("patno")["Subtype"].first() == 1).astype(int)
    n = 0

    def features(scores):
        cols = ["patno", "disease_duration"] + scores
        return {"slope": extract_slope_intercept(data[cols], scores),
                "baseline": extract_baseline(data[cols], scores)}

    for set_name, scores in SCORE_SETS.items():
        # primary (full score set)
        feats = features(scores)
        for short in SHORT:
            for suffix, X in feats.items():
                y = y_full.loc[X.index]
                if _fit_one(f"{short}_{set_name}_{suffix}", X, y):
                    n += 1
        # fallbacks (one per core-presence pattern)
        for pattern in CORE_DROP:
            kept = fallback_scores(scores, pattern)
            feats = features(kept)
            for short in SHORT:
                for suffix, X in feats.items():
                    y = y_full.loc[X.index]
                    if _fit_one(f"{short}_{set_name}_{suffix}_{pattern}", X, y):
                        n += 1
        print(f"  {set_name}: done", flush=True)
    print(f"Fertig. {n} Venn-Abers-Kalibratoren.", flush=True)


if __name__ == "__main__":
    main()
