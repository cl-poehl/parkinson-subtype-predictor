"""Computes partial-dependence and ICE data per classifier for the most
important features. Output: data/pdp_data.csv.

Format: classifier, feature, x, mean_pdp, ice_<patno>...
We store the data in long format (classifier, feature, x, kind,
patno, prediction) so that Altair can filter it easily.

kind in {'pdp', 'ice'}: 'pdp' is the averaged curve, 'ice' are the
individual per-patient curves. Per feature, 30 ICE lines (random
sample) are stored to keep the file small.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(DATA_DIR, "pdp_data.csv")

TOP_FEATURES = [
    "UPDRS2_slope", "UPDRS3_on_slope", "MOCA_slope", "SCOPA_slope",
    "UPDRS1_slope", "PIGD_on_slope",
]

CLASSIFIERS = [
    ("Random Forest", "rf_luxpark_slope.joblib"),
    ("XGBoost", "xgb_luxpark_slope.joblib"),
    ("Logistic Regression", "logreg_luxpark_slope.joblib"),
]


def main():
    feats = joblib.load(os.path.join(DATA_DIR, "training_features_luxpark_slope.joblib"))
    X = feats["X"]
    y = feats["y"]
    # X may contain NaNs if patients have missing features; PDP operates on
    # the training set after imputation. We pass X directly to the
    # CalibratedClassifierCV model (pipeline with KNNImputer inside).
    rng = np.random.default_rng(42)
    ice_sample = rng.choice(len(X), size=min(30, len(X)), replace=False)

    rows = []
    for label, fname in CLASSIFIERS:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            print(f"Skip {label}: {fname} not found")
            continue
        model = joblib.load(path)
        for feat in TOP_FEATURES:
            if feat not in X.columns:
                continue
            vals = X[feat].dropna().values
            grid = np.linspace(np.quantile(vals, 0.02), np.quantile(vals, 0.98), 25)

            # PDP: each grid point yields a mean prediction over
            # all training patients (with the feature value replaced)
            for x in grid:
                Xp = X.copy()
                Xp[feat] = x
                # CalibratedClassifierCV.predict_proba needs the column order
                proba = model.predict_proba(Xp)[:, 1]
                rows.append({
                    "classifier": label,
                    "feature": feat,
                    "x": float(x),
                    "kind": "pdp",
                    "patno_idx": -1,
                    "prediction": float(proba.mean()),
                })
                # ICE lines: individual patient predictions
                for idx in ice_sample:
                    rows.append({
                        "classifier": label,
                        "feature": feat,
                        "x": float(x),
                        "kind": "ice",
                        "patno_idx": int(idx),
                        "prediction": float(proba[idx]),
                    })
        print(f"  {label} done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} rows -> {OUT}")
    print(out.groupby(["classifier", "feature", "kind"]).size().head(15))


if __name__ == "__main__":
    main()
