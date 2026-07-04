"""Trains three simple baseline models as a benchmark against the full
classifiers:

1. Constant 'Slow' -- trivial baseline. Predicts all patients as Slow,
   AUC undefined (constant prediction), but accuracy is clear
   (equals the Slow prevalence).
2. UPDRS3-slope-only LogReg -- slope+intercept of UPDRS3 as the
   only feature.
3. MoCA-slope-only LogReg -- slope+intercept of MoCA.

All with 10-fold CV grouped by patient (GroupKFold), kNN imputation per
score, StandardScaler. Output: data/baseline_predictions.csv with
columns (model, patno, y_true, y_prob).

This file is read by the About tab of the webapp and shown alongside the
headline classifiers with bootstrap CIs.
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

PPMI_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import extract_slope_intercept


def visits_with_subtype(data):
    """Maps the SubtypePredictions schema (PATNO, Disease_duration,
    Subtype + Scores) to the webapp schema (patno, disease_duration,
    Scores)."""
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    return df


def main():
    data = load_data()
    df = visits_with_subtype(data)
    df = df.dropna(subset=["disease_duration"])
    # y: subtype 1=fast, 2=slow. y_true = (subtype == 1).
    subtype = df.groupby("patno")["Subtype"].first()
    y_true = (subtype == 1).astype(int)

    rows = []

    # 1. Constant Slow
    for patno, y in y_true.items():
        # 'Slow' = y_prob for Fast = 0. Low score == Slow.
        rows.append({"model": "constant_slow", "patno": int(patno),
                       "y_true": int(y), "y_prob": 0.0})

    # 2. UPDRS3-only LogReg, 3. MoCA-only LogReg
    for score, label in (("UPDRS3_on", "updrs3_only"),
                          ("MOCA", "moca_only")):
        feats = extract_slope_intercept(df, [score])
        common = feats.index.intersection(y_true.index)
        X = feats.loc[common].values
        y = y_true.loc[common].values
        patnos = list(common)
        gkf = StratifiedGroupKFold(n_splits=10, random_state=0, shuffle=True)
        groups = np.asarray(patnos)
        for tr, te in gkf.split(X, y, groups=groups):
            pipe = Pipeline([
                ("imp", KNNImputer(n_neighbors=5)),
                ("sc", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ])
            pipe.fit(X[tr], y[tr])
            proba = pipe.predict_proba(X[te])[:, 1]
            for i, idx in enumerate(te):
                rows.append({"model": label, "patno": int(patnos[idx]),
                              "y_true": int(y[idx]),
                              "y_prob": float(proba[i])})

    out = pd.DataFrame(rows)
    path = os.path.join(ROOT, "data", "baseline_predictions.csv")
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} rows -> {path}")
    print(out.groupby("model").size())

    # Quick AUC check
    from sklearn.metrics import roc_auc_score
    for m, g in out.groupby("model"):
        if g["y_prob"].nunique() > 1:
            auc = roc_auc_score(g["y_true"], g["y_prob"])
            print(f"  {m:20s} AUC={auc:.3f} n={len(g)}")
        else:
            acc = (g["y_true"] == 0).mean()  # all predicted as Slow
            print(f"  {m:20s} constant prediction, accuracy={acc:.3f} (slow rate)")


if __name__ == "__main__":
    main()
