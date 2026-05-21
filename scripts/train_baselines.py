"""Trainiert drei einfache Baseline-Modelle als Benchmark zu den vollen
Klassifikatoren:

1. Constant 'Slow' -- Trivial-Baseline. Sagt alle Patienten als Slow
   voraus, AUC undefiniert (konstante Prediction), aber Accuracy klar
   (entspricht der Slow-Praevalenz).
2. UPDRS3-slope-only LogReg -- slope+intercept des UPDRS3 als
   einziges Feature.
3. MoCA-slope-only LogReg -- slope+intercept des MoCA.

Alle 10-fold CV grouped by patient (GroupKFold), kNN-Imputation pro
Score, StandardScaler. Output: data/baseline_predictions.csv mit
Spalten (model, patno, y_true, y_prob).

Diese Datei wird im About-Tab von der Webapp gelesen und neben den
Headline-Klassifikatoren mit Bootstrap-CIs gezeigt.
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
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import extract_slope_intercept


def visits_with_subtype(data):
    """Mappt das SubtypePredictions-Schema (PATNO, Disease_duration,
    Subtype + Scores) auf das webapp-Schema (patno, disease_duration,
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
        # 'Slow' = y_prob fuer Fast = 0. Niedriger Score == Slow.
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

    # Quick AUC-Check
    from sklearn.metrics import roc_auc_score
    for m, g in out.groupby("model"):
        if g["y_prob"].nunique() > 1:
            auc = roc_auc_score(g["y_true"], g["y_prob"])
            print(f"  {m:20s} AUC={auc:.3f} n={len(g)}")
        else:
            acc = (g["y_true"] == 0).mean()  # alle als Slow vorhergesagt
            print(f"  {m:20s} constant prediction, accuracy={acc:.3f} (slow rate)")


if __name__ == "__main__":
    main()
