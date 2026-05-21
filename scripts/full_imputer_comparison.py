"""Empirischer Vergleich aller verfuegbaren Imputations-Methoden auf PPMI
fuer den deployten Slope+Intercept Featureraum.

Imputer:
- median, mean, knn, mice (standard sklearn)
- missforest: IterativeImputer(RandomForestRegressor) -- 'missForest'-style
- knn_ind, median_ind: + Missing-Indicator
- softimpute: fancyimpute.SoftImpute (Matrix-Completion via SVD)
- native_nan: kein Imputer (nur XGBoost - lernt NaN-Splits selbst)

10-fold patient-grouped CV pro Klassifikator x Imputer.
Plus 1000-Resample Bootstrap-CI auf den OOF-Predictions.

Output: data/full_imputer_comparison.csv mit Spalten
(classifier, score_set, imputer, auc, auc_lo, auc_hi).
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import SCORE_LABELS, SCORES_LUXPARK
from src.features import extract_slope_intercept


def make_imputer(name):
    if name == "median": return SimpleImputer(strategy="median")
    if name == "mean": return SimpleImputer(strategy="mean")
    if name == "knn": return KNNImputer(n_neighbors=5)
    if name == "mice": return IterativeImputer(max_iter=10, random_state=42)
    if name == "missforest":
        return IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=50, max_depth=10,
                                              random_state=42, n_jobs=-1),
            max_iter=5, random_state=42)
    if name == "knn_ind":
        return KNNImputer(n_neighbors=5, add_indicator=True)
    if name == "median_ind":
        return SimpleImputer(strategy="median", add_indicator=True)
    if name == "softimpute":
        from fancyimpute import SoftImpute
        class _SoftWrap:
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return SoftImpute(verbose=False).fit_transform(X.copy())
            def fit_transform(self, X, y=None):
                return self.transform(X)
            def set_output(self, *a, **k):
                return self
        return _SoftWrap()
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


def evaluate_cv(X, y, groups, classifier_name, imputer_name, folds=10):
    gkf = GroupKFold(n_splits=folds)
    proba = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups=groups):
        n_pos = int((y[tr] == 1).sum())
        n_neg = int((y[tr] == 0).sum())
        clf = make_classifier(classifier_name, n_pos, n_neg)
        if imputer_name == "native_nan":
            if classifier_name != "xgb":
                return None
            clf.fit(X[tr], y[tr])
            proba[te] = clf.predict_proba(X[te])[:, 1]
        else:
            pipe = Pipeline([
                ("imp", make_imputer(imputer_name)),
                ("sc", StandardScaler()),
                ("clf", clf),
            ])
            pipe.fit(X[tr], y[tr])
            proba[te] = pipe.predict_proba(X[te])[:, 1]
    return proba


def bootstrap_auc(y, p, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, p[idx]))
    if not aucs:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(aucs)),
             float(np.quantile(aucs, 0.025)),
             float(np.quantile(aucs, 0.975)))


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
    imputers = ("median", "mean", "knn", "mice", "missforest",
                 "knn_ind", "median_ind", "softimpute", "native_nan")
    classifiers = ("rf", "xgb", "logreg")

    rows = []
    for set_name, scores in score_sets.items():
        feats = extract_slope_intercept(df, scores)
        common = feats.index.intersection(y_full.index)
        X = feats.loc[common].values
        y = y_full.loc[common].values
        groups = np.array(list(common))
        print(f"\n=== {set_name} (n={len(common)}, {X.shape[1]} features) ===")
        for imp_name in imputers:
            for clf_name in classifiers:
                t0 = time.time()
                print(f"  {clf_name} + {imp_name} ...", end=" ", flush=True)
                try:
                    proba = evaluate_cv(X, y, groups, clf_name, imp_name)
                except Exception as e:
                    print(f"FAIL: {type(e).__name__}: {e}")
                    continue
                if proba is None:
                    print("skip (not applicable)")
                    continue
                pt_auc = float(roc_auc_score(y, proba))
                m, lo, hi = bootstrap_auc(y, proba)
                rows.append({
                    "score_set": set_name,
                    "classifier": clf_name,
                    "imputer": imp_name,
                    "auc": pt_auc,
                    "auc_mean": m, "auc_lo": lo, "auc_hi": hi,
                    "duration_s": time.time() - t0,
                })
                print(f"AUC={pt_auc:.3f} [{lo:.3f}, {hi:.3f}]  "
                       f"({time.time()-t0:.1f}s)")

    out = pd.DataFrame(rows)
    out_path = os.path.join(ROOT, "data", "full_imputer_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print()
    print("Summary pivot (AUC):")
    print(out.pivot_table(index=["score_set", "classifier"], columns="imputer",
                            values="auc").round(3))


if __name__ == "__main__":
    main()
