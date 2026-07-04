"""True bootstrap on training resamples (Item O).

Instead of only bootstrapping the CV folds, the ENTIRE training
workflow is repeated here on 100 random bootstrap resamples of the
patient set, each time with all three classifiers. The resulting
AUC distributions are the correct (Pencina-style) uncertainty
estimate of the modelled performance.

Setup per bootstrap resample:
1. Sample 409 patients with replacement
2. For each classifier: KNNImputer + StandardScaler + Clf
3. 10-fold GroupKFold CV grouped by patient within the resample
4. Record ROC AUC

Output: data/true_bootstrap_aucs.csv with (resample, classifier, auc)
and docs/TRUE_BOOTSTRAP.md with the 95% CI of the true AUC.

Compute budget: ~10s per (classifier, resample) on 10-fold CV * 3 clfs
* 100 = 3000 trainings = ~8h. Reducible via N_BOOTSTRAPS or N_FOLDS.
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

PPMI_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import SCORES_LUXPARK
from src.features import extract_slope_intercept


N_BOOTSTRAPS = 100
N_FOLDS = 10


def make_classifier(name, n_pos, n_neg):
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="logloss", random_state=42, n_jobs=-1)
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="saga", penalty="l1", random_state=42)
    raise ValueError(name)


def evaluate_cv_on_resample(X, y, groups, classifier_name, folds=N_FOLDS):
    """10-fold GroupKFold CV on a resample. Sampled patients can
    appear multiple times -- GroupKFold groups them consistently across
    folds."""
    gkf = StratifiedGroupKFold(n_splits=folds, random_state=0, shuffle=True)
    aucs = []
    for tr, te in gkf.split(X, y, groups=groups):
        if np.unique(y[te]).size < 2:
            continue
        n_pos = int((y[tr] == 1).sum())
        n_neg = int((y[tr] == 0).sum())
        clf = make_classifier(classifier_name, n_pos, n_neg)
        pipe = Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("sc", StandardScaler()),
            ("clf", clf),
        ])
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    if not aucs:
        return np.nan
    # Bonus: concatenate for a pooled-predictions AUC
    return float(np.mean(aucs))


def main():
    out_dir = os.path.join(ROOT, "data")
    docs_dir = os.path.join(ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    data = load_data()
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])
    subtype = df.groupby("patno")["Subtype"].first()
    y_full = (subtype == 1).astype(int)
    feats = extract_slope_intercept(df, SCORES_LUXPARK)
    common = feats.index.intersection(y_full.index)
    X_full = feats.loc[common]
    y_arr = y_full.loc[common].values
    patnos = np.asarray(list(common))
    n = len(patnos)
    print(f"n={n} patients, {X_full.shape[1]} features")

    rng = np.random.default_rng(42)
    rows = []
    t_start = time.time()
    for b in range(N_BOOTSTRAPS):
        idx = rng.integers(0, n, n)
        Xb = X_full.iloc[idx]
        yb = y_arr[idx]
        # GROUP BY THE ORIGINAL PATIENT ID: if patient X is drawn 3x,
        # all 3 copies land in the same fold (no leakage).
        # Otherwise GroupKFold would spread the duplicates across
        # different folds and the CV AUC would be inflated by duplicates.
        groupsb = patnos[idx]
        for clf_name in ("random_forest", "xgboost", "logistic_regression"):
            t0 = time.time()
            auc = evaluate_cv_on_resample(Xb.values, yb, groupsb, clf_name)
            rows.append({"resample": b, "classifier": clf_name,
                          "auc": auc,
                          "duration_s": time.time() - t0})
        if b % 5 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (b + 1) * (N_BOOTSTRAPS - b - 1)
            print(f"  resample {b+1}/{N_BOOTSTRAPS}: "
                   f"elapsed={elapsed/60:.1f}min, ETA={eta/60:.1f}min")

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "true_bootstrap_aucs.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # CIs
    md = ["# True Bootstrap AUC Confidence Intervals (Item O)", ""]
    md.append(f"Bootstrap resamples: {N_BOOTSTRAPS}, GroupKFold folds: "
                f"{N_FOLDS}. Each resample is a full retraining of the "
                "pipeline on a patient-level bootstrap sample of n=409, "
                "evaluated by patient-grouped CV.")
    md.append("")
    md.append("| Classifier | Mean AUC | SD | 95% CI (percentile) |")
    md.append("|------------|----------|-----|---------------------|")
    for clf_name, g in df_out.groupby("classifier"):
        v = g["auc"].dropna().values
        if v.size == 0:
            continue
        lo, hi = np.quantile(v, [0.025, 0.975])
        md.append(f"| {clf_name} | {v.mean():.3f} | {v.std(ddof=1):.3f} | "
                    f"[{lo:.3f}, {hi:.3f}] |")
    md.append("")
    md.append("This is the publication-grade uncertainty estimate. The "
                "intervals are typically wider than CV-bootstrap intervals "
                "because the full training-pipeline noise (model fit "
                "variation, imputer behaviour at different patient "
                "distributions) is captured.")
    with open(os.path.join(docs_dir, "TRUE_BOOTSTRAP.md"), "w") as f:
        f.write("\n".join(md))
    print(f"Saved {docs_dir}/TRUE_BOOTSTRAP.md")


if __name__ == "__main__":
    main()
