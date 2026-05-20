"""Nested-CV Hyperparameter-Tuning via Optuna fuer RF, XGBoost, LogReg
auf dem 17-Score-Slopes+Intercepts-Setup.

Setup:
- Outer 5-fold CV grouped by patient -> 5 unverzerrte Test-AUCs
- Inner Tuning per Outer-Fold: Optuna TPE-Sampler, n_trials=50, 3-fold
  inner CV
- Speichert pro Klassifikator: Trial-Verlauf, Best-Params pro Outer-Fold,
  Konsistenz der gewaehlten Params (sind sie stabil?)
- Vergleich Tuned vs Defaults: Hat sich das Tunen gelohnt?

Output: docs/HYPERPARAMETER_TUNING.md + data/hyperparameter_results.csv
"""
import os
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import SCORES_LUXPARK
from src.features import extract_slope_intercept


optuna.logging.set_verbosity(optuna.logging.WARNING)


def cv_score(pipe, X, y, groups, folds=3):
    gkf = GroupKFold(n_splits=folds)
    aucs = []
    for tr, te in gkf.split(X, y, groups=groups):
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aucs))


def make_rf(params):
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params.get("max_depth", None),
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def make_xgb(params, n_pos, n_neg):
    return XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=n_neg / max(n_pos, 1),
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )


def make_logreg(params):
    return LogisticRegression(
        C=params["C"], penalty=params["penalty"], solver="saga",
        max_iter=5000, class_weight="balanced", random_state=42,
    )


def rf_search(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features",
                                                    ["sqrt", "log2", 0.5, 0.8]),
    }


def xgb_search(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def lr_search(trial):
    return {
        "C": trial.suggest_float("C", 0.001, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
    }


def main():
    out_dir = os.path.join(ROOT, "data")
    docs_dir = os.path.join(ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    data = load_data()
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])
    subtype = df.groupby("patno")["Subtype"].first()
    y_true_dict = (subtype == 1).astype(int)

    feats = extract_slope_intercept(df, SCORES_LUXPARK)
    common = feats.index.intersection(y_true_dict.index)
    X = feats.loc[common]
    y = y_true_dict.loc[common].values
    groups = np.asarray(list(common))
    print(f"n={len(X)} patients, {X.shape[1]} features")

    OUTER_FOLDS = 5
    N_TRIALS = 50
    INNER_FOLDS = 3

    outer = GroupKFold(n_splits=OUTER_FOLDS)
    all_results = []
    for clf_name in ("random_forest", "xgboost", "logistic_regression"):
        print(f"\n=== {clf_name} ===")
        outer_results = []
        for fold_i, (tr, te) in enumerate(outer.split(X, y, groups=groups)):
            t0 = time.time()
            Xtr_raw = X.iloc[tr].copy()
            ytr = y[tr]
            groups_tr = groups[tr]
            n_pos = int((ytr == 1).sum())
            n_neg = int((ytr == 0).sum())

            def objective(trial):
                if clf_name == "random_forest":
                    params = rf_search(trial)
                    clf = make_rf(params)
                elif clf_name == "xgboost":
                    params = xgb_search(trial)
                    clf = make_xgb(params, n_pos, n_neg)
                else:
                    params = lr_search(trial)
                    if params["penalty"] == "elasticnet":
                        params["l1_ratio"] = 0.5
                    clf = make_logreg({**params, **({"l1_ratio": 0.5}
                                                       if params["penalty"] == "elasticnet"
                                                       else {})})
                pipe = Pipeline([
                    ("imp", KNNImputer(n_neighbors=5)),
                    ("sc", StandardScaler()),
                    ("clf", clf),
                ])
                return cv_score(pipe, Xtr_raw.values, ytr, groups_tr,
                                  folds=INNER_FOLDS)

            study = optuna.create_study(direction="maximize",
                                          sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            best = study.best_params
            # Refit + evaluate on outer test
            if clf_name == "random_forest":
                clf = make_rf(best)
            elif clf_name == "xgboost":
                clf = make_xgb(best, n_pos, n_neg)
            else:
                params = best.copy()
                if params["penalty"] == "elasticnet":
                    params["l1_ratio"] = 0.5
                clf = make_logreg(params)
            pipe = Pipeline([
                ("imp", KNNImputer(n_neighbors=5)),
                ("sc", StandardScaler()),
                ("clf", clf),
            ])
            pipe.fit(Xtr_raw.values, ytr)
            test_proba = pipe.predict_proba(X.iloc[te].values)[:, 1]
            test_auc = roc_auc_score(y[te], test_proba)

            outer_results.append({
                "classifier": clf_name,
                "outer_fold": fold_i,
                "tuned_inner_auc": study.best_value,
                "tuned_outer_test_auc": test_auc,
                "best_params": str(best),
                "duration_s": time.time() - t0,
            })
            print(f"  fold {fold_i}: inner={study.best_value:.3f}  "
                   f"outer={test_auc:.3f}  ({time.time()-t0:.1f}s)  best={best}")
        all_results.extend(outer_results)

    df_out = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, "hyperparameter_results.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Zusammenfassung
    md_path = os.path.join(docs_dir, "HYPERPARAMETER_TUNING.md")
    lines = ["# Hyperparameter Tuning (Nested CV)", ""]
    lines.append(f"Outer folds: {OUTER_FOLDS}, inner folds: {INNER_FOLDS}, "
                  f"Optuna trials per outer fold: {N_TRIALS}.")
    lines.append("")
    lines.append("## Tuned outer-fold AUCs per classifier")
    lines.append("")
    lines.append("| Classifier | Mean tuned AUC | SD | Default AUC (from main CV) |")
    lines.append("|------------|----------------|----|----------------------------|")
    defaults = {
        "random_forest": 0.943,
        "xgboost": 0.945,
        "logistic_regression": 0.905,
    }
    for clf_name, dflt in defaults.items():
        g = df_out[df_out["classifier"] == clf_name]["tuned_outer_test_auc"]
        if not g.empty:
            mean = g.mean()
            sd = g.std(ddof=1)
            lines.append(f"| {clf_name:25s} | {mean:.3f} | {sd:.3f} | {dflt:.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If tuned AUC is not substantially higher than default AUC "
                  "(within 1 SD), the conclusion is that default "
                  "hyperparameters are already near-optimal for this cohort "
                  "size and feature set. This is the expected outcome at "
                  "n=409 with 34 features.")
    lines.append("")
    lines.append("## Best parameters per outer fold")
    lines.append("")
    for clf_name in defaults:
        lines.append(f"### {clf_name}")
        lines.append("")
        g = df_out[df_out["classifier"] == clf_name]
        for _, r in g.iterrows():
            lines.append(f"- Fold {int(r['outer_fold'])}: {r['best_params']} "
                          f"-> outer AUC {r['tuned_outer_test_auc']:.3f}")
        lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
