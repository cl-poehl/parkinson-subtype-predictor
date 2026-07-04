"""ML classifiers for subtype prediction with StratifiedGroupKFold CV.

StratifiedGroupKFold keeps the 4.5:1 slow:fast class balance constant per fold
despite patient grouping -- important for stable AUC estimates given the small
fast set (n=74)."""
import warnings
# sklearn 1.8 marks LogisticRegression(penalty=...) as deprecated in favor of
# l1_ratio. Semantics unchanged, so filter the warning locally so the notebooks
# run without red-box outputs.
for cat in (FutureWarning, UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=cat,
                              module=r"sklearn\.linear_model\._logistic")
    warnings.filterwarnings("ignore", category=cat,
                              message=r".*penalty.*")
    warnings.filterwarnings("ignore", category=cat,
                              message=r".*Inconsistent values.*")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # plain tqdm avoids IProgressWarning on headless runs
from xgboost import XGBClassifier

from constants import SCORE_LABELS, SUBTYPE_FAST
from feature_extraction import extract_features, get_labels
from likelihood import (
    filter_missings_by_cutoff,
    introduce_missingness_by_cutoff,
    filter_follow_up_by_cutoff,
    shorten_follow_up_to_cutoff,
)

ML_MODEL_TYPES = {
    "slopes": "Slopes",
    "slopes+intercepts": "Slopes + Intercepts",
}

CLASSIFIERS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "xgboost": lambda: XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    ),
    # sklearn 1.8 deprecates the `penalty` kwarg in favor of `l1_ratio`. We
    # keep the semantically clear `penalty="l1"` form and suppress the
    # FutureWarning via a module filter (logic identical).
    "logistic_regression": lambda: LogisticRegression(
        max_iter=5000, class_weight="balanced",
        random_state=42, solver="saga", penalty="l1",
    ),
}

CLASSIFIER_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}

IMPUTERS = {
    "median": lambda: SimpleImputer(strategy="median"),
    "mean": lambda: SimpleImputer(strategy="mean"),
    "knn": lambda: KNNImputer(n_neighbors=5),
    "mice": lambda: IterativeImputer(max_iter=10, random_state=42, sample_posterior=False),
    # With add_indicator: adds binary missing-flag features per score,
    # so the model can distinguish whether a value was imputed or measured
    "median+indicator": lambda: SimpleImputer(strategy="median", add_indicator=True),
    "knn+indicator": lambda: KNNImputer(n_neighbors=5, add_indicator=True),
}


def evaluate_cv(
    data, model_type, classifier_name,
    patno_col="PATNO", subtype_col="Subtype", time_col="Disease_duration",
    positive_subtype=SUBTYPE_FAST,
    filter_missings_cutoff=1, introduce_missingness=0,
    filter_follow_up=0, shorten_follow_up=np.inf,
    folds=10, scores=None, progress_bar=False,
    imputer="knn", calibrate=False,
):
    # calibrate=True wraps the per-fold pipeline in
    # CalibratedClassifierCV(isotonic, cv=5) -- the deployed configuration.
    # Discrimination (AUC) is unchanged because isotonic is monotonic, but the
    # resulting y_prob is calibrated. Use this when the OUTPUT probabilities are
    # consumed (calibration table, conformal coverage, decision thresholds),
    # not just AUC ranking. See run_calibration.py.
    if scores is None:
        scores = list(SCORE_LABELS.keys())

    # StratifiedGroupKFold: keeps the class balance per fold despite patient grouping.
    # With 74 fast / 335 slow across 10 folds this is key for a stable AUC estimate.
    # We need to extract the class label per patient (StratifiedGroupKFold
    # expects a y vector and a groups vector of the same length).
    gkf = StratifiedGroupKFold(n_splits=folds, random_state=0, shuffle=True)
    all_preds = []
    imp_sums = None
    imp_count = 0

    # StratifiedGroupKFold needs y per row: we map each patient's subtype
    # onto each of their visit rows.
    y_per_row = (data[subtype_col] == positive_subtype).astype(int).values
    for k, (train_idx, test_idx) in enumerate(
        tqdm(gkf.split(data, y=y_per_row, groups=data[patno_col]),
              total=folds, disable=not progress_bar,
              desc=f"CV {classifier_name} ({model_type})")
    ):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        # scenario filters on test set only
        if filter_follow_up > 0:
            test_df = filter_follow_up_by_cutoff(data=test_df, patno_col=patno_col,
                                                  time_col=time_col, cutoff=filter_follow_up)
        if shorten_follow_up < np.inf:
            test_df = shorten_follow_up_to_cutoff(data=test_df, timepoint_col=time_col,
                                                   patno_col=patno_col, cutoff=shorten_follow_up)
        if filter_missings_cutoff < 1:
            test_df = filter_missings_by_cutoff(data=test_df, patno_col=patno_col,
                                                 scores=scores, cutoff=filter_missings_cutoff)
        if introduce_missingness > 0:
            test_df = introduce_missingness_by_cutoff(data=test_df, patno_col=patno_col,
                                                      scores=scores, cutoff=introduce_missingness)

        if test_df[patno_col].nunique() < 2:
            continue

        X_train = extract_features(train_df, model_type, scores, patno_col, time_col)
        y_train = get_labels(train_df, patno_col, subtype_col).loc[X_train.index]
        X_test = extract_features(test_df, model_type, scores, patno_col, time_col)
        y_test = get_labels(test_df, patno_col, subtype_col).loc[X_test.index]

        y_train_bin = (y_train == positive_subtype).astype(int)
        y_test_bin = (y_test == positive_subtype).astype(int)

        if y_test_bin.nunique() < 2:
            continue

        clf = CLASSIFIERS[classifier_name]()
        pipe = Pipeline([
            ("imputer", IMPUTERS[imputer]()),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

        # balance XGBoost per fold
        if isinstance(clf, XGBClassifier):
            n_pos = np.sum(y_train_bin.values == 1)
            if n_pos > 0:
                clf.set_params(scale_pos_weight=(len(y_train_bin) - n_pos) / n_pos)

        if calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            estimator = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
        else:
            estimator = pipe

        estimator.fit(X_train.values, y_train_bin.values)
        y_prob = estimator.predict_proba(X_test.values)[:, 1]

        all_preds.append(pd.DataFrame({
            patno_col: X_test.index, "y_true": y_test_bin.values,
            "y_prob": y_prob, "fold": k,
        }))

        # collect feature importances where available
        inner_clf = pipe.named_steps["clf"]
        if hasattr(inner_clf, "feature_importances_"):
            imp = inner_clf.feature_importances_
        elif hasattr(inner_clf, "coef_"):
            imp = np.abs(inner_clf.coef_[0])
        else:
            imp = None

        if imp is not None:
            if imp_sums is None:
                imp_sums = np.zeros(X_train.shape[1])
            imp_sums += imp
            imp_count += 1

    if not all_preds:
        return {"roc_auc": np.nan, "predictions": pd.DataFrame(), "feature_importances": None}

    preds = pd.concat(all_preds, ignore_index=True)
    auc = roc_auc_score(preds["y_true"], preds["y_prob"])

    feat_imp = None
    if imp_sums is not None and imp_count > 0:
        feat_imp = pd.Series(imp_sums / imp_count, index=list(X_train.columns)).sort_values(ascending=False)

    return {"roc_auc": auc, "predictions": preds, "feature_importances": feat_imp}


def bootstrap_auc_ci(predictions, n_boot=1000, ci=0.95, seed=42,
                     patno_col="PATNO", y_true_col="y_true", y_prob_col="y_prob"):
    """Bootstrap confidence interval for ROC AUC, sampled at the patient level."""
    if predictions.empty:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    patnos = predictions[patno_col].unique()
    aucs = []
    for _ in range(n_boot):
        sample = rng.choice(patnos, size=len(patnos), replace=True)
        # build resampled prediction frame (a patient may be drawn multiple times)
        rows = predictions.set_index(patno_col).loc[sample]
        y_t, y_p = rows[y_true_col].values, rows[y_prob_col].values
        if len(np.unique(y_t)) < 2:
            continue
        aucs.append(roc_auc_score(y_t, y_p))
    if not aucs:
        return (np.nan, np.nan, np.nan)
    aucs = np.array(aucs)
    alpha = (1 - ci) / 2
    return (float(np.mean(aucs)), float(np.quantile(aucs, alpha)),
            float(np.quantile(aucs, 1 - alpha)))


def evaluate_per_score_cv(data, model_type, classifier_name, scores=None, folds=10,
                          progress_bar=False, **kwargs):
    if scores is None:
        scores = list(SCORE_LABELS.keys())

    results = []
    for score in tqdm(scores, disable=not progress_bar, desc=f"Per-score {classifier_name}"):
        res = evaluate_cv(data, model_type, classifier_name, scores=[score], folds=folds, **kwargs)
        results.append({"score": score, "roc_auc": res["roc_auc"],
                         "classifier": classifier_name, "model_type": model_type})
    return pd.DataFrame(results)


def evaluate_all_models_cv(data, classifier_name, models=None, folds=10,
                           progress_bar=False, **kwargs):
    if models is None:
        models = list(ML_MODEL_TYPES.keys())

    results = []
    for mt in tqdm(models, disable=not progress_bar, desc=f"Models ({classifier_name})"):
        res = evaluate_cv(data, mt, classifier_name, folds=folds, **kwargs)
        results.append({"model_type": mt, "roc_auc": res["roc_auc"], "classifier": classifier_name})
    return pd.DataFrame(results)


def compare_classifiers_cv(data, classifier_names=None, models=None, folds=10,
                           progress_bar=False, **kwargs):
    if classifier_names is None:
        classifier_names = list(CLASSIFIERS.keys())
    if models is None:
        models = list(ML_MODEL_TYPES.keys())

    all_res = []
    for clf_name in classifier_names:
        res = evaluate_all_models_cv(data, clf_name, models, folds=folds,
                                     progress_bar=progress_bar, **kwargs)
        all_res.append(res)
    return pd.concat(all_res, ignore_index=True)
