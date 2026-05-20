"""Temporale Validierung: trainiere auf frueh-eingeschriebenen PPMI-
Patienten (vor 2018), teste auf spaet-eingeschriebenen (2018+).

Hintergrund: PPMI 1.0 lief 2010-2018, PPMI 2.0 ab 2021. Patienten-
Selektion und Visiten-Strukturen koennen sich unterschieden haben.
Wenn unsere Modelle sauber generalisieren, sollte die Performance auf
spaet-eingeschriebenen Patienten aehnlich bleiben.

Output: data/temporal_validation.csv + docs/TEMPORAL_VALIDATION.md
"""
import os
import sys

import numpy as np
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import SCORES_LUXPARK
from src.features import extract_slope_intercept


def make_classifier(name, n_pos, n_neg):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                        class_weight="balanced", random_state=42, n_jobs=-1)
    if name == "xgboost":
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=n_neg / max(n_pos, 1),
                              eval_metric="logloss", random_state=42, n_jobs=-1)
    if name == "logistic_regression":
        return LogisticRegression(max_iter=5000, class_weight="balanced",
                                    solver="saga", penalty="l1", random_state=42)
    raise ValueError(name)


def bootstrap_auc(y, p, n=500, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(aucs)), float(np.quantile(aucs, 0.025)),
             float(np.quantile(aucs, 0.975)))


def main():
    docs_dir = os.path.join(ROOT, "docs")
    out_dir = os.path.join(ROOT, "data")
    os.makedirs(docs_dir, exist_ok=True)

    df_raw = pd.read_csv(os.path.join(PPMI_REPO, "data",
                                        "PPMI_PD_2024-03-13.csv"),
                          low_memory=False)
    subtypes = pd.read_csv(os.path.join(PPMI_REPO, "data",
                                          "ParkinsonPredict_PPMI_progression_subtypes.csv"))
    df_raw = df_raw.merge(subtypes, on="PATNO", how="inner")
    df_raw["Disease_duration"] = (
        df_raw["Age_at_BL"] - df_raw["Age_at_diagnosis"]
    ) * 12 + df_raw["Timepoint"]
    df_raw["enroll_year"] = pd.to_datetime(df_raw["ENROLL_DATE"],
                                             errors="coerce").dt.year
    df_raw["Subtype"] = pd.to_numeric(df_raw["Subtype"])

    df = df_raw.rename(columns={"PATNO": "patno",
                                  "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])
    enroll_year = df.groupby("patno")["enroll_year"].first()

    subtype = df.groupby("patno")["Subtype"].first()
    y_full = (subtype == 1).astype(int)
    feats = extract_slope_intercept(df, SCORES_LUXPARK)
    common = feats.index.intersection(y_full.index)
    X = feats.loc[common]
    y = y_full.loc[common]
    patnos = X.index

    # PPMI 2.0 (enrollment 2018+) hat keine Subtyp-Labels in unserem
    # extrahierten Set. Wir splitten daher INNERHALB der PPMI 1.0 Aera:
    # frueh-eingeschriebene (2010-2012) vs spaet-eingeschriebene (2013+).
    # Das prueft Robustheit gegen Verschiebungen INNERHALB des Studien-
    # designs (Site-Aktivierungen, Rekrutierungs-Aenderungen).
    rows = []
    for split_year in (2012, 2013, 2014):
        early_patnos = enroll_year[enroll_year < split_year].index
        late_patnos = enroll_year[enroll_year >= split_year].index
        is_early = patnos.isin(early_patnos)
        is_late = patnos.isin(late_patnos)
        Xtr = X[is_early].values
        ytr = y[is_early].values
        Xte = X[is_late].values
        yte = y[is_late].values
        n_pos = int((ytr == 1).sum())
        n_neg = int((ytr == 0).sum())
        if n_pos < 10 or n_neg < 10 or len(Xte) < 20 or np.unique(yte).size < 2:
            print(f"  skip split={split_year}: train n_pos={n_pos}, "
                   f"test n={len(Xte)}, classes={np.unique(yte).size}")
            continue
        print(f"\nSplit at {split_year}: train n={len(Xtr)} "
               f"(prev={ytr.mean():.2f}), test n={len(Xte)} "
               f"(prev={yte.mean():.2f})")
        for clf_name in ("random_forest", "xgboost", "logistic_regression"):
            clf = make_classifier(clf_name, n_pos, n_neg)
            pipe = Pipeline([
                ("imp", KNNImputer(n_neighbors=5)),
                ("sc", StandardScaler()),
                ("clf", clf),
            ])
            pipe.fit(Xtr, ytr)
            p_tr = pipe.predict_proba(Xtr)[:, 1]
            p_te = pipe.predict_proba(Xte)[:, 1]
            auc_tr, lo_tr, hi_tr = bootstrap_auc(ytr, p_tr)
            auc_te, lo_te, hi_te = bootstrap_auc(yte, p_te)
            rows.append({"split_year": split_year, "classifier": clf_name,
                          "auc_early_train": auc_tr,
                          "lo_early": lo_tr, "hi_early": hi_tr,
                          "auc_late_test": auc_te,
                          "lo_late": lo_te, "hi_late": hi_te,
                          "n_train": len(Xtr), "n_test": len(Xte)})
            print(f"  {clf_name}: train AUC={auc_tr:.3f}, late AUC={auc_te:.3f}")

    out = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "temporal_validation.csv")
    out.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    md = ["# Temporal Validation within PPMI 1.0", ""]
    md.append("**Note:** All 409 patients with subtype labels were enrolled "
                "in PPMI 1.0 (2010-2017). The subtype clustering "
                "publication used only PPMI 1.0; PPMI 2.0 patients have no "
                "subtype labels yet. Therefore we split *within* PPMI 1.0 "
                "by enrollment year to test robustness to recruitment "
                "drift (different sites activating over time, evolving "
                "inclusion criteria). True PPMI 1.0 vs PPMI 2.0 validation "
                "will require new subtype labels for the PPMI 2.0 cohort.")
    md.append("")
    md.append("## Results")
    md.append("")
    md.append("| Split year | Classifier | Train AUC [95% CI] | "
                "Late-test AUC [95% CI] | n_train / n_test |")
    md.append("|------------|------------|--------------------|"
                "----------------------|------------------|")
    for _, r in out.iterrows():
        train = f"{r['auc_early_train']:.3f} [{r['lo_early']:.3f}, {r['hi_early']:.3f}]"
        test = f"{r['auc_late_test']:.3f} [{r['lo_late']:.3f}, {r['hi_late']:.3f}]"
        md.append(f"| {int(r['split_year'])} | {r['classifier']} | "
                    f"{train} | {test} | {int(r['n_train'])} / "
                    f"{int(r['n_test'])} |")
    md.append("")
    md.append("Performance on the late-train test set approximates "
                "out-of-time generalization within the PPMI 1.0 era. A "
                "drop of more than 0.05-0.10 AUC across split years "
                "indicates temporal distribution shift in features or "
                "labels.")
    with open(os.path.join(docs_dir, "TEMPORAL_VALIDATION.md"), "w") as f:
        f.write("\n".join(md))
    print(f"Saved {docs_dir}/TEMPORAL_VALIDATION.md")


if __name__ == "__main__":
    main()
