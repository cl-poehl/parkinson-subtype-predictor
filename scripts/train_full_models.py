"""Trainiert Modelle auf der KOMPLETTEN PPMI-Kohorte (kein CV) und
speichert sie fuer per-Patient-Inferenz:

1. Cox-PH-Modell auf Time-to-HY-3 mit den 34 slope+intercept-Features
   -> models/cox_survival.joblib
2. UPDRS3-only LogReg (slope + intercept) -> models/baseline_updrs3_only.joblib
3. MoCA-only LogReg (slope + intercept) -> models/baseline_moca_only.joblib

Damit kann jede neue Patient-Feature-Reihe sofort eine Survival- und
Baseline-Vorhersage bekommen.
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
from lifelines import CoxPHFitter
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import SCORES_LUXPARK
from src.features import extract_slope_intercept


HY_THRESHOLD = 3.0


def derive_time_to_event(df):
    df_pat = []
    for patno, grp in df.sort_values("disease_duration").groupby("patno"):
        hy = grp["HY_on"].fillna(grp["HY_off"])
        if hy.notna().sum() == 0:
            continue
        t = grp["disease_duration"].values
        t0 = t[0]
        hy_vals = hy.values
        event_idx = np.where(hy_vals >= HY_THRESHOLD)[0]
        if event_idx.size > 0:
            ev = 1
            t_event = t[event_idx[0]] - t0
        else:
            ev = 0
            t_event = t[-1] - t0
        if t_event <= 0:
            t_event = max(t_event, 0.1)
        df_pat.append({"patno": patno, "time": t_event, "event": ev})
    return pd.DataFrame(df_pat)


def main():
    out_dir = os.path.join(ROOT, "models")
    os.makedirs(out_dir, exist_ok=True)

    data = load_data()
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])

    # Subtype-Label fuer Baselines
    subtype = df.groupby("patno")["Subtype"].first()
    y_full = (subtype == 1).astype(int)
    feats = extract_slope_intercept(df, SCORES_LUXPARK)
    common = feats.index.intersection(y_full.index)
    X_all = feats.loc[common]
    y_all = y_full.loc[common].values

    from sklearn.metrics import roc_auc_score

    # ---- Baselines: UPDRS3-only und MoCA-only LogReg
    for score, fname in (("UPDRS3_on", "baseline_updrs3_only.joblib"),
                          ("MOCA", "baseline_moca_only.joblib")):
        cols = [f"{score}_slope", f"{score}_intercept"]
        Xb = X_all[cols].copy()
        pipe = Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=42)),
        ])
        pipe.fit(Xb.values, y_all)
        joblib.dump({"pipeline": pipe, "features": cols},
                     os.path.join(out_dir, fname))
        auc = roc_auc_score(y_all, pipe.predict_proba(Xb.values)[:, 1])
        print(f"Saved baseline -> {fname}, train AUC {auc:.3f}")

    # ---- Cox-PH auf Time-to-HY-3
    survival = derive_time_to_event(df)
    surv_indexed = survival.set_index("patno")
    common2 = list(X_all.index.intersection(surv_indexed.index))
    if not common2:
        raise RuntimeError(f"No overlap between features and survival; "
                            f"X_all idx sample: {list(X_all.index)[:3]}, "
                            f"surv idx sample: {list(surv_indexed.index)[:3]}")
    X_cox = X_all.loc[common2].copy()
    median_imp = X_cox.median()
    X_imp = X_cox.fillna(median_imp)
    surv_aligned = surv_indexed.loc[common2]
    df_cox = surv_aligned.join(X_imp).dropna()
    print(f"Cox training rows: {len(df_cox)}, events: "
           f"{int(df_cox['event'].sum())}")
    if len(df_cox) < 10:
        raise RuntimeError("Too few rows for Cox training")
    cph = CoxPHFitter(penalizer=0.05)
    cph.fit(df_cox.reset_index(drop=True),
             duration_col="time", event_col="event")
    joblib.dump({"cox": cph, "features": list(X_imp.columns),
                  "median_imp": median_imp.to_dict()},
                 os.path.join(out_dir, "cox_survival.joblib"))
    print(f"Saved Cox -> cox_survival.joblib, c-index "
           f"{cph.concordance_index_:.3f}")


if __name__ == "__main__":
    main()
