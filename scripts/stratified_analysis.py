"""Stratified subgroup analysis (age, sex) by SLICING the full-cohort
out-of-fold predictions that produce the headline metrics -- i.e. the deployed
model's predictions restricted to each subgroup.

This is the standard way to report subgroup performance, and it fixes a flaw in
the earlier within-subgroup retraining: training a fresh 10-fold model inside
the 73-patient young-onset subset skipped folds whose test set lacked both
classes, so only ~46 of 73 young patients ever received a prediction and the
subgroup AUC was estimated on a biased remainder. Slicing the full-cohort OOF
predictions instead keeps every young-onset patient (n=73) and matches how the
model is actually deployed (trained on the whole cohort, not per subgroup).

Age subgroups use the clinical young-onset PD definition (symptom onset < 50 y).
Output keys stay "young"/"old" for web-app backward compatibility.

Reads:   data/ml_calibration_predictions.csv  (full-cohort OOF, per clf/mt/score_set)
Outputs:
  data/ml_stratified.csv               (AUC + 95% CI per stratum)
  data/ml_stratified_predictions.csv   (OOF predictions tagged per stratum)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_loading import load_data  # noqa: E402
from src.clinical_metrics import bootstrap_auc  # noqa: E402

SCORE_SET = "luxpark"        # the deployed Standard-17 set
YOPD_ONSET_CUTOFF = 50.0     # young-onset PD: symptom onset < 50 years


def main():
    oof = pd.read_csv(os.path.join(ROOT, "data", "ml_calibration_predictions.csv"))
    oof = oof[oof["score_set"] == SCORE_SET].copy()

    d = load_data()
    onset = d.groupby("PATNO")["Age_at_onset"].first()
    sex = d.groupby("PATNO")["SEX"].first()
    oof["onset"] = oof["patno"].map(onset)
    oof["sexv"] = oof["patno"].map(sex)
    # NaN onset cannot be classified -- leave as None so it is excluded from
    # both age strata (np.where would otherwise mislabel NaN<50 as "old").
    oof["age_grp"] = np.where(
        oof["onset"].notna(),
        np.where(oof["onset"] < YOPD_ONSET_CUTOFF, "young", "old"), None)

    # Predictions file: each patient tagged once as an age stratum (sex='all')
    # and once as a sex stratum (age='all'), plus an 'all'/'all' block. This is
    # exactly the layout the web-app fairness panel expects.
    base_cols = ["patno", "y_true", "y_prob", "classifier", "model_type"]
    all_tag = oof.assign(age="all", sex="all")
    age_block = oof.dropna(subset=["onset"]).assign(age=lambda x: x["age_grp"],
                                                     sex="all")
    sex_block = (oof.dropna(subset=["sexv"])
                 .assign(age="all",
                         sex=lambda x: x["sexv"].astype(int).astype(str)))
    preds_out = pd.concat(
        [all_tag[base_cols + ["age", "sex"]],
         age_block[base_cols + ["age", "sex"]],
         sex_block[base_cols + ["age", "sex"]]], ignore_index=True)
    preds_out = preds_out.rename(columns={"patno": "PATNO"})
    preds_out.to_csv(os.path.join(ROOT, "data",
                                   "ml_stratified_predictions.csv"), index=False)

    # AUC + bootstrap CI per (classifier, model_type, stratum).
    strata = [
        ("all", "all", lambda x: x),
        ("young", "all", lambda x: x[x["age_grp"] == "young"]),
        ("old", "all", lambda x: x[x["age_grp"] == "old"]),
        ("all", "0", lambda x: x[x["sexv"] == 0]),
        ("all", "1", lambda x: x[x["sexv"] == 1]),
    ]
    rows = []
    for (clf, mt), g in oof.groupby(["classifier", "model_type"]):
        for age_label, sex_label, fn in strata:
            s = fn(g)
            n = s["patno"].nunique()
            if s["y_true"].nunique() < 2:
                continue
            bo = bootstrap_auc(s["y_true"].values, s["y_prob"].values)
            rows.append({"classifier": clf, "model_type": mt,
                         "age": age_label, "sex": sex_label, "n_patients": n,
                         "roc_auc": round(bo["auc"], 3),
                         "ci_lo": round(bo["auc_lo"], 3),
                         "ci_hi": round(bo["auc_hi"], 3),
                         "ci_mean": round(bo["auc_mean"], 3)})
            print(f"{clf:20s} {mt:18s} age={age_label:5s} sex={sex_label:3s} "
                  f"n={n:3d}  AUC {bo['auc']:.3f} "
                  f"[{bo['auc_lo']:.2f}-{bo['auc_hi']:.2f}]", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "data", "ml_stratified.csv"),
                               index=False)
    print("Saved data/ml_stratified.csv and ml_stratified_predictions.csv")


if __name__ == "__main__":
    main()
