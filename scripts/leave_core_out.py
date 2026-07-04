"""Leave-core-out: when MDS-UPDRS core score(s) are missing, does the deployed
full model (which imputes them) lose to a model trained natively on the scores
that remain?

Motivation: the score-set comparison showed the three MDS-UPDRS core scores
carry almost all the signal (AUC 0.939 vs 0.943 for the full 17). That settles
the case where the core is PRESENT and the periphery is missing. It does NOT
settle the case the clinician worries about: a core score itself being missing,
where the full model imputes its single most informative feature. This script
tests, across one, two, and all three core scores missing, whether a model
trained only on the remaining scores beats that imputation.

Design -- same 10-fold StratifiedGroupKFold on PPMI, deployed config (Random
Forest, slopes+intercepts, KNN(k=5) imputer). For each missingness scenario two
arms share the exact same folds and patients, so the difference is paired:

  A "impute"  : Random Forest trained on ALL 17 scores; at TEST time the dropped
                core columns are blanked to NaN, so the fitted KNNImputer fills
                them from the patient's remaining features. This is exactly what
                the deployed app does when a score is absent.
  B "native"  : Random Forest trained AND tested only on the scores that remain.

AUC is rank-based and isotonic calibration is monotonic, so discrimination is
unaffected by calibration; we run uncalibrated (faster) -- the reported AUCs
equal the calibrated model's AUCs.

Output: data/leave_core_out.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from constants import SUBTYPE_FAST  # noqa: E402
from data_loading import load_data  # noqa: E402
from feature_extraction import extract_features, get_labels  # noqa: E402
from ml_models import CLASSIFIERS, IMPUTERS  # noqa: E402

FULL = ["UPDRS3_off", "UPDRS3_on", "UPDRS1", "UPDRS2", "UPDRS4", "MOCA",
        "SCOPA", "RBDScr", "VFT_phon_f", "JLO", "HY_off", "HY_on", "AXSC_off",
        "AXSC_on", "PIGD_off", "PIGD_on", "LEDD"]

UPDRS3 = ["UPDRS3_off", "UPDRS3_on"]  # "UPDRS-III missing" = both med states

# Each scenario lists the scores treated as MISSING at deployment, grouped by
# how many of the three MDS-UPDRS core parts are gone.
SCENARIOS = {
    # one core part missing
    "UPDRS-I only": ["UPDRS1"],
    "UPDRS-II only": ["UPDRS2"],
    "UPDRS-III only": UPDRS3,
    # two core parts missing
    "UPDRS-I + II": ["UPDRS1", "UPDRS2"],
    "UPDRS-I + III": ["UPDRS1"] + UPDRS3,
    "UPDRS-II + III": ["UPDRS2"] + UPDRS3,
    # all three core parts missing
    "All MDS-UPDRS core (I-III)": ["UPDRS1", "UPDRS2"] + UPDRS3,
}

MODEL_TYPE = "slopes+intercepts"
CLF = "random_forest"
IMP = "knn"
FOLDS = 10
N_BOOT = 1000
PATNO, SUBTYPE, TIME = "PATNO", "Subtype", "Disease_duration"


def _estimator():
    return Pipeline([
        ("imputer", IMPUTERS[IMP]()),
        ("scaler", StandardScaler()),
        ("clf", CLASSIFIERS[CLF]()),
    ])


def _feats(df, scores):
    return extract_features(df, MODEL_TYPE, scores, PATNO, TIME)


def _labels(df, index):
    return (get_labels(df, PATNO, SUBTYPE).loc[index] == SUBTYPE_FAST).astype(int)


def _oof(data, splits, train_scores, test_scores, mask_scores=None):
    """One arm: train on train_scores, test on test_scores, optionally blanking
    mask_scores columns at test time. Returns OOF DataFrame patno/y_true/y_prob."""
    rows = []
    for train_idx, test_idx in splits:
        tr, te = data.iloc[train_idx], data.iloc[test_idx]
        Xtr = _feats(tr, train_scores)
        ytr = _labels(tr, Xtr.index)
        Xte = _feats(te, test_scores)
        yte = _labels(te, Xte.index)
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        if mask_scores:
            Xte = Xte.copy()
            cols = [c for c in Xte.columns if c.rsplit("_", 1)[0] in mask_scores]
            Xte[cols] = np.nan
        est = _estimator()
        est.fit(Xtr.values, ytr.values)
        p = est.predict_proba(Xte.values)[:, 1]
        rows.append(pd.DataFrame({"patno": Xte.index, "y_true": yte.values,
                                   "y_prob": p}))
    return pd.concat(rows, ignore_index=True)


def _paired_delta_ci(a, b, seed=0):
    """Paired patient-level bootstrap 95% CI of AUC(A) - AUC(B) on shared patnos."""
    m = a.merge(b, on="patno", suffixes=("_a", "_b"))
    assert (m["y_true_a"] == m["y_true_b"]).all()
    y = m["y_true_a"].values
    pa, pb = m["y_prob_a"].values, m["y_prob_b"].values
    auc_a, auc_b = roc_auc_score(y, pa), roc_auc_score(y, pb)
    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return auc_a, auc_b, auc_a - auc_b, lo, hi, len(m)


def main():
    data = load_data()
    y_row = (data[SUBTYPE] == SUBTYPE_FAST).astype(int).values
    gkf = StratifiedGroupKFold(n_splits=FOLDS, random_state=0, shuffle=True)
    splits = list(gkf.split(data, y=y_row, groups=data[PATNO].values))

    ref = _oof(data, splits, FULL, FULL)
    ref_auc = roc_auc_score(ref["y_true"], ref["y_prob"])
    print(f"Reference (full 17, all present): AUC {ref_auc:.3f}", flush=True)

    out = [{"scenario": "Full 17 (reference)", "n_core_missing": 0,
            "auc_impute": round(ref_auc, 3), "auc_native": round(ref_auc, 3),
            "delta": 0.0, "delta_lo": 0.0, "delta_hi": 0.0, "n_patients": len(ref)}]

    # count core parts missing: UPDRS3_off+UPDRS3_on count as one part
    def n_core(drop):
        parts = set()
        for s in drop:
            parts.add("UPDRS3" if s in UPDRS3 else s)
        return len(parts & {"UPDRS1", "UPDRS2", "UPDRS3"})

    for name, drop in SCENARIOS.items():
        keep = [s for s in FULL if s not in drop]
        a = _oof(data, splits, FULL, FULL, mask_scores=drop)
        b = _oof(data, splits, keep, keep)
        auc_a, auc_b, d, lo, hi, npat = _paired_delta_ci(a, b)
        sig = "" if (lo <= 0 <= hi) else "  *"
        print(f"{name:30s} core_missing={n_core(drop)}  impute={auc_a:.3f}  "
              f"native={auc_b:.3f}  Δ(imp−nat)={d:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]{sig}", flush=True)
        out.append({"scenario": name, "n_core_missing": n_core(drop),
                    "auc_impute": round(auc_a, 3), "auc_native": round(auc_b, 3),
                    "delta": round(d, 3), "delta_lo": round(lo, 3),
                    "delta_hi": round(hi, 3), "n_patients": npat})

    pd.DataFrame(out).to_csv(os.path.join(ROOT, "data", "leave_core_out.csv"),
                              index=False)
    print("Saved data/leave_core_out.csv", flush=True)


if __name__ == "__main__":
    main()
