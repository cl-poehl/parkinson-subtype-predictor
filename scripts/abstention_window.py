"""Accuracy-abstention trade-off by observation window (Section 3.8a).

Trains the calibrated Random Forest pipeline on the full PPMI cohort and, at
prediction time, restricts each held-out patient's slope/intercept features to
the visits within a fixed observation window (12..120 months and full). At each
window it reports the calibrated out-of-fold AUC and, from the 90% split-
conformal procedure averaged over 200 calibration/test splits, the abstention
rate (fraction of non-single-label prediction sets) and the accuracy on the
committed (single-label) patients.

Outputs:
  data/abstention_window.csv              (per-window summary)
  data/abstention_window_predictions.csv  (per-window OOF predictions)
"""
import os
import sys

import numpy as np
import pandas as pd

PPMI_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PPMI_REPO)
sys.path.insert(0, ROOT)

from data_loading import load_data
from ml_models import evaluate_cv
from sklearn.metrics import roc_auc_score

from src.constants import SCORES_LUXPARK

WINDOWS = [12, 24, 36, 48, 60, 84, 120, np.inf]
CLF = "random_forest"
N_SPLITS = 200


def conformal_metrics(y, p, alpha=0.10, n_splits=N_SPLITS):
    """LAC split-conformal abstention/accuracy/coverage averaged over splits."""
    y = np.asarray(y).astype(int)
    probs = np.column_stack([1 - np.asarray(p, float), np.asarray(p, float)])
    A, C, V = [], [], []
    for s in range(n_splits):
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y))
        h = len(y) // 2
        cal, te = idx[:h], idx[h:]
        nc = 1 - probs[cal, y[cal]]
        n = len(cal)
        qhat = np.sort(nc)[min(int(np.ceil((n + 1) * (1 - alpha))) - 1, n - 1)]
        ins = probs[te] >= (1 - qhat)
        comm = ins.sum(1) == 1
        pred = np.where(ins[:, 1] & comm, 1, 0)
        A.append((~comm).mean())
        C.append((pred[comm] == y[te][comm]).mean() if comm.any() else np.nan)
        V.append(ins[np.arange(len(te)), y[te]].mean())
    return float(np.nanmean(A)), float(np.nanmean(C)), float(np.nanmean(V))


def main():
    data = load_data()
    rows, preds = [], []
    for w in WINDOWS:
        res = evaluate_cv(data=data, model_type="slopes+intercepts",
                          classifier_name=CLF, scores=SCORES_LUXPARK, folds=10,
                          imputer="knn", calibrate=True, shorten_follow_up=w)
        pr = res["predictions"]
        y, p = pr["y_true"].values, pr["y_prob"].values
        wl = "full" if np.isinf(w) else int(w)
        for pat, yt, yp in zip(pr["PATNO"], y, p):
            preds.append({"window": wl, "patno": pat, "y_true": int(yt),
                          "y_prob": float(yp)})
        abst, accc, cov = conformal_metrics(y, p)
        rows.append({"window_months": wl, "n": len(pr),
                     "auc": round(roc_auc_score(y, p), 3),
                     "abstain_rate": round(abst, 3),
                     "committed_acc": round(accc, 3),
                     "coverage": round(cov, 3)})
        print(f"w={str(wl):>4} AUC {roc_auc_score(y, p):.3f} "
              f"abstain {abst:.3f} committed_acc {accc:.3f} cov {cov:.3f}",
              flush=True)
    out_dir = os.path.join(ROOT, "data")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "abstention_window.csv"),
                              index=False)
    pd.DataFrame(preds).to_csv(
        os.path.join(out_dir, "abstention_window_predictions.csv"), index=False)
    print("Saved data/abstention_window.csv and _predictions.csv")


if __name__ == "__main__":
    main()
