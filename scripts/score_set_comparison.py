"""Does training on a reduced 'common clinical' score set lose value versus the
full 17-score set (with imputation)? Answers the deployability / train-on-demand
question: if a routine MDS-UPDRS core matches the full panel, bespoke
per-score-set models are unnecessary.

Output: data/score_set_comparison.csv
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data_loading import load_data
from ml_models import evaluate_cv

SETS = {
    "Full (17, deployed)": ["UPDRS3_off", "UPDRS3_on", "UPDRS1", "UPDRS2",
        "UPDRS4", "MOCA", "SCOPA", "RBDScr", "VFT_phon_f", "JLO", "HY_off",
        "HY_on", "AXSC_off", "AXSC_on", "PIGD_off", "PIGD_on", "LEDD"],
    "MDS-UPDRS core (I-III)": ["UPDRS1", "UPDRS2", "UPDRS3_on"],
    "Routine clinic (UPDRS I-III + MoCA + HY)": ["UPDRS1", "UPDRS2", "UPDRS3_on", "MOCA", "HY_on"],
    "Motor only (UPDRS II/III + PIGD)": ["UPDRS2", "UPDRS3_on", "PIGD_on"],
    "Single best (UPDRS II)": ["UPDRS2"],
}


def main():
    d = load_data()
    rows = []
    for name, sc in SETS.items():
        r = evaluate_cv(data=d, model_type="slopes+intercepts",
                        classifier_name="random_forest", scores=sc, folds=10,
                        imputer="knn", calibrate=True)
        rows.append({"score_set": name, "n_scores": len(sc),
                     "roc_auc": round(r["roc_auc"], 3)})
        print(f"{name:46s} k={len(sc):>2} AUC={r['roc_auc']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(ROOT, "data", "score_set_comparison.csv"), index=False)
    print("Saved data/score_set_comparison.csv")


if __name__ == "__main__":
    main()
