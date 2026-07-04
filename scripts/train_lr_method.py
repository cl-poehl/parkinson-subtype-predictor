"""Trains the per-subtype slope distributions for the likelihood ratio method.

For each score and each subtype, per-patient slopes are collected across all
PPMI patients (using a linear mixed effects model with fixed + random
intercept and slope, analogous to `calc_score_slope_distribution` from the
SubtypePredictions code base).

We also store percentile references (slope and intercept per subtype per
score) so the webapp can show where a patient lies relative to the PPMI
cohort.

Output: models/lr_reference_<mode>.joblib with structure
{
    'slope_distributions': {score: {1: np.array, 2: np.array}},
    'intercept_distributions': {score: {1: np.array, 2: np.array}},
}
"""
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

PPMI_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PPMI_REPO)

from data_loading import load_data
from likelihood import calc_score_slope_distribution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import SCORE_LABELS, SCORES_LUXPARK

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "auxiliary")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_SETS = {
    "luxpark": list(SCORES_LUXPARK),
    "full": list(SCORE_LABELS.keys()),
}


def per_patient_intercepts(data, scores, subtype_col="Subtype",
                            patno_col="PATNO", time_col="Disease_duration"):
    """Per-patient intercept (value at Disease_duration=0) via OLS per score, per subtype."""
    out = {s: {1: [], 2: []} for s in scores}
    for subtype in [1, 2]:
        sub = data[data[subtype_col] == subtype]
        for patno, grp in sub.groupby(patno_col):
            for score in scores:
                vals = grp[[time_col, score]].dropna()
                if len(vals) < 2:
                    continue
                x = vals[time_col].values
                y = vals[score].values
                # OLS intercept
                m = np.polyfit(x, y, 1)
                out[score][subtype].append(float(m[1]))
    return {s: {k: np.array(v) for k, v in d.items()} for s, d in out.items()}


def main():
    print("Loading PPMI data ...")
    data = load_data()

    for set_name, scores in SCORE_SETS.items():
        print(f"\n=== Score-Set '{set_name}' ({len(scores)} Scores) ===")
        slope_distributions = {}
        for score in scores:
            print(f"  Slope distribution for {score} ...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df_slopes = calc_score_slope_distribution(data=data, score_col=score)
                    # Columns are named after the score + 'Subtype'
                    slope_distributions[score] = {
                        1: df_slopes[df_slopes["Subtype"] == 1][score].values,
                        2: df_slopes[df_slopes["Subtype"] == 2][score].values,
                    }
                except Exception as e:
                    print(f"    ! could not be fitted: {e}")
                    slope_distributions[score] = {1: np.array([]), 2: np.array([])}

        print("  Intercept distributions ...")
        intercept_distributions = per_patient_intercepts(data, scores)

        # Additionally store OLS slopes per patient per subtype.
        # These match the feature_extraction of the webapp and are used for
        # percentiles relative to the PPMI distribution.
        print("  OLS slopes per patient (for percentiles) ...")
        ols_slopes = {s: {1: [], 2: []} for s in scores}
        for subtype in [1, 2]:
            sub = data[data["Subtype"] == subtype]
            for patno, grp in sub.groupby("PATNO"):
                for score in scores:
                    vals = grp[["Disease_duration", score]].dropna()
                    if len(vals) < 2:
                        continue
                    m = np.polyfit(vals["Disease_duration"].values,
                                   vals[score].values, 1)
                    ols_slopes[score][subtype].append(float(m[0]))
        ols_slopes = {s: {k: np.array(v) for k, v in d.items()}
                       for s, d in ols_slopes.items()}

        payload = {
            "slope_distributions": slope_distributions,
            "intercept_distributions": intercept_distributions,
            "ols_slope_distributions": ols_slopes,
            "scores": scores,
        }
        outpath = os.path.join(OUT_DIR, f"lr_reference_{set_name}.joblib")
        joblib.dump(payload, outpath)
        print(f"  -> {outpath}")


if __name__ == "__main__":
    main()
