"""Trainiert die Slope-Verteilungen pro Subtyp fuer Toms Likelihood-Ratio-Methode.

Fuer jeden Score und jeden Subtyp werden Per-Patient-Slopes ueber alle PPMI-
Patienten gesammelt (mittels Linear Mixed Effects Modell, fixed + random
intercept und slope -- exakt wie in Tom's `calc_score_slope_distribution`).

Ausserdem speichern wir Perzentil-Referenzen (Slope und Intercept pro
Subtyp pro Score), damit die Webapp zeigen kann, wo ein Patient relativ
zur PPMI-Kohorte liegt.

Output: models/lr_reference_<mode>.joblib mit Struktur
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

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)

from data_loading import load_data
from likelihood import calc_score_slope_distribution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.constants import SCORE_LABELS, SCORES_LUXPARK

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(OUT_DIR, exist_ok=True)

SCORE_SETS = {
    "luxpark": list(SCORES_LUXPARK),
    "full": list(SCORE_LABELS.keys()),
}


def per_patient_intercepts(data, scores, subtype_col="Subtype",
                            patno_col="PATNO", time_col="Disease_duration"):
    """Per-Patient Intercept (Wert bei Disease_duration=0) via OLS pro Score, pro Subtyp."""
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
                # OLS-Intercept
                m = np.polyfit(x, y, 1)
                out[score][subtype].append(float(m[1]))
    return {s: {k: np.array(v) for k, v in d.items()} for s, d in out.items()}


def main():
    print("PPMI-Daten laden ...")
    data = load_data()

    for set_name, scores in SCORE_SETS.items():
        print(f"\n=== Score-Set '{set_name}' ({len(scores)} Scores) ===")
        slope_distributions = {}
        for score in scores:
            print(f"  Slope-Verteilung fuer {score} ...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    df_slopes = calc_score_slope_distribution(data=data, score_col=score)
                    # Spalten sind nach dem Score benannt + 'Subtype'
                    slope_distributions[score] = {
                        1: df_slopes[df_slopes["Subtype"] == 1][score].values,
                        2: df_slopes[df_slopes["Subtype"] == 2][score].values,
                    }
                except Exception as e:
                    print(f"    ! konnte nicht gefittet werden: {e}")
                    slope_distributions[score] = {1: np.array([]), 2: np.array([])}

        print("  Intercept-Verteilungen ...")
        intercept_distributions = per_patient_intercepts(data, scores)

        # Zusaetzlich OLS-Slopes pro Patient pro Subtyp speichern.
        # Diese matchen die feature_extraction der Webapp und werden fuer
        # Perzentile gegenueber der PPMI-Verteilung genutzt.
        print("  OLS-Slopes pro Patient (fuer Perzentile) ...")
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
