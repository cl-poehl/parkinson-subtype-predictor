"""Adversariale Robustheit / Stress-Test gegen Mess-Rauschen.

Frage: Wie robust sind die Vorhersagen, wenn die Eingangs-Scores
verrauscht sind (z.B. UPDRS3 +/- 2 Punkte Inter-Rater-Variabilitaet)?
Anteil der Patienten, deren Klassen-Vorhersage bei Rauschen flippt.

Methodik:
- Fuer jedes Noise-Level (0%, 5%, 10%, 20%, 30% relative SD):
  - 100 Realisierungen mit Gauss-Rauschen auf den OBSERVATIONS-Werten
    der einzelnen Visits (vor Slope-Extraktion!)
  - Pipeline neu mit verrauschten Inputs auswerten
  - Anteil der Patienten, deren P(Fast) ueber 0.5 wechselt: Flip-Rate
  - Mean absolute change in P(Fast)

Output: data/stress_test.csv + docs/STRESS_TEST.md
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import SCORES_LUXPARK, SCORE_RANGES
from src.features import extract_slope_intercept


N_REALISATIONS = 50
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]  # SD as fraction of score range


def make_pipeline():
    return Pipeline([
        ("imp", KNNImputer(n_neighbors=5)),
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])


def main():
    out_dir = os.path.join(ROOT, "data")
    docs_dir = os.path.join(ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    data = load_data()
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])

    subtype = df.groupby("patno")["Subtype"].first()
    y_true = (subtype == 1).astype(int)

    # Score ranges fuer realistisches Rauschen (in raw units)
    scales = {s: (SCORE_RANGES.get(s, (0, 1))[1] -
                   SCORE_RANGES.get(s, (0, 1))[0])
              for s in SCORES_LUXPARK}

    # Referenz: ungestoerte Features, ungestoertes Modell
    feats_ref = extract_slope_intercept(df, SCORES_LUXPARK)
    common = feats_ref.index.intersection(y_true.index)
    X_ref = feats_ref.loc[common]
    y = y_true.loc[common].values
    pipe = make_pipeline()
    pipe.fit(X_ref.values, y)
    p_ref = pipe.predict_proba(X_ref.values)[:, 1]
    pred_ref = (p_ref >= 0.5).astype(int)

    rng = np.random.default_rng(42)
    rows = []
    for noise_lvl in NOISE_LEVELS:
        for rep in range(N_REALISATIONS if noise_lvl > 0 else 1):
            t0 = time.time()
            df_noisy = df.copy()
            if noise_lvl > 0:
                for s in SCORES_LUXPARK:
                    sd = noise_lvl * scales[s]
                    mask = df_noisy[s].notna()
                    noise = rng.normal(0, sd, size=mask.sum())
                    df_noisy.loc[mask, s] = df_noisy.loc[mask, s] + noise
                    # Clip auf gueltigen Range
                    lo, hi = SCORE_RANGES.get(s, (None, None))[:2]
                    if lo is not None:
                        df_noisy[s] = df_noisy[s].clip(lower=lo, upper=hi)
            feats_n = extract_slope_intercept(df_noisy, SCORES_LUXPARK)
            X_n = feats_n.loc[common]
            p_n = pipe.predict_proba(X_n.values)[:, 1]
            pred_n = (p_n >= 0.5).astype(int)
            flip = (pred_n != pred_ref).mean()
            abs_delta = float(np.abs(p_n - p_ref).mean())
            rows.append({
                "noise_lvl": noise_lvl, "realisation": rep,
                "flip_rate": float(flip),
                "mean_abs_p_change": abs_delta,
                "duration_s": time.time() - t0,
            })
        last = rows[-1]
        print(f"  noise={noise_lvl}: flip={last['flip_rate']:.3f}  "
               f"mean|dP|={last['mean_abs_p_change']:.3f}")

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "stress_test.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    summary = df_out.groupby("noise_lvl").agg(
        flip_mean=("flip_rate", "mean"), flip_sd=("flip_rate", "std"),
        abs_mean=("mean_abs_p_change", "mean"),
        abs_sd=("mean_abs_p_change", "std"),
    ).reset_index()

    md_path = os.path.join(docs_dir, "STRESS_TEST.md")
    lines = ["# Stress Test: Robustness to Measurement Noise", ""]
    lines.append("Random Forest predictions are evaluated after injecting "
                  "Gaussian noise into the input scores (per-visit). "
                  "Noise SD is given as fraction of each score's full "
                  "range; e.g., 10% noise on MoCA (range 0-30) corresponds "
                  "to SD = 3 points.")
    lines.append("")
    lines.append(f"Realisations per noise level: {N_REALISATIONS}.")
    lines.append("")
    lines.append("## Flip rate and probability drift")
    lines.append("")
    lines.append("| Noise SD (rel) | Flip rate (mean +/- SD) | Mean |dP(Fast)| (mean +/- SD) |")
    lines.append("|----------------|--------------------------|--------------------------------|")
    for _, r in summary.iterrows():
        lines.append(f"| {r['noise_lvl']*100:.0f}%           | "
                      f"{r['flip_mean']:.3f} +/- {r['flip_sd']:.3f}   | "
                      f"{r['abs_mean']:.3f} +/- {r['abs_sd']:.3f}     |")
    lines.append("")
    lines.append("Flip rate = fraction of patients whose predicted class "
                  "(Fast vs Slow at threshold 0.5) changes due to noise. "
                  "Mean |dP| = average absolute change in predicted "
                  "P(Fast) across patients.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- At zero noise (deterministic baseline) the flip rate "
                  "is zero by construction.")
    lines.append("- A flip rate < 5% at the typical inter-rater noise level "
                  "(~5-10% of range for clinical scores) indicates the "
                  "model is robust to plausible measurement variability.")
    lines.append("- Flip rates above 10% would suggest patients near the "
                  "decision boundary -- the Conformal prediction set "
                  "{Fast, Slow} should be invoked rather than a sharp "
                  "classification.")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
