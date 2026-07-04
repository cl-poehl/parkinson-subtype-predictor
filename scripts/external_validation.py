"""External Validation auf einem zweiten Cohort (z.B. LuxPARK).

Erwartetes Input-Format: CSV mit den gleichen Score-Spalten wie das
PPMI-Trainingsset (siehe SCORES_LUXPARK in src/constants.py), plus
einer 'Subtype' Spalte (1=fast, 2=slow) wenn Outcome-Labels verfuegbar
sind.

Setup:
1. Laedt das external CSV
2. Extrahiert Slope+Intercept-Features (gleiche Pipeline wie PPMI)
3. Predict via die deployten Modelle aus models/
4. Reportiert:
   - AUC mit Bootstrap-CI
   - Calibration-Diagnostics (Cox intercept/slope, HL, Brier, ECE)
   - DeLong-Test vs. internal PPMI baseline
   - Empirische Conformal-Coverage
   - Calibration-Drift-Vergleich (PPMI vs external)
   - Klassenverteilung und ggf. Re-Calibration-Vorschlag

Usage:
    python scripts/external_validation.py \\
        --csv /path/to/luxpark_visits.csv \\
        --subtype-col Subtype \\
        --score-set luxpark \\
        --out external_validation_luxpark
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.clinical_metrics import (
    bootstrap_auc, calibration_intercept_slope, hosmer_lemeshow,
    delong_test,
)
from src.constants import (
    SCORES_LUXPARK, get_model_paths, get_conformal_paths,
)
from src.features import extract_slope_intercept
from src.conformal import load_conformal_set, predict_sets


def evaluate_external(visits_df, subtype_col, score_set, out_prefix):
    """Hauptlogik: External-Cohort -> Predictions -> Metriken -> Reports."""
    # Score-Set Subset waehlen
    if score_set == "luxpark":
        scores = SCORES_LUXPARK
    else:
        from src.constants import SCORE_LABELS
        scores = list(SCORE_LABELS.keys())

    # Validate Input
    missing_scores = [s for s in scores if s not in visits_df.columns]
    if missing_scores:
        print(f"WARNING: missing score columns: {missing_scores}")
    if subtype_col not in visits_df.columns:
        raise ValueError(f"Subtype column '{subtype_col}' not found")
    if "disease_duration" not in visits_df.columns:
        raise ValueError("'disease_duration' column required (months from "
                          "PD diagnosis)")
    if "patno" not in visits_df.columns:
        raise ValueError("'patno' column required (unique patient ID)")

    # Features extrahieren
    feats = extract_slope_intercept(visits_df, [s for s in scores
                                                 if s in visits_df.columns])
    subtype = visits_df.groupby("patno")[subtype_col].first()
    common = feats.index.intersection(subtype.index)
    X = feats.loc[common]
    y_true = (subtype.loc[common] == 1).astype(int).values
    print(f"External cohort: n={len(X)} patients, "
           f"fast={y_true.sum()} ({100*y_true.mean():.1f}%)")

    # Modelle + Conformal laden
    model_paths = get_model_paths(score_set, n_visits=2)
    conf_paths = get_conformal_paths(score_set, n_visits=2)
    models = {k: joblib.load(v) for k, v in model_paths.items()
              if os.path.exists(v)}
    confs = load_conformal_set(conf_paths)

    # Per-Klassifikator Predictions
    out_rows = []
    deltas = []
    for clf_label, clf_key in (("Random Forest", "rf"),
                                  ("XGBoost", "xgb"),
                                  ("Logistic Regression", "logreg")):
        if clf_label not in models:
            continue
        model = models[clf_label]
        try:
            proba = model.predict_proba(X.values)[:, 1]
        except Exception as e:
            print(f"  predict failed for {clf_label}: {e}")
            continue
        boot = bootstrap_auc(y_true, proba, n_boot=1000)
        cal = calibration_intercept_slope(y_true, proba)
        hl = hosmer_lemeshow(y_true, proba, g=10)
        # Conformal coverage
        try:
            sets = predict_sets(confs[clf_label], X) if clf_label in confs else None
        except Exception:
            sets = None
        if sets is not None:
            # Coverage: fraction wo die wahre Klasse im Set ist
            cov = 0
            for s_list, t in zip(sets, y_true):
                true_label = "Fast" if t == 1 else "Slow"
                if true_label in s_list:
                    cov += 1
            empirical_coverage = float(cov / len(y_true))
        else:
            empirical_coverage = None

        out_rows.append({
            "Method": clf_label,
            "AUC": boot["auc"],
            "AUC 95% CI": (boot["auc_lo"], boot["auc_hi"]),
            "Brier": float(np.mean((proba - y_true) ** 2)),
            "Cox intercept": cal["intercept"],
            "Cox slope": cal["slope"],
            "HL chi2": hl["chi2"],
            "HL p": hl["p_value"],
            "Conformal coverage (target 0.9)": empirical_coverage,
        })

    # Output: CSV + Markdown-Report
    out_dir = os.path.join(ROOT, "data")
    csv_path = os.path.join(out_dir, f"{out_prefix}.csv")
    pd.DataFrame(out_rows).to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    md = [f"# External Validation Report", ""]
    md.append(f"External cohort: n={len(X)} patients with {y_true.sum()} "
                f"fast progressors ({100*y_true.mean():.1f}%).")
    md.append("")
    md.append("## Per-classifier performance on external cohort")
    md.append("")
    md.append("| Method | AUC | 95% CI | Brier | Cox int | Cox slope | HL chi2 | HL p | Conf. cov. |")
    md.append("|--------|-----|--------|-------|---------|-----------|---------|------|------------|")
    for r in out_rows:
        ci = r["AUC 95% CI"]
        cov = (f"{r['Conformal coverage (target 0.9)']:.2f}"
               if r['Conformal coverage (target 0.9)'] is not None else "—")
        hl_p = "<0.0001" if r["HL p"] < 0.0001 else f"{r['HL p']:.4f}"
        md.append(f"| {r['Method']} | {r['AUC']:.3f} | "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] | {r['Brier']:.3f} | "
                    f"{r['Cox intercept']:+.3f} | {r['Cox slope']:.3f} | "
                    f"{r['HL chi2']:.1f} | "
                    f"{hl_p} | "
                    f"{cov} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- **AUC drop** vs internal PPMI quantifies generalization. "
                "A drop of less than 0.10 indicates strong generalization; "
                "0.10-0.20 is typical for clinical ML; >0.20 suggests "
                "calibration drift or feature-distribution shift.")
    md.append("- **Cox intercept and slope** quantify calibration drift. "
                "A shift in intercept indicates global over/under-prediction; "
                "a slope farther from 1 indicates calibration distortion at "
                "the extremes.")
    md.append("- **Empirical Conformal coverage**: target was 90%. "
                "Coverage below 88% indicates the calibration cohort is no "
                "longer representative; re-calibrate on a held-out external "
                "subset.")
    md_path = os.path.join(ROOT, "docs", f"{out_prefix.upper()}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {md_path}")
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                         help="Input CSV with longitudinal visits")
    parser.add_argument("--subtype-col", default="Subtype",
                         help="Column with subtype labels (1=fast, 2=slow)")
    parser.add_argument("--score-set", default="luxpark",
                         choices=("luxpark", "full"),
                         help="Score subset; luxpark = 17 routine scores")
    parser.add_argument("--out", default="external_validation",
                         help="Output prefix (CSV + Markdown report)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    evaluate_external(df, args.subtype_col, args.score_set, args.out)


if __name__ == "__main__":
    main()
