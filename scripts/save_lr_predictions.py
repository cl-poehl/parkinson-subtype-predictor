"""Speichert die per-Patient Likelihood-Ratio-Predictions aus
SubtypePredictions/intermediate_data/ppmi_lr_scores_all.csv in
parkinson-subtype-predictor/data/lr_cv_predictions.csv, im gleichen
Format wie ml_calibration_predictions.csv (patno, y_true, y_prob,
score_set, model_type, classifier).

LR-AUC ist ein Punktschaetzer auf log10_lr_total. score_set wird auf
'luxpark' gesetzt fuer das slopes+absolute_first-Modell (entspricht der
Standard-17 Konfiguration). Fuer das full-Set existiert aktuell keine
parallele LR-Variante mit 25 scores -- es wird die 17-Score-Konfig
gespiegelt, falls noetig.
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBTYPES = "/Users/carl/Documents/SubtypePredictions/intermediate_data"
OUT = os.path.join(ROOT, "data", "lr_cv_predictions.csv")

LR_FILE = os.path.join(SUBTYPES, "ppmi_lr_scores_all.csv")
if not os.path.exists(LR_FILE):
    print(f"Skip: {LR_FILE} not found")
    sys.exit(0)

src = pd.read_csv(LR_FILE)
# Subtype 1 = fast, 2 = slow. y_true = 1 fuer fast.
src["y_true"] = (src["Subtype"] == 1).astype(int)

# Wir nehmen slopes+absolute_first als LR-Analog zu slopes+intercepts (ML).
sub = src[(src["model"] == "slopes+absolute_first") &
           src["log10_lr_total"].notna()].copy()
sub = sub.rename(columns={"Unnamed: 0": "patno", "log10_lr_total": "y_prob"})

# Auf [0,1] Probability-Skala bringen: monoton mit log10_lr_total, daher
# fuers AUC egal. Wir nehmen min-max Skalierung damit es als probability
# darstellbar ist. AUC und alle ranking-basierten Metriken sind invariant.
y = sub["y_prob"].values
y_min, y_max = y.min(), y.max()
sub["y_prob"] = (y - y_min) / (y_max - y_min) if y_max > y_min else 0.5

rows = []
for score_set in ("luxpark", "full"):
    for model_type in ("slopes", "slopes+intercepts"):
        out = sub[["patno", "y_true", "y_prob"]].copy()
        out["classifier"] = "likelihood_ratio"
        out["score_set"] = score_set
        out["model_type"] = model_type
        rows.append(out)

out = pd.concat(rows, ignore_index=True)
out = out[["classifier", "score_set", "model_type", "patno", "y_true", "y_prob"]]
out.to_csv(OUT, index=False)
print(f"Saved {len(out)} rows -> {OUT}")
print(out.groupby(["score_set", "model_type"]).size())
