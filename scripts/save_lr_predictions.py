"""Saves the per-patient likelihood-ratio predictions from
SubtypePredictions/intermediate_data/ppmi_lr_scores_all.csv into
parkinson-subtype-predictor/data/lr_cv_predictions.csv, in the same
format as ml_calibration_predictions.csv (patno, y_true, y_prob,
score_set, model_type, classifier).

The LR AUC is a point estimate on log10_lr_total. score_set is set to
'luxpark' for the slopes+absolute_first model (corresponding to the
standard 17 configuration). For the full set there is currently no
parallel LR variant with 25 scores -- the 17-score config is mirrored
if needed.
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "lr_cv_predictions.csv")

# Precomputed per-patient Likelihood-Ratio scores. Place ppmi_lr_scores_all.csv
# under data/ (or set LR_SCORES_FILE) to regenerate; the committed
# data/lr_cv_predictions.csv is the output of this step.
LR_FILE = os.environ.get(
    "LR_SCORES_FILE", os.path.join(ROOT, "data", "ppmi_lr_scores_all.csv"))
if not os.path.exists(LR_FILE):
    print(f"Skip: {LR_FILE} not found "
          f"(set LR_SCORES_FILE or place the file under data/)")
    sys.exit(0)

src = pd.read_csv(LR_FILE)
# Subtype 1 = fast, 2 = slow. y_true = 1 for fast.
src["y_true"] = (src["Subtype"] == 1).astype(int)

# We use slopes+absolute_first as the LR analog to slopes+intercepts (ML).
sub = src[(src["model"] == "slopes+absolute_first") &
           src["log10_lr_total"].notna()].copy()
sub = sub.rename(columns={"Unnamed: 0": "patno", "log10_lr_total": "y_prob"})

# Map onto the [0,1] probability scale: monotone in log10_lr_total, so
# irrelevant for the AUC. We use min-max scaling so it can be represented
# as a probability. AUC and all ranking-based metrics are invariant.
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
