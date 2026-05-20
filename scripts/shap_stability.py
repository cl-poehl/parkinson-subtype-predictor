"""SHAP-Feature-Importance-Stabilitaet ueber Bootstrap-Resamples des
Trainingssets.

Frage: Sind die Top-Features stabil? Wenn wir das Modell auf einem
zufaelligen Bootstrap-Resample der 409 PPMI-Patienten trainieren,
bekommen wir konsistent dieselben Top-Features?

Metrik: Spearman-Rangkorrelation der per-feature mean(|SHAP|)-Werte
zwischen Bootstrap-Iterationen und einem Referenz-Modell.

Output: data/shap_stability.csv und docs/SHAP_STABILITY.md
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import SCORES_LUXPARK
from src.features import extract_slope_intercept


N_BOOTSTRAPS = 50


def shap_means(model, X_imp):
    """Mittelwert |SHAP| pro Feature. Funktioniert mit Tree-Modellen."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_imp)
    if isinstance(sv, list):
        # binary classification: TreeExplainer returns list per class
        sv = sv[1] if len(sv) == 2 else sv[0]
    elif sv.ndim == 3:
        sv = sv[..., 1]
    return np.abs(sv).mean(axis=0)


def fit_rf_pipeline(X, y):
    pipe = Pipeline([
        ("imp", KNNImputer(n_neighbors=5)),
        ("sc", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X, y)
    return pipe


def main():
    out_dir = os.path.join(ROOT, "data")
    docs_dir = os.path.join(ROOT, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    data = load_data()
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])
    subtype = df.groupby("patno")["Subtype"].first()
    y_full = (subtype == 1).astype(int)
    feats = extract_slope_intercept(df, SCORES_LUXPARK)
    common = feats.index.intersection(y_full.index)
    X = feats.loc[common].copy()
    y = y_full.loc[common].values
    feature_names = list(X.columns)
    print(f"n={len(X)} patients, {X.shape[1]} features")

    # Referenzmodell auf vollem Datensatz
    pipe_ref = fit_rf_pipeline(X.values, y)
    imp = pipe_ref.named_steps["imp"]
    sc = pipe_ref.named_steps["sc"]
    clf_ref = pipe_ref.named_steps["clf"]
    X_imp_ref = sc.transform(imp.transform(X.values))
    ref_shap = shap_means(clf_ref, X_imp_ref)
    ref_ranking = pd.Series(ref_shap, index=feature_names).rank(ascending=False)

    # Bootstrap-Resamples
    rng = np.random.default_rng(42)
    n = len(X)
    rows = []
    for b in range(N_BOOTSTRAPS):
        t0 = time.time()
        idx = rng.integers(0, n, n)
        Xb = X.iloc[idx].copy()
        yb = y[idx]
        if np.unique(yb).size < 2:
            continue
        pipe = fit_rf_pipeline(Xb.values, yb)
        imp_b = pipe.named_steps["imp"]
        sc_b = pipe.named_steps["sc"]
        clf_b = pipe.named_steps["clf"]
        X_b_imp = sc_b.transform(imp_b.transform(X.values))
        sh = shap_means(clf_b, X_b_imp)
        ser = pd.Series(sh, index=feature_names)
        rho, _ = spearmanr(ref_shap, sh)
        # Top-5 Stabilitaet
        top5_ref = set(ref_ranking[ref_ranking <= 5].index)
        top5_boot = set(ser.rank(ascending=False)[ser.rank(ascending=False) <= 5].index)
        overlap = len(top5_ref & top5_boot)
        for feat, v in zip(feature_names, sh):
            rows.append({"bootstrap": b, "feature": feat, "abs_shap": v,
                          "rank": int(ser.rank(ascending=False)[feat]),
                          "spearman_vs_ref": float(rho),
                          "top5_overlap": overlap})
        print(f"  bootstrap {b}: rho={rho:.3f}, top5-overlap={overlap}/5 "
               f"({time.time()-t0:.1f}s)")

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "shap_stability.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Zusammenfassung
    per_boot = df_out.groupby("bootstrap").agg(
        spearman=("spearman_vs_ref", "first"),
        top5_overlap=("top5_overlap", "first"),
    ).reset_index()

    md_path = os.path.join(docs_dir, "SHAP_STABILITY.md")
    lines = ["# SHAP Feature Importance Stability", ""]
    lines.append(f"Random Forest re-trained on {N_BOOTSTRAPS} bootstrap "
                  "resamples (patient-level) of the training set. "
                  "Feature importance computed as mean |SHAP value| per "
                  "feature over the full PPMI cohort.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Mean Spearman correlation of |SHAP| rankings vs. "
                  f"full-data reference: **{per_boot['spearman'].mean():.3f}**"
                  f" (SD {per_boot['spearman'].std(ddof=1):.3f})")
    lines.append(f"- Mean Top-5 feature overlap with reference: "
                  f"**{per_boot['top5_overlap'].mean():.1f}/5** (SD "
                  f"{per_boot['top5_overlap'].std(ddof=1):.1f})")
    lines.append("")
    lines.append("Spearman correlation > 0.7 indicates stable feature "
                  "ranking. Top-5 overlap of 4-5 indicates that the most "
                  "important features are robustly identified across "
                  "different random samples of the training data.")
    lines.append("")
    lines.append("## Per-feature mean |SHAP| (Bootstrap)")
    lines.append("")
    grp = df_out.groupby("feature").agg(
        mean_shap=("abs_shap", "mean"),
        sd_shap=("abs_shap", "std"),
        mean_rank=("rank", "mean"),
    ).sort_values("mean_shap", ascending=False)
    lines.append("| Feature | Mean |SHAP| | SD | Mean rank |")
    lines.append("|---------|-------------|----|-----------|")
    for feat, r in grp.head(20).iterrows():
        lines.append(f"| {feat} | {r['mean_shap']:.4f} | {r['sd_shap']:.4f} "
                      f"| {r['mean_rank']:.1f} |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
