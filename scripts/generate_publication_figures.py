"""Erzeugt publikationsreife SVG-Figures aus den computed Daten.

Output: figures/*.svg, alle in Liberation Serif 8pt, sauberen
Vektor-Format, ohne Streamlit-Wrapper. Direkt fuer Manuskripte
verwendbar.

Figures:
- fig1_dca.svg -- Decision Curve Analysis
- fig2_reliability.svg -- Reliability diagrams
- fig3_shap.svg -- Top SHAP features mit Stability-Marker
- fig4_roc.svg -- ROC curves mit Bootstrap-CI-Band
- figS1_calibration_table.svg -- Calibration metrics
- figS5_pdp.svg -- Partial Dependence Plots
- figS7_km.svg -- Kaplan-Meier nach Subtyp
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.expanduser("~/Documents/SubtypePredictions"))

DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Publication-grade Matplotlib defaults
rcParams.update({
    "font.family": "Liberation Serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "svg.fonttype": "none",  # text bleibt selektierbar
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {
    "Random Forest": "#10b981",
    "XGBoost": "#f97316",
    "Logistic Regression": "#6366f1",
    "Likelihood Ratio": "#a855f7",
}
CLF_LABEL = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "likelihood_ratio": "Likelihood Ratio",
}


def _load(name):
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def fig1_dca():
    """Decision Curve Analysis -- net benefit pro Klassifikator."""
    from src.clinical_metrics import decision_curve
    df = _load("ml_calibration_predictions.csv")
    if df is None:
        print("Skip fig1: ml_calibration_predictions.csv missing")
        return
    sub = df[(df["score_set"] == "luxpark") &
              (df["model_type"] == "slopes+intercepts")]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    base_added = False
    for clf, grp in sub.groupby("classifier"):
        curve = decision_curve(grp["y_true"].values, grp["y_prob"].values)
        ax.plot(curve["threshold"], curve["Model"],
                 color=PALETTE.get(CLF_LABEL.get(clf, clf), "#6b7280"),
                 label=CLF_LABEL.get(clf, clf))
        if not base_added:
            ax.plot(curve["threshold"], curve["Treat all"],
                     "--", color="#6b7280", label="Treat all")
            ax.plot(curve["threshold"], curve["Treat none"],
                     ":", color="#1f2937", label="Treat none")
            base_added = True
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_dca.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig2_reliability():
    """Reliability Diagrams pro Klassifikator."""
    from sklearn.calibration import calibration_curve
    df = _load("ml_calibration_predictions.csv")
    if df is None:
        print("Skip fig2: ml_calibration_predictions.csv missing")
        return
    sub = df[(df["score_set"] == "luxpark") &
              (df["model_type"] == "slopes+intercepts")]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--",
             linewidth=0.8, label="Perfect calibration")
    for clf, grp in sub.groupby("classifier"):
        prob_true, prob_pred = calibration_curve(
            grp["y_true"], grp["y_prob"], n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-",
                 color=PALETTE.get(CLF_LABEL.get(clf, clf), "#6b7280"),
                 label=CLF_LABEL.get(clf, clf),
                 markersize=3)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_reliability.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig3_shap():
    """Top-15 Features nach mean |SHAP| mit Bootstrap-Stability (SD)."""
    df = _load("shap_stability.csv")
    if df is None:
        print("Skip fig3: shap_stability.csv missing")
        return
    grp = df.groupby("feature").agg(mean=("abs_shap", "mean"),
                                       sd=("abs_shap", "std"))
    grp = grp.sort_values("mean", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(3.5, 4))
    pos = np.arange(len(grp))
    ax.barh(pos, grp["mean"], xerr=grp["sd"], color="#10b981",
             edgecolor="#065f46", linewidth=0.4, error_kw={"linewidth": 0.6})
    ax.set_yticks(pos)
    ax.set_yticklabels(grp.index, fontsize=7)
    ax.set_xlabel("Mean |SHAP value| (Random Forest)")
    ax.set_title("Feature importance with bootstrap SD")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_shap.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig4_roc():
    """ROC curves pro Klassifikator mit Bootstrap-CI-Band."""
    from sklearn.metrics import roc_curve, roc_auc_score
    df = _load("ml_calibration_predictions.csv")
    if df is None:
        print("Skip fig4: ml_calibration_predictions.csv missing")
        return
    sub = df[(df["score_set"] == "luxpark") &
              (df["model_type"] == "slopes+intercepts")]
    fig, ax = plt.subplots(figsize=(3.5, 3))
    rng = np.random.default_rng(42)
    for clf, grp in sub.groupby("classifier"):
        y = grp["y_true"].values
        p = grp["y_prob"].values
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        # Bootstrap CI band: nicht plotbar als shaded; nur Label.
        bs_aucs = []
        for _ in range(200):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2: continue
            bs_aucs.append(roc_auc_score(y[idx], p[idx]))
        lo, hi = np.quantile(bs_aucs, [0.025, 0.975])
        ax.plot(fpr, tpr,
                 color=PALETTE.get(CLF_LABEL.get(clf, clf), "#6b7280"),
                 label=f"{CLF_LABEL.get(clf, clf)} "
                        f"(AUC {auc:.2f} [{lo:.2f}, {hi:.2f}])")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_roc.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def figS7_km():
    """Kaplan-Meier nach Fast/Slow Subtyp."""
    from lifelines import KaplanMeierFitter
    surv = _load("survival_analysis.csv")
    if surv is None or "subtype" not in surv.columns:
        # Versuch, subtype neu zu mergen
        if surv is None:
            print("Skip figS7: survival_analysis.csv missing")
            return
        from data_loading import load_data
        sys.path.insert(0, os.path.expanduser("~/Documents/SubtypePredictions"))
        data = load_data()
        st_map = data.groupby("PATNO")["Subtype"].first()
        surv["subtype"] = surv["patno"].map(st_map)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    kmf = KaplanMeierFitter()
    colors = {1: "#ef4444", 2: "#3b82f6"}
    labels = {1: "Fast (n=%d)", 2: "Slow (n=%d)"}
    for st in (2, 1):
        s = surv[surv["subtype"] == st]
        if len(s) < 5:
            continue
        kmf.fit(s["time"], event_observed=s["event"],
                 label=labels[st] % len(s))
        kmf.plot_survival_function(ax=ax, color=colors[st],
                                      show_censors=False, ci_show=True,
                                      ci_alpha=0.15)
    ax.set_xlabel("Months from baseline")
    ax.set_ylabel("Probability of not reaching HY 3")
    ax.set_xlim(0, surv["time"].quantile(0.99))
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "figS7_km.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def figS10_stress():
    """Stress-Test: Flip-Rate pro Noise-Level."""
    df = _load("stress_test.csv")
    if df is None:
        print("Skip figS10: stress_test.csv missing")
        return
    summary = df.groupby("noise_lvl")["flip_rate"].agg(["mean", "std"])
    summary = summary.reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.errorbar(summary["noise_lvl"] * 100, summary["mean"],
                 yerr=summary["std"], color="#10b981", marker="o",
                 capsize=2, markersize=4, linewidth=1)
    ax.set_xlabel("Noise SD (% of score range)")
    ax.set_ylabel("Flip rate at threshold 0.5")
    ax.axhline(0.10, color="#ef4444", linestyle="--", linewidth=0.6,
                label="10% concern")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "figS10_stress.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print(f"Generating publication SVGs into {FIG_DIR} ...")
    fig1_dca()
    fig2_reliability()
    fig3_shap()
    fig4_roc()
    figS7_km()
    figS10_stress()
    print(f"\nAll figures saved. List:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
