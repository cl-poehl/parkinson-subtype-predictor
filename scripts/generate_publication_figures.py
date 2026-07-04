"""Erzeugt publikationsreife SVG-Figures aus den computed Daten.

Output: figures/*.svg, alle in Liberation Serif 8pt, sauberen
Vektor-Format, ohne Streamlit-Wrapper. Direkt fuer Manuskripte
verwendbar.

Figures:
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def fig5_abstention():
    """Accuracy-abstention trade-off by observation window (Section 3.8a)."""
    df = _load("abstention_window.csv")
    if df is None:
        print("Skip fig5: abstention_window.csv missing")
        return
    d = df[df["window_months"] != "full"].copy()
    d["w"] = d["window_months"].astype(int)
    d = d.sort_values("w")
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot(d["w"], d["auc"], "-o", color="#1f77b4", markersize=4,
             linewidth=1, label="ROC-AUC")
    ax1.plot(d["w"], d["committed_acc"], "-s", color="#2ca02c", markersize=4,
             linewidth=1, label="Accuracy when committed")
    ax1.set_xlabel("Observation window (months)")
    ax1.set_ylabel("AUC / committed accuracy")
    ax1.set_ylim(0.5, 1.0)
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(d["w"], d["abstain_rate"], "--^", color="#d62728", markersize=4,
             linewidth=1, label="Abstention rate")
    ax2.set_ylabel("Abstention rate", color="#d62728")
    ax2.set_ylim(0, 0.7)
    ax2.tick_params(axis="y", labelcolor="#d62728")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig5_abstention.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig6_trajectories():
    """Score trajectories by subtype: mean +/- SEM with individual patient
    overlays, plus the per-patient progression-slope distribution (UPDRS-II)."""
    from data_loading import load_data
    from scipy import stats
    SCORE = "UPDRS2"
    d = load_data().rename(columns={"PATNO": "patno", "Disease_duration": "dd"})
    d = d[["patno", "dd", SCORE, "Subtype"]].dropna()
    d["sub"] = np.where(d["Subtype"].values == 1, "fast", "slow")
    col = {"fast": "#d62728", "slow": "#1f77b4"}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7, 3.0))
    edges = np.arange(0, 145, 12)
    centers = (edges[:-1] + edges[1:]) / 2
    for sub in ["slow", "fast"]:
        g = d[d["sub"] == sub]
        bi = np.digitize(g["dd"].values, edges) - 1
        means, sems, xs = [], [], []
        for b in range(len(centers)):
            v = g[SCORE].values[bi == b]
            if len(v) >= 5:
                means.append(v.mean()); sems.append(v.std(ddof=1) / np.sqrt(len(v))); xs.append(centers[b])
        means, sems, xs = np.array(means), np.array(sems), np.array(xs)
        axA.plot(xs, means, "-", color=col[sub], lw=1.8, label=f"{sub} (mean)", zorder=3)
        axA.fill_between(xs, means - sems, means + sems, color=col[sub], alpha=0.2, zorder=1)
        rng = np.random.default_rng(1)
        for p in rng.choice(g["patno"].unique(), size=min(4, g["patno"].nunique()), replace=False):
            pg = g[g["patno"] == p].sort_values("dd")
            axA.plot(pg["dd"], pg[SCORE], "-", color=col[sub], lw=0.5, alpha=0.35, zorder=2)
    axA.set_xlabel("Disease duration (months)"); axA.set_ylabel("MDS-UPDRS II")
    axA.legend(frameon=False, loc="upper left")
    axA.set_title("Mean trajectory (+/- SEM) with individual patients")
    slopes = {"fast": [], "slow": []}
    for sub in ["fast", "slow"]:
        for p, pg in d[d["sub"] == sub].groupby("patno"):
            v = pg[["dd", SCORE]].dropna()
            if len(v) >= 2:
                slopes[sub].append(stats.linregress(v["dd"].values, v[SCORE].values).slope)
    LO, HI = -0.5, 1.2
    clip = {k: [min(max(s, LO), HI) for s in v] for k, v in slopes.items()}
    parts = axB.violinplot([clip["slow"], clip["fast"]], showmedians=True, positions=[0, 1])
    for pc, sub in zip(parts["bodies"], ["slow", "fast"]):
        pc.set_facecolor(col[sub]); pc.set_alpha(0.5)
    axB.set_ylim(LO, HI)
    axB.axhline(np.median(slopes["fast"] + slopes["slow"]), color="#888", ls="--", lw=0.6, label="PPMI median")
    axB.set_xticks([0, 1]); axB.set_xticklabels(["slow", "fast"])
    axB.set_ylabel("UPDRS-II slope (points/month)")
    axB.set_title("Per-patient progression slope"); axB.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig6_trajectories.svg")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    print(f"Generating publication SVGs into {FIG_DIR} ...")
    fig2_reliability()
    fig3_shap()
    fig4_roc()
    fig5_abstention()
    fig6_trajectories()
    figS7_km()
    figS10_stress()
    print(f"\nAll figures saved. List:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
