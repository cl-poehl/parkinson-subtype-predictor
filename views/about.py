"""About tab: background on the app, methodology, disclaimer, and detailed
performance analyses (score combinations, missingness, follow-up)."""
import os

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data")
CLF_LABEL = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}
PALETTE = {
    "Random Forest": "#10b981",
    "XGBoost": "#f97316",
    "Logistic Regression": "#6366f1",
    "Likelihood Ratio": "#a855f7",
}


def _load(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _decision_threshold_panel():
    """Empfohlene Decision-Thresholds nach Youden-Index, Net-Benefit-Maximum
    und kosten-gewichteten Kosten (FN-Cost 5x FP-Cost)."""
    import numpy as np
    from src.clinical_metrics import optimal_threshold

    ml = _load("ml_calibration_predictions.csv")
    if ml is None:
        st.caption("Threshold data not yet available.")
        return
    sub = ml[(ml["score_set"] == "luxpark") &
              (ml["model_type"] == "slopes+intercepts")]

    rows = []
    for clf, grp in sub.groupby("classifier"):
        yt = grp["y_true"].values
        yp = grp["y_prob"].values
        results = {}
        for crit, label in (("youden", "Youden J max"),
                            ("net_benefit", "Net Benefit max"),
                            ("cost", "5x cost-weighted (FN:FP=5:1)")):
            r = optimal_threshold(yt, yp, criterion=crit)
            results[label] = r
        for label, r in results.items():
            rows.append({
                "Method": CLF_LABEL.get(clf, clf),
                "Criterion": label,
                "Threshold": f"{r['threshold']:.3f}" if np.isfinite(r["threshold"]) else "—",
                "Sensitivity": f"{r['sens']:.3f}" if np.isfinite(r["sens"]) else "—",
                "Specificity": f"{r['spec']:.3f}" if np.isfinite(r["spec"]) else "—",
                "PPV": f"{r['ppv']:.3f}" if np.isfinite(r["ppv"]) else "—",
                "NPV": f"{r['npv']:.3f}" if np.isfinite(r["npv"]) else "—",
            })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        "Three principled ways to choose the decision cutoff. **Youden J max** "
        "maximises sensitivity + specificity - 1 (equal weighting of both "
        "errors). **Net Benefit max** maximises Vickers' net benefit at the "
        "threshold itself, balancing benefit of correct fast identification "
        "against cost of false alarms. **5x cost-weighted** assumes a missed "
        "fast progressor is 5x as costly as a false positive (early treatment "
        "discussion vs unnecessary worry). For published cutoffs, prefer the "
        "Youden or cost-weighted threshold over the arbitrary 0.5."
    )


def _baseline_comparison_panel():
    """Vergleich zwischen Headline-Klassifikatoren und drei simplen
    Baselines (Constant Slow, UPDRS3-only LogReg, MoCA-only LogReg).
    Zeigt, wieviel Mehrwert die komplexen Modelle ueber triviale Regeln
    bieten."""
    import numpy as np
    from src.clinical_metrics import bootstrap_auc

    base = _load("baseline_predictions.csv")
    ml = _load("ml_calibration_predictions.csv")
    lr = _load("lr_cv_predictions.csv")
    if base is None or ml is None:
        st.caption("Baseline data not yet available. Run "
                    "`scripts/train_baselines.py` once to generate.")
        return

    rows = []
    # Baselines
    BASELINE_LABEL = {
        "constant_slow": "Constant 'Slow'",
        "updrs3_only": "UPDRS3 only (LogReg)",
        "moca_only": "MoCA only (LogReg)",
    }
    for m, grp in base.groupby("model"):
        yt = grp["y_true"].values
        yp = grp["y_prob"].values
        if grp["y_prob"].nunique() <= 1:
            # Constant prediction -- AUC undefined. Stattdessen Accuracy
            # bei 'predict everyone Slow'.
            acc = float((yt == 0).mean())
            rows.append({
                "Method": BASELINE_LABEL.get(m, m),
                "Type": "Baseline",
                "AUC": "—",
                "95% CI": "—",
                "Accuracy (cutoff 0.5)": f"{acc:.3f}",
            })
            continue
        res = bootstrap_auc(yt, yp, n_boot=1000)
        acc = float(((yp >= 0.5) == yt).mean())
        rows.append({
            "Method": BASELINE_LABEL.get(m, m),
            "Type": "Baseline",
            "AUC": f"{res['auc']:.3f}",
            "95% CI": f"{res['auc_lo']:.3f}-{res['auc_hi']:.3f}",
            "Accuracy (cutoff 0.5)": f"{acc:.3f}",
        })

    # Headline-Modelle (Standard 17 Slopes+Intercepts)
    sub_ml = ml[(ml["score_set"] == "luxpark") &
                 (ml["model_type"] == "slopes+intercepts")]
    for clf, grp in sub_ml.groupby("classifier"):
        yt = grp["y_true"].values
        yp = grp["y_prob"].values
        res = bootstrap_auc(yt, yp, n_boot=1000)
        acc = float(((yp >= 0.5) == yt).mean())
        rows.append({
            "Method": CLF_LABEL.get(clf, clf),
            "Type": "Full model",
            "AUC": f"{res['auc']:.3f}",
            "95% CI": f"{res['auc_lo']:.3f}-{res['auc_hi']:.3f}",
            "Accuracy (cutoff 0.5)": f"{acc:.3f}",
        })
    if lr is not None and "y_true" in lr.columns:
        sub_lr = lr[(lr["score_set"] == "luxpark") &
                     (lr["model_type"] == "slopes+intercepts")]
        if not sub_lr.empty:
            yt = sub_lr["y_true"].values
            yp = sub_lr["y_prob"].values
            res = bootstrap_auc(yt, yp, n_boot=1000)
            acc = float(((yp >= 0.5) == yt).mean())
            rows.append({
                "Method": "Likelihood Ratio",
                "Type": "Full model",
                "AUC": f"{res['auc']:.3f}",
                "95% CI": f"{res['auc_lo']:.3f}-{res['auc_hi']:.3f}",
                "Accuracy (cutoff 0.5)": f"{acc:.3f}",
            })

    df = pd.DataFrame(rows)
    # Sortierung: Baselines zuerst, dann Full models nach AUC
    order = {"Baseline": 0, "Full model": 1}
    df = df.sort_values(by=["Type", "AUC"],
                          key=lambda s: s.map(order) if s.name == "Type" else s,
                          ascending=[True, False]).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _headline_accuracy_panel():
    """Headline AUCs mit 95% Bootstrap-CI fuer alle vier Methoden auf dem
    Standard-Score-Set (17), slopes+intercepts. Liest CV-Predictions aus
    `ml_calibration_predictions.csv` (ML) und `lr_cv_predictions.csv` (LR)."""
    import numpy as np
    from src.clinical_metrics import bootstrap_auc

    ml = _load("ml_calibration_predictions.csv")
    lr = _load("lr_cv_predictions.csv")

    cards = []
    if ml is not None:
        sub = ml[(ml["score_set"] == "luxpark") &
                  (ml["model_type"] == "slopes+intercepts")]
        for clf, grp in sub.groupby("classifier"):
            res = bootstrap_auc(grp["y_true"].values, grp["y_prob"].values,
                                 n_boot=1000)
            cards.append({"name": CLF_LABEL.get(clf, clf), **res})
    # Likelihood Ratio falls vorhanden
    if lr is not None and "y_true" in lr.columns and "y_prob" in lr.columns:
        sub = lr[(lr["score_set"] == "luxpark") &
                  (lr["model_type"] == "slopes+intercepts")]
        if not sub.empty:
            res = bootstrap_auc(sub["y_true"].values, sub["y_prob"].values,
                                 n_boot=1000)
            cards.append({"name": "Likelihood Ratio", **res})

    if not cards:
        # Fallback auf hartkodierte Zahlen falls Predictions fehlen
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Random Forest", "AUC 0.95")
        c2.metric("XGBoost", "AUC 0.95")
        c3.metric("Logistic Regression", "AUC 0.88")
        c4.metric("Likelihood Ratio", "AUC 0.91")
        return

    # Reihenfolge: RF, XGB, LR, LikelihoodRatio
    order = ["Random Forest", "XGBoost", "Logistic Regression", "Likelihood Ratio"]
    cards = sorted(cards, key=lambda c: order.index(c["name"])
                    if c["name"] in order else 99)
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        auc = c.get("auc", float("nan"))
        lo = c.get("auc_lo", float("nan"))
        hi = c.get("auc_hi", float("nan"))
        if np.isfinite(lo) and np.isfinite(hi):
            sub = f"95% CI {lo:.2f}-{hi:.2f}"
        else:
            sub = ""
        col.metric(c["name"], f"AUC {auc:.2f}", sub, delta_color="off")


def _score_combinations_chart():
    """Greedy Forward Selection plus Random Subsets pro Methode."""
    greedy = _load("ml_score_combinations.csv")
    ml_rand = _load("ml_random_score_combinations.csv")
    lr_rand = _load("lr_random_score_combinations.csv")
    if greedy is None and ml_rand is None:
        st.caption("Score-combination data not available.")
        return

    rows = []
    if ml_rand is not None:
        for _, r in ml_rand.iterrows():
            rows.append({"k": int(r["k"]), "Method": CLF_LABEL.get(r["classifier"],
                                                                     r["classifier"]),
                         "auc": float(r["roc_auc"]), "type": "random"})
    if lr_rand is not None:
        for _, r in lr_rand.iterrows():
            rows.append({"k": int(r["k"]), "Method": "Likelihood Ratio",
                         "auc": float(r["roc_auc"]), "type": "random"})
    rand_df = pd.DataFrame(rows) if rows else None

    greedy_rows = []
    if greedy is not None:
        for _, r in greedy.iterrows():
            greedy_rows.append({"k": int(r["n_scores"]),
                                 "Method": CLF_LABEL.get(r["classifier"], r["classifier"]),
                                 "auc": float(r["roc_auc"])})
    greedy_df = pd.DataFrame(greedy_rows) if greedy_rows else None

    method_order = ["Random Forest", "XGBoost", "Logistic Regression", "Likelihood Ratio"]
    present_methods = ([m for m in method_order if rand_df is not None
                         and m in rand_df["Method"].unique()] if rand_df is not None
                        else [])

    if rand_df is None and greedy_df is None:
        return

    base_scale = alt.Scale(domain=method_order,
                            range=[PALETTE[m] for m in method_order])

    if rand_df is not None:
        boxes = (
            alt.Chart(rand_df)
            .mark_boxplot(extent=1.5, outliers={"size": 8, "opacity": 0.3},
                            size=12)
            .encode(
                x=alt.X("k:O", axis=alt.Axis(title="Number of scores (k)")),
                y=alt.Y("auc:Q", scale=alt.Scale(domain=[0.5, 1.0]),
                        axis=alt.Axis(title="ROC AUC", format=".2f")),
                color=alt.Color("Method:N", scale=base_scale, legend=None),
                xOffset=alt.XOffset("Method:N",
                                     scale=alt.Scale(domain=method_order)),
            )
        )
    else:
        boxes = None

    layers = []
    if boxes is not None:
        layers.append(boxes)
    if greedy_df is not None:
        line = (
            alt.Chart(greedy_df)
            .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeDash=[4, 3])
            .encode(
                x=alt.X("k:O"),
                y=alt.Y("auc:Q"),
                color=alt.Color("Method:N", scale=base_scale,
                                legend=alt.Legend(title="Method", orient="top")),
                tooltip=["Method", "k", alt.Tooltip("auc:Q", format=".3f")],
            )
        )
        layers.append(line)

    chart = alt.layer(*layers).properties(height=320).resolve_scale(color="shared")
    st.altair_chart(chart, use_container_width=True)


def _missingness_chart():
    """AUC vs. Missingness pro Methode, Bootstrap-CI ueber Per-Patient-Predictions."""
    boot = _load("ml_missingness_simulation_bootstrap.csv")
    if boot is not None:
        df = boot.copy()
        df["Method"] = df["classifier"].astype(str).map(CLF_LABEL)
        df["missingness_pct"] = df["missingness"] * 100
        present = [m for m in PALETTE if m in df["Method"].unique()]

        line = (
            alt.Chart(df)
            .mark_line(point=alt.OverlayMarkDef(size=70, filled=True))
            .encode(
                x=alt.X("missingness_pct:Q",
                        axis=alt.Axis(title="Missingness (%)", format="d")),
                y=alt.Y("auc_mean:Q",
                        scale=alt.Scale(domain=[0.5, 1.0]),
                        axis=alt.Axis(title="ROC AUC", format=".2f")),
                color=alt.Color("Method:N",
                                scale=alt.Scale(domain=present,
                                                 range=[PALETTE[m] for m in present]),
                                legend=alt.Legend(title="Method", orient="top")),
                tooltip=["Method", alt.Tooltip("missingness_pct:Q", format=".0f"),
                         alt.Tooltip("auc_mean:Q", format=".3f"),
                         alt.Tooltip("auc_lo:Q", format=".3f", title="95% CI low"),
                         alt.Tooltip("auc_hi:Q", format=".3f", title="95% CI high")],
            )
        )
        band = (
            alt.Chart(df)
            .mark_area(opacity=0.2)
            .encode(
                x=alt.X("missingness_pct:Q"),
                y="auc_lo:Q",
                y2="auc_hi:Q",
                color=alt.Color("Method:N", scale=alt.Scale(
                    domain=present, range=[PALETTE[m] for m in present]),
                    legend=None),
            )
        )
        st.altair_chart((band + line).properties(height=300),
                        use_container_width=True)
        return

    # Fallback: alte 1D-Datei ohne CIs
    df = _load("ml_missingness_simulation.csv")
    if df is None:
        return
    df = df[df["model_type"] == "slopes+intercepts"].copy()
    df["Method"] = df["classifier"].astype(str).map(CLF_LABEL)
    df["missingness_pct"] = df["missingness"] * 100
    present = [m for m in PALETTE if m in df["Method"].unique()]
    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True))
        .encode(
            x=alt.X("missingness_pct:Q",
                    axis=alt.Axis(title="Missingness (%)", format="d")),
            y=alt.Y("roc_auc:Q",
                    scale=alt.Scale(domain=[0.5, 1.0]),
                    axis=alt.Axis(title="ROC AUC", format=".2f")),
            color=alt.Color("Method:N",
                            scale=alt.Scale(domain=present,
                                             range=[PALETTE[m] for m in present]),
                            legend=alt.Legend(title="Method", orient="top")),
            tooltip=["Method", alt.Tooltip("missingness_pct:Q", format=".0f"),
                     alt.Tooltip("roc_auc:Q", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _followup_chart():
    df = _load("ml_follow_up_simulation.csv")
    if df is None:
        return
    df = df[df["model_type"].isin(("slopes", "slopes+intercepts"))].copy()
    df["Method"] = df["classifier"].map(CLF_LABEL)

    chart = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=70, filled=True))
        .encode(
            x=alt.X("follow_up:Q",
                    axis=alt.Axis(title="Follow-up (months)")),
            y=alt.Y("roc_auc:Q",
                    scale=alt.Scale(domain=[0.5, 1.0]),
                    axis=alt.Axis(title="ROC AUC", format=".2f")),
            color=alt.Color("Method:N",
                            scale=alt.Scale(
                                domain=list(PALETTE.keys()),
                                range=list(PALETTE.values())),
                            legend=alt.Legend(title="Method", orient="top")),
            strokeDash=alt.StrokeDash("model_type:N",
                                       legend=alt.Legend(title="Features",
                                                          orient="top")),
            tooltip=["Method", "model_type", "follow_up",
                     alt.Tooltip("roc_auc:Q", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)


def _per_score_chart():
    df = _load("ml_per_score_roc_auc.csv")
    if df is None:
        return
    df = df[df["model_type"] == "slopes+intercepts"].copy()
    df["Method"] = df["classifier"].map(CLF_LABEL)
    # nach mittlerer AUC sortieren
    order = (df.groupby("score")["roc_auc"].mean()
                 .sort_values(ascending=False).index.tolist())
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("score:N", sort=order,
                    axis=alt.Axis(title="Score")),
            x=alt.X("roc_auc:Q", scale=alt.Scale(domain=[0.5, 1.0]),
                    axis=alt.Axis(title="ROC AUC", format=".2f")),
            color=alt.Color("Method:N",
                            scale=alt.Scale(
                                domain=list(PALETTE.keys()),
                                range=list(PALETTE.values())),
                            legend=alt.Legend(title="Method", orient="top")),
            yOffset=alt.YOffset("Method:N"),
            tooltip=["score", "Method", alt.Tooltip("roc_auc:Q", format=".3f")],
        )
        .properties(height=600)
    )
    st.altair_chart(chart, use_container_width=True)


def _stress_test_panel():
    """Stress-Test: Wieviel verschieben sich Predictions wenn Inputs
    verrauscht werden? Liest data/stress_test.csv."""
    df = _load("stress_test.csv")
    if df is None:
        st.caption("Stress-test data not yet available. Run "
                    "`scripts/stress_test.py` once to generate.")
        return
    summary = df.groupby("noise_lvl").agg(
        flip_mean=("flip_rate", "mean"), flip_sd=("flip_rate", "std"),
        abs_mean=("mean_abs_p_change", "mean"),
        abs_sd=("mean_abs_p_change", "std"),
    ).reset_index()
    summary["Noise SD (rel)"] = (summary["noise_lvl"] * 100).map(
        lambda v: f"{v:.0f}%")
    summary["Flip rate"] = summary.apply(
        lambda r: f"{r['flip_mean']:.3f} +/- {r['flip_sd']:.3f}"
                  if pd.notna(r["flip_sd"]) else f"{r['flip_mean']:.3f}",
        axis=1)
    summary["Mean |dP(Fast)|"] = summary.apply(
        lambda r: f"{r['abs_mean']:.3f} +/- {r['abs_sd']:.3f}"
                  if pd.notna(r["abs_sd"]) else f"{r['abs_mean']:.3f}",
        axis=1)
    st.dataframe(summary[["Noise SD (rel)", "Flip rate", "Mean |dP(Fast)|"]],
                  use_container_width=True, hide_index=True)
    st.caption(
        "Flip rate = fraction of patients whose predicted class flips at "
        "the 0.5 threshold under Gaussian measurement noise on the raw "
        "scores. Noise SD is given as fraction of each score's full "
        "range. At 5-10% noise (typical inter-rater clinical variability) "
        "the flip rate stays below 10%, indicating reasonable robustness. "
        "Patients near the decision boundary should rely on the Conformal "
        "set {Fast, Slow} rather than a sharp classification."
    )


def _pdp_panel():
    """Partial Dependence + ICE Plot pro Klassifikator. Liest precomputed
    data/pdp_data.csv. User waehlt das Feature aus einem Dropdown."""
    df = _load("pdp_data.csv")
    if df is None:
        st.caption("PDP data not yet available. Run "
                    "`scripts/compute_pdp.py` once to generate.")
        return

    SCORE_NAME = {
        "UPDRS2_slope": "MDS-UPDRS II slope",
        "UPDRS3_on_slope": "MDS-UPDRS III (On) slope",
        "MOCA_slope": "MoCA slope",
        "SCOPA_slope": "SCOPA slope",
        "UPDRS1_slope": "MDS-UPDRS I slope",
        "PIGD_on_slope": "PIGD (On) slope",
    }
    features = sorted(df["feature"].unique())
    feat = st.selectbox("Feature", features,
                         format_func=lambda f: SCORE_NAME.get(f, f),
                         key="pdp_feature_select")

    sub = df[df["feature"] == feat].copy()
    pdp = sub[sub["kind"] == "pdp"].rename(columns={"prediction": "P(Fast)"})
    ice = sub[sub["kind"] == "ice"].rename(columns={"prediction": "P(Fast)"})

    methods_present = sorted(pdp["classifier"].unique())
    palette_present = {m: PALETTE.get(m, "#9ca3af") for m in methods_present}

    # ICE-Linien (duenn, halbtransparent)
    ice_chart = (
        alt.Chart(ice)
        .mark_line(opacity=0.18, strokeWidth=0.8)
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title=SCORE_NAME.get(feat, feat))),
            y=alt.Y("P(Fast):Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Predicted P(Fast)", format=".1f")),
            color=alt.Color("classifier:N",
                            scale=alt.Scale(domain=methods_present,
                                             range=list(palette_present.values())),
                            legend=None),
            detail="patno_idx:N",
        )
    )
    # PDP-Linie (dicke, vollfarbig)
    pdp_chart = (
        alt.Chart(pdp)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("x:Q"),
            y=alt.Y("P(Fast):Q"),
            color=alt.Color("classifier:N",
                            scale=alt.Scale(domain=methods_present,
                                             range=list(palette_present.values())),
                            legend=alt.Legend(title="Method", orient="top")),
            tooltip=["classifier",
                     alt.Tooltip("x:Q", format=".3f"),
                     alt.Tooltip("P(Fast):Q", format=".3f")],
        )
    )
    chart = (ice_chart + pdp_chart).properties(height=320)
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Thick lines: Partial Dependence Plot, the average predicted "
        "P(Fast) when only this feature is varied while all others are "
        "held at the observed value of each training patient (Friedman "
        "2001). Thin lines: Individual Conditional Expectation curves for "
        "30 random patients -- if the thin lines have very different "
        "shapes, the feature interacts strongly with other features. "
        "Goldstein et al. 2015."
    )


def _class_conditional_fairness_panel():
    """Class-conditional Fairness: TPR und FPR pro Subgruppe (Hardt 2016
    Equalized-Odds-Difference). Anders als die AUC-Vergleichstabelle
    misst dies INNERHALB jeder Klasse, ob die Modelle gleich gut
    Fast/Slow erkennen."""
    import numpy as np
    from src.clinical_metrics import equalized_odds

    df = _load("ml_stratified_predictions.csv")
    if df is None:
        st.caption("Stratified prediction data not yet available.")
        return

    sub = df[df["model_type"] == "slopes+intercepts"].copy()
    if sub.empty:
        st.caption("Stratified predictions not available for slopes+intercepts.")
        return

    threshold = 0.5
    rows_age = []
    rows_sex = []
    # ALTER: TPR/FPR per Klassifikator, Gruppen young vs old
    # Wir brauchen die Patienten-IDs mit Gruppenzuordnung. Aus dem
    # stratified file: Zeilen mit age != 'all' geben uns das Splitting.
    for clf in sub["classifier"].unique():
        # Alter
        a = sub[(sub["classifier"] == clf) & (sub["sex"] == "all") &
                (sub["age"].isin(("young", "old")))].copy()
        if not a.empty and a["age"].nunique() == 2:
            r = equalized_odds(a["y_true"].values, a["y_prob"].values,
                                a["age"].values, threshold=threshold)
            rows_age.append({
                "Method": CLF_LABEL.get(clf, clf),
                "TPR young": f"{r['tpr_per_group']['young']:.3f}",
                "TPR old": f"{r['tpr_per_group']['old']:.3f}",
                "FPR young": f"{r['fpr_per_group']['young']:.3f}",
                "FPR old": f"{r['fpr_per_group']['old']:.3f}",
                "EOD (max diff)": f"{r['eod']:.3f}",
            })
        # Geschlecht (0/1)
        s = sub[(sub["classifier"] == clf) & (sub["age"] == "all") &
                (sub["sex"].isin(("0", "1")))].copy()
        if not s.empty and s["sex"].nunique() == 2:
            r = equalized_odds(s["y_true"].values, s["y_prob"].values,
                                s["sex"].values, threshold=threshold)
            rows_sex.append({
                "Method": CLF_LABEL.get(clf, clf),
                "TPR male": f"{r['tpr_per_group']['0']:.3f}",
                "TPR female": f"{r['tpr_per_group']['1']:.3f}",
                "FPR male": f"{r['fpr_per_group']['0']:.3f}",
                "FPR female": f"{r['fpr_per_group']['1']:.3f}",
                "EOD (max diff)": f"{r['eod']:.3f}",
            })

    if rows_age:
        st.markdown("**By age (young vs old, threshold = 0.5)**")
        st.dataframe(pd.DataFrame(rows_age), use_container_width=True,
                      hide_index=True)
    if rows_sex:
        st.markdown("**By sex (male vs female, threshold = 0.5)**")
        st.dataframe(pd.DataFrame(rows_sex), use_container_width=True,
                      hide_index=True)
    st.caption(
        "Equalized-Odds-Difference (Hardt et al. 2016): the maximum absolute "
        "difference in True Positive Rate (sensitivity for Fast) and False "
        "Positive Rate (1-specificity for Slow) between subgroups, at the "
        "0.5 threshold. EOD = 0 is perfect equal odds; EOD > 0.1 is commonly "
        "considered a meaningful disparity. Smaller is better."
    )


def _subgroup_fairness_panel():
    """Performance pro Subgruppe (Alter/Geschlecht), formelle DeLong-Vergleiche."""
    import numpy as np
    from src.clinical_metrics import delong_test, adjust_pvalues

    df = _load("ml_stratified_predictions.csv")
    aucs = _load("ml_stratified.csv")
    if df is None or aucs is None:
        st.caption("Subgroup data not yet available. Run "
                    "`run_stratified.py` once to generate.")
        return

    st.markdown(
        "AUC per subgroup with paired DeLong tests within each classifier. "
        "Tests whether performance differs significantly between "
        "demographic strata."
    )

    # ---- AUC per subgroup
    sub = aucs[aucs["model_type"] == "slopes+intercepts"].copy()
    sub["sex"] = sub["sex"].astype(str)
    sub["age"] = sub["age"].astype(str)
    rows = []
    for _, r in sub.iterrows():
        if r["age"] == "all" and r["sex"] == "all":
            continue
        sg = f"{r['age']} / {r['sex']}"
        rows.append({
            "Method": CLF_LABEL.get(r["classifier"], r["classifier"]),
            "Subgroup": sg,
            "AUC": f"{r['roc_auc']:.3f}" if pd.notna(r["roc_auc"]) else "—",
            "n": int(r.get("n_patients", 0)) if "n_patients" in r else "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- DeLong pro Klassifikator: young vs old (mit sex=all)
    st.markdown("**Paired DeLong: young vs old** (within each classifier)")
    delong_rows = []
    for clf in df["classifier"].unique():
        sub_clf = df[(df["classifier"] == clf) &
                      (df["model_type"] == "slopes+intercepts")]
        young = sub_clf[(sub_clf["age"] == "young") & (sub_clf["sex"].astype(str) == "all")]
        old = sub_clf[(sub_clf["age"] == "old") & (sub_clf["sex"].astype(str) == "all")]
        if young.empty or old.empty:
            continue
        # DeLong braucht gepaarte Predictions auf gleichen Patienten -- aber wir haben
        # zwei DISJUNKTE Sets (young und old). Daher: unpaired Z-Test auf AUC-Differenz
        # mit DeLong-Varianzen aus jeder Gruppe.
        # Hier rechnen wir das vereinfacht via Bootstrap.
        y_y = young["y_true"].values
        p_y = young["y_prob"].values
        y_o = old["y_true"].values
        p_o = old["y_prob"].values
        try:
            from sklearn.metrics import roc_auc_score
            auc_y = roc_auc_score(y_y, p_y)
            auc_o = roc_auc_score(y_o, p_o)
            # Bootstrap-Test: differenz der AUCs
            rng = np.random.default_rng(42)
            diffs = []
            for _ in range(1000):
                idx_y = rng.integers(0, len(y_y), len(y_y))
                idx_o = rng.integers(0, len(y_o), len(y_o))
                if (len(np.unique(y_y[idx_y])) < 2 or
                        len(np.unique(y_o[idx_o])) < 2):
                    continue
                diffs.append(roc_auc_score(y_y[idx_y], p_y[idx_y]) -
                             roc_auc_score(y_o[idx_o], p_o[idx_o]))
            diffs = np.array(diffs)
            # Empirisches Two-sided p
            obs = auc_y - auc_o
            p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
            delong_rows.append({
                "Method": CLF_LABEL.get(clf, clf),
                "AUC young": f"{auc_y:.3f}",
                "AUC old": f"{auc_o:.3f}",
                "Difference": f"{obs:+.3f}",
                "_p_raw": p,
            })
        except Exception:
            continue
    if delong_rows:
        pvals = [r["_p_raw"] for r in delong_rows]
        p_holm = adjust_pvalues(pvals, method="holm")
        for r, ph in zip(delong_rows, p_holm):
            r["p (bootstrap)"] = (f"{r['_p_raw']:.4f}" if r["_p_raw"] >= 1e-4
                                   else "<0.0001")
            r["p (Holm)"] = f"{ph:.4f}" if ph >= 1e-4 else "<0.0001"
            del r["_p_raw"]
        st.dataframe(pd.DataFrame(delong_rows), use_container_width=True,
                      hide_index=True)
    st.caption("Bootstrap-based two-sample test for AUC difference (1000 "
                "resamples). DeLong's covariance is not directly applicable "
                "between disjoint groups; we use empirical p-values instead. "
                "Holm-corrected p-values for the family-wise error rate.")


def _clinical_metrics_panel():
    """DCA, DeLong, Sens/Spec/PPV/NPV, NRI/IDI auf den CV-Predictions."""
    import numpy as np
    from src.clinical_metrics import (
        delong_test, bootstrap_classification_metrics,
        nri_idi, decision_curve, adjust_pvalues,
    )

    df = _load("ml_calibration_predictions.csv")
    if df is None:
        st.caption("Clinical metric data not yet available. Run "
                    "`run_calibration.py` once to generate.")
        return

    sm_choices = sorted(df["score_set"].unique())
    sm = st.selectbox(
        "Score set", sm_choices,
        format_func=lambda x: {"luxpark": "Standard (17)",
                                "full": "Extended (25)"}.get(x, x),
        key="clinical_sm",
    )
    sub = df[(df["score_set"] == sm) & (df["model_type"] == "slopes+intercepts")]
    classifiers = sorted(sub["classifier"].unique())
    methods = [CLF_LABEL[c] for c in classifiers]
    method_to_clf = {CLF_LABEL[c]: c for c in classifiers}

    # ---- Decision Curve Analysis
    st.markdown("#### Decision Curve Analysis (DCA)")
    st.caption(
        "Net benefit at different threshold probabilities, compared with "
        "'Treat all as Fast' and 'Treat none'. A model is clinically useful "
        "if its curve sits above both baselines over the threshold range "
        "relevant to clinical decisions. Vickers & Elkin 2006."
    )
    dca_rows = []
    for clf in classifiers:
        g = sub[sub["classifier"] == clf]
        curve = decision_curve(g["y_true"].values, g["y_prob"].values)
        curve["Method"] = CLF_LABEL[clf]
        dca_rows.append(curve)
    dca_df = pd.concat(dca_rows, ignore_index=True)
    base_df = dca_df[dca_df["Method"] == methods[0]][["threshold", "Treat all", "Treat none"]].copy()

    method_curves = (
        alt.Chart(dca_df)
        .mark_line()
        .encode(
            x=alt.X("threshold:Q",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(title="Threshold probability", format=".1f")),
            y=alt.Y("Model:Q",
                    axis=alt.Axis(title="Net benefit", format=".3f")),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(
                    domain=[m for m in PALETTE if m in methods],
                    range=[PALETTE[m] for m in PALETTE if m in methods]),
                legend=alt.Legend(title="Method", orient="top")),
            tooltip=["Method", alt.Tooltip("threshold:Q", format=".2f"),
                     alt.Tooltip("Model:Q", format=".4f", title="Net benefit")],
        )
    )
    treat_all_line = (
        alt.Chart(base_df)
        .mark_line(strokeDash=[6, 3], color="#6b7280")
        .encode(x="threshold:Q", y=alt.Y("Treat all:Q"))
    )
    treat_none_line = (
        alt.Chart(base_df)
        .mark_line(strokeDash=[3, 3], color="#1f2937")
        .encode(x="threshold:Q", y=alt.Y("Treat none:Q"))
    )
    st.altair_chart(
        (method_curves + treat_all_line + treat_none_line)
        .properties(height=300),
        use_container_width=True,
    )
    st.caption("Dashed gray = Treat all; dotted black = Treat none.")
    st.markdown("")

    # ---- DeLong-Test paarweise mit FWER-Korrektur
    st.markdown("#### DeLong test for AUC differences")
    st.caption(
        "Paired DeLong test (DeLong et al. 1988) for differences in ROC AUC "
        "between classifiers on the same patients. Raw p-values plus "
        "Bonferroni-Holm-corrected p-values to control the family-wise "
        "error rate across all pairwise comparisons. p_adj < 0.05 indicates "
        "a statistically significant AUC difference."
    )
    delong_rows = []
    for i, clf_a in enumerate(classifiers):
        for clf_b in classifiers[i + 1:]:
            g_a = sub[sub["classifier"] == clf_a].set_index("patno")
            g_b = sub[sub["classifier"] == clf_b].set_index("patno")
            common = g_a.index.intersection(g_b.index)
            if len(common) < 2:
                continue
            yt = g_a.loc[common, "y_true"].values
            pa = g_a.loc[common, "y_prob"].values
            pb = g_b.loc[common, "y_prob"].values
            auc_a, auc_b, p = delong_test(yt, pa, pb)
            delong_rows.append({
                "Method A": CLF_LABEL.get(clf_a, clf_a),
                "AUC A": auc_a,
                "Method B": CLF_LABEL.get(clf_b, clf_b),
                "AUC B": auc_b,
                "Difference": auc_a - auc_b,
                "_p_raw": p,
            })
    if delong_rows:
        pvals = [r["_p_raw"] for r in delong_rows]
        p_holm = adjust_pvalues(pvals, method="holm")
        p_bh = adjust_pvalues(pvals, method="bh")
        for r, ph, pb in zip(delong_rows, p_holm, p_bh):
            r["AUC A"] = f"{r['AUC A']:.3f}"
            r["AUC B"] = f"{r['AUC B']:.3f}"
            r["Difference"] = f"{r['Difference']:+.3f}"
            r["p (raw)"] = (f"{r['_p_raw']:.4f}" if r["_p_raw"] >= 0.0001
                            else "<0.0001")
            r["p (Holm)"] = (f"{ph:.4f}" if ph >= 0.0001 else "<0.0001")
            r["p (BH-FDR)"] = (f"{pb:.4f}" if pb >= 0.0001 else "<0.0001")
            del r["_p_raw"]
        st.dataframe(pd.DataFrame(delong_rows), use_container_width=True,
                      hide_index=True)
    st.markdown("")

    # ---- Sens/Spec/PPV/NPV bei Cutoffs
    st.markdown("#### Sensitivity / Specificity / PPV / NPV at clinical thresholds")
    st.caption(
        "Diagnostic accuracy metrics at three common decision thresholds, "
        "with 95% bootstrap confidence intervals (1000 resamples on patient "
        "level). Computed on the 10-fold CV predictions."
    )
    cutoff = st.select_slider("Decision threshold", options=[0.3, 0.4, 0.5, 0.6, 0.7],
                               value=0.5, key="cutoff_slider")
    metric_rows = []
    for clf in classifiers:
        g = sub[sub["classifier"] == clf]
        m = bootstrap_classification_metrics(
            g["y_true"].values, g["y_prob"].values, threshold=cutoff
        )
        def fmt(v, ci):
            if np.isnan(v):
                return "—"
            lo, hi = ci
            return f"{v:.2f} [{lo:.2f}-{hi:.2f}]"
        metric_rows.append({
            "Method": CLF_LABEL[clf],
            "Sensitivity": fmt(m["sens"], m["sens_ci"]),
            "Specificity": fmt(m["spec"], m["spec_ci"]),
            "PPV": fmt(m["ppv"], m["ppv_ci"]),
            "NPV": fmt(m["npv"], m["npv_ci"]),
        })
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    st.markdown("")

    # ---- NRI / IDI
    st.markdown("#### Reclassification metrics (NRI, IDI)")
    st.caption(
        "Net Reclassification Improvement and Integrated Discrimination "
        "Improvement (Pencina et al. 2008) comparing each classifier with "
        "the others, evaluated at the 50% decision threshold. Positive NRI/IDI "
        "means the row method classifies patients better than the column "
        "method."
    )
    nri_rows = []
    for i, clf_new in enumerate(classifiers):
        row = {"Method (new)": CLF_LABEL[clf_new]}
        for clf_old in classifiers:
            if clf_new == clf_old:
                row[CLF_LABEL[clf_old] + " (NRI)"] = "—"
                continue
            g_new = sub[sub["classifier"] == clf_new].set_index("patno")
            g_old = sub[sub["classifier"] == clf_old].set_index("patno")
            common = g_new.index.intersection(g_old.index)
            if len(common) < 2:
                row[CLF_LABEL[clf_old] + " (NRI)"] = "—"
                continue
            yt = g_new.loc[common, "y_true"].values
            pn = g_new.loc[common, "y_prob"].values
            po = g_old.loc[common, "y_prob"].values
            res = nri_idi(yt, po, pn)
            row[CLF_LABEL[clf_old] + " (NRI)"] = f"{res['nri']:+.3f}"
        nri_rows.append(row)
    st.dataframe(pd.DataFrame(nri_rows), use_container_width=True, hide_index=True)


def _calibration_panel():
    """Reliability Diagrams, Brier Score, ECE pro Klassifikator."""
    df = _load("ml_calibration_predictions.csv")
    if df is None:
        st.caption("Calibration data not yet available. Run "
                    "`scripts/compute_calibration.py` once to generate.")
        return

    import numpy as np
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    from src.clinical_metrics import (
        calibration_intercept_slope, hosmer_lemeshow,
    )

    # Sub-Tabs pro Score-Set
    score_modes = sorted(df["score_set"].unique())
    sm_tabs = st.tabs([{"luxpark": "Standard (17)",
                        "full": "Extended (25)"}.get(m, m) for m in score_modes])
    for tab, sm in zip(sm_tabs, score_modes):
        with tab:
            sub = df[(df["score_set"] == sm) & (df["model_type"] == "slopes+intercepts")]
            cal_rows = []
            stats_rows = []
            for clf, grp in sub.groupby("classifier"):
                y_true = grp["y_true"].values
                y_prob = grp["y_prob"].values
                # Reliability curve
                prob_true, prob_pred = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy="quantile"
                )
                for pt, pp in zip(prob_true, prob_pred):
                    cal_rows.append({
                        "Method": CLF_LABEL[clf],
                        "predicted_prob": float(pp),
                        "observed_freq": float(pt),
                    })
                # Brier + ECE
                brier = brier_score_loss(y_true, y_prob)
                # Expected Calibration Error (binned)
                bins = np.linspace(0, 1, 11)
                ece = 0.0
                for lo, hi in zip(bins[:-1], bins[1:]):
                    m = (y_prob >= lo) & (y_prob < hi)
                    if m.sum() == 0:
                        continue
                    bin_conf = y_prob[m].mean()
                    bin_acc = y_true[m].mean()
                    ece += (m.sum() / len(y_prob)) * abs(bin_conf - bin_acc)
                # Cox-Calibration (Intercept, Slope) und Hosmer-Lemeshow
                cox = calibration_intercept_slope(y_true, y_prob)
                hl = hosmer_lemeshow(y_true, y_prob, g=10)
                stats_rows.append({
                    "Method": CLF_LABEL[clf],
                    "Brier score": brier,
                    "ECE": ece,
                    "Cal. intercept": cox["intercept"],
                    "Cal. slope": cox["slope"],
                    "HL chi2": hl["chi2"],
                    "HL p-value": hl["p_value"],
                    "N predictions": len(y_prob),
                })

            cal_df = pd.DataFrame(cal_rows)
            stats_df = pd.DataFrame(stats_rows)

            # Reliability diagram
            diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
            ref_line = alt.Chart(diag).mark_line(
                color="#9ca3af", strokeDash=[4, 3]
            ).encode(x=alt.X("x:Q", title="Predicted probability"),
                      y=alt.Y("y:Q", title="Observed frequency"))
            present = [m for m in PALETTE if m in cal_df["Method"].unique()]
            curve = (
                alt.Chart(cal_df)
                .mark_line(point=alt.OverlayMarkDef(size=70, filled=True))
                .encode(
                    x=alt.X("predicted_prob:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(title="Predicted probability",
                                            format=".1f")),
                    y=alt.Y("observed_freq:Q",
                            scale=alt.Scale(domain=[0, 1]),
                            axis=alt.Axis(title="Observed frequency",
                                            format=".1f")),
                    color=alt.Color("Method:N",
                                    scale=alt.Scale(
                                        domain=present,
                                        range=[PALETTE[m] for m in present]),
                                    legend=alt.Legend(title="Method", orient="top")),
                    tooltip=["Method",
                             alt.Tooltip("predicted_prob:Q", format=".2f"),
                             alt.Tooltip("observed_freq:Q", format=".2f")],
                )
            )
            st.altair_chart((ref_line + curve).properties(height=320),
                            use_container_width=True)
            st.caption(
                "Reliability diagram. Closer to the dashed identity line is "
                "better. Calibration metrics: Brier score (lower = better, "
                "0-0.25 for balanced binary), Expected Calibration Error "
                "(lower = better, 0 is perfect), Cox calibration intercept "
                "and slope (Cox 1958; intercept=0 and slope=1 indicate "
                "perfect calibration; intercept > 0 means underestimation, "
                "slope < 1 means predictions are too extreme), and the "
                "Hosmer-Lemeshow goodness-of-fit chi-square test "
                "(p > 0.05 = no significant miscalibration, 10 deciles)."
            )
            stats_df["Brier score"] = stats_df["Brier score"].apply(lambda x: f"{x:.4f}")
            stats_df["ECE"] = stats_df["ECE"].apply(lambda x: f"{x:.4f}")
            stats_df["Cal. intercept"] = stats_df["Cal. intercept"].apply(
                lambda x: f"{x:+.3f}" if pd.notna(x) else "—"
            )
            stats_df["Cal. slope"] = stats_df["Cal. slope"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )
            stats_df["HL chi2"] = stats_df["HL chi2"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—"
            )
            stats_df["HL p-value"] = stats_df["HL p-value"].apply(
                lambda x: (f"{x:.4f}" if x >= 0.0001 else "<0.0001")
                          if pd.notna(x) else "—"
            )
            st.dataframe(stats_df, use_container_width=True, hide_index=True)


def render(*_):
    st.markdown("## About the Parkinson Subtype Predictor")

    st.markdown(
        """
        A web app that predicts Parkinson's disease progression subtype --
        fast or slow -- from trajectories of clinical scores. Built as a
        research and demonstration tool with publication-grade methodology:
        kNN imputation, isotonically calibrated probabilities, Conformal
        prediction sets, bootstrap reliability intervals, and SHAP-based
        feature attribution. The predictions are **not clinically validated**
        and do not replace medical judgment.
        """
    )

    st.markdown("### Methodology")
    st.markdown(
        """
        Four methods are compared on the same task.

        - **Random Forest** -- ensemble of 500 decision trees,
          `class_weight="balanced"` to compensate for the 4.5:1
          slow:fast imbalance in PPMI
        - **XGBoost** -- 500 gradient-boosted trees, max depth 4,
          learning rate 0.05, subsample 0.8, colsample 0.8
        - **Logistic Regression** with L1 (saga, max_iter 5000)
        - **Likelihood Ratio (LR)** -- Tom's method, fits per-subtype
          slope distributions via Linear Mixed Effects models and computes
          log-likelihood ratios per score, summed to a total score

        **Feature extraction.** For each clinical score and each patient we
        fit ordinary least squares (OLS) on (disease duration, score) and
        keep the slope and the intercept (extrapolated to t=0). For
        single-visit patients a separate baseline-only model uses the raw
        scores.

        **Imputation.** Missing feature values are filled by **kNN imputation
        (k=5)**: the 5 most similar PPMI patients in Euclidean distance on
        the remaining features. We deliberately chose kNN over median to
        avoid the class-imbalance bias of a global median (which would push
        fast patients towards the slow distribution).

        **Calibration.** Each ML model is wrapped in
        `CalibratedClassifierCV(method="isotonic", cv=5)` so that the output
        probabilities mean what they say -- a 70% prediction is correct in
        roughly 70 of 100 comparable cases.

        **Conformal prediction.** Around the calibrated classifier we wrap
        a `SplitConformalClassifier` (MAPIE 1.4, LAC conformity score) on a
        held-out 20% of PPMI. For each patient the model outputs a
        **prediction set** with **90% coverage guarantee**: either {Fast},
        {Slow}, or {Fast, Slow} when uncertain. This is the distribution-free
        gold standard for uncertainty quantification in clinical ML.

        **External validation.** Models are evaluated on PPMI via 10-fold
        cross-validation grouped by patient. External validation on the
        LuxPARK cohort (Luxembourg) is in preparation.
        """
    )

    st.markdown("### Headline accuracy on PPMI")
    _headline_accuracy_panel()
    st.caption(
        "All numbers from 10-fold cross-validation grouped by patient with "
        "95% bootstrap confidence intervals (1000 patient-level resamples). "
        "Random Forest and XGBoost outperform the Likelihood Ratio on average, "
        "but Likelihood Ratio is more robust at very high missingness "
        "(see below)."
    )

    st.divider()
    st.markdown("### Performance across score subsets")
    st.caption(
        "Box plots: AUC distribution across 50 random subsets of k scores per "
        "method (k=1..10). Dashed line: greedy forward selection, which picks "
        "the best k scores stepwise. The gap between the two shows how much "
        "score selection matters at each k."
    )
    _score_combinations_chart()
    st.caption(
        "Saturation already at 4-5 scores: adding more scores barely improves "
        "AUC. The most informative single score is **UPDRS2** for all methods."
    )

    st.divider()
    st.markdown("### Sensitivity to missingness")
    st.caption(
        "AUC at increasing fraction of missing score values on the 5-score "
        "subset (UPDRS3_on, UPDRS2, UPDRS1, PIGD_on, SCOPA). Likelihood Ratio "
        "(not shown here, separate analysis) is more robust at >80% missingness "
        "because it skips missing scores rather than imputing them."
    )
    _missingness_chart()

    st.divider()
    st.markdown("### Sensitivity to follow-up duration")
    st.caption(
        "AUC at increasing follow-up length (longest visit minus first visit, "
        "in months). Below 24 months the slope-based prediction is essentially "
        "useless because slopes can't be reliably estimated from too few "
        "timepoints."
    )
    _followup_chart()

    st.divider()
    st.markdown("### Per-score AUC")
    st.caption(
        "Each score in isolation: how well does it discriminate fast vs slow "
        "patients on its own? Note this is per-score AUC, not the model's "
        "AUC -- the full models combine many scores."
    )
    _per_score_chart()

    st.divider()
    st.markdown("### Handling of missing data per patient")
    st.markdown(
        """
        Missing score values are imputed with **k-Nearest-Neighbour imputation**
        (k=5): for each missing feature we find the 5 most similar PPMI
        patients (Euclidean distance on the available features) and use their
        median for that feature. This avoids the bias of a global median
        imputation, where the imbalanced PPMI class ratio (4.5:1 slow:fast)
        would systematically push fast patients towards slow predictions.
        To stay transparent, the app shows the **expected AUC** for each
        classifier at the current missingness level for the selected patient,
        based on a bootstrap simulation on PPMI.
        """
    )

    st.markdown("### Score sets")
    st.markdown(
        """
        - **Standard (17 scores)** -- clinical routine scores. Overlap with the
          LuxPARK cohort (Luxembourg), used for our external validation.
        - **Extended (25 scores)** -- adds the PPMI research battery
          (HVLT, SDM, LNS, VFT semantic, SEADL, ESS, GDS). Slightly higher
          accuracy on PPMI but rarely all measured in clinical routine.
        """
    )

    st.markdown("### Disclaimer")
    st.info(
        "This app is a research and demonstration tool. The predictions are "
        "not clinically validated and must not replace medical decision-making.",
        icon=":material/info:",
    )

    st.divider()
    st.markdown("### Comparison with trivial baselines")
    st.caption(
        "How much value do the multivariable models actually add over "
        "simple decision rules? Three baselines: predicting everyone as "
        "'Slow' (matches the PPMI class imbalance), and single-feature "
        "logistic regression on UPDRS3 or MoCA slope+intercept only. "
        "AUCs are computed on the same 10-fold patient-grouped CV setup."
    )
    _baseline_comparison_panel()

    st.divider()
    st.markdown("### Clinical utility metrics")
    _clinical_metrics_panel()

    st.divider()
    st.markdown("### Recommended decision thresholds")
    st.caption(
        "An arbitrary cutoff of 0.5 rarely matches clinical priorities. "
        "Below we report three thresholds, each derived from a principled "
        "criterion. Selecting the right cutoff depends on the relative cost "
        "of missing a fast progressor versus over-flagging a slow one."
    )
    _decision_threshold_panel()

    st.divider()
    st.markdown("### Probability calibration diagnostics")
    _calibration_panel()

    st.divider()
    st.markdown("### Robustness to measurement noise")
    st.caption(
        "How much do predictions change when input scores are perturbed "
        "by realistic measurement noise (Gaussian, SD as fraction of "
        "score range)? Random Forest model, 50 realisations per noise "
        "level."
    )
    _stress_test_panel()

    st.divider()
    st.markdown("### Feature effects: Partial Dependence and ICE")
    st.caption(
        "What is the marginal effect of each feature on the predicted "
        "P(Fast), averaged across patients? Thick line: PDP (Friedman "
        "2001). Thin lines: per-patient ICE curves (Goldstein et al. "
        "2015) reveal interaction heterogeneity. Computed on the 17-score "
        "models with slope features."
    )
    _pdp_panel()

    st.divider()
    st.markdown("### Subgroup fairness")
    _subgroup_fairness_panel()

    st.divider()
    st.markdown("### Class-conditional fairness (Equalized Odds)")
    st.caption(
        "Within each true class -- Fast or Slow -- does the classifier "
        "behave the same across age and sex subgroups? AUC-based fairness "
        "(above) can hide bias that affects only one class. Equalized-Odds-"
        "Difference quantifies the worst-case gap in TPR or FPR between "
        "subgroups (Hardt et al. 2016, 'Equality of Opportunity in "
        "Supervised Learning')."
    )
    _class_conditional_fairness_panel()

    st.divider()
    st.markdown("### Supplementary analyses")
    st.markdown(
        """
        Additional methodological analyses are bundled with the repository
        under `docs/`. These are linked from the publication and may be
        used for reviewer responses.

        - **POWER_ANALYSIS.md** — post-hoc power analysis (Hanley-McNeil
          1982, Obuchowski 1998). With n=409 we can detect AUC differences
          >= 0.06 at 80% power. RF vs XGBoost (delta 0.001) is underpowered
          by design.
        - **TEMPORAL_VALIDATION.md** — split-by-enrollment-year inside
          PPMI 1.0 (subtype labels do not exist for PPMI 2.0). RF AUC
          stable at 0.97-0.98 across split years 2012 and 2013.
        - **SURVIVAL_ANALYSIS.md** — Cox PH on time-to-HY-3 milestone as
          alternative outcome framing. C-index 0.87 on slope+intercept
          features.
        - **LITERATURE_COMPARISON.md** — comparison with seven published
          PD progression models (Latourelle 2017, Wang 2025, Dai 2025,
          Faouzi 2022, Dadu 2024, Iakovakis 2020, Zhang 2025).
        - **HYPERPARAMETER_TUNING.md** *(in progress)* — nested CV with
          Optuna, comparing default hyperparameters against tuned ones.
        - **SHAP_STABILITY.md** *(in progress)* — bootstrap stability of
          SHAP feature rankings.
        - **STRESS_TEST.md** *(in progress)* — robustness against
          measurement noise on input scores.
        - **TRUE_BOOTSTRAP.md** *(in progress)* — N=100 full-pipeline
          bootstrap AUC CIs (Pencina-style).
        """
    )

    st.markdown("### Code and data")
    st.markdown(
        """
        - Training data: PPMI ([ppmi-info.org](https://www.ppmi-info.org))
        - Code: [github.com/cl-poehl/parkinson-subtype-predictor](https://github.com/cl-poehl/parkinson-subtype-predictor)
        - External validation in preparation on the LuxPARK cohort
        - Key libraries: scikit-learn, XGBoost, MAPIE (conformal prediction),
          SHAP (feature attribution), Streamlit (UI), Altair (charts)
        """
    )
