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
                stats_rows.append({
                    "Method": CLF_LABEL[clf],
                    "Brier score": brier,
                    "ECE": ece,
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
                "better. Brier score (lower = better, range 0-0.25 for "
                "balanced binary) and Expected Calibration Error (lower = "
                "better, ECE=0 is perfect) below."
            )
            stats_df["Brier score"] = stats_df["Brier score"].apply(lambda x: f"{x:.4f}")
            stats_df["ECE"] = stats_df["ECE"].apply(lambda x: f"{x:.4f}")
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
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Random Forest", "AUC 0.95")
    g2.metric("XGBoost", "AUC 0.95")
    g3.metric("Logistic Regression", "AUC 0.88")
    g4.metric("Likelihood Ratio", "AUC 0.91")
    st.caption(
        "All numbers from 10-fold cross-validation grouped by patient. "
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
    st.markdown("### Probability calibration diagnostics")
    _calibration_panel()

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
