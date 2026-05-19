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


def render(*_):
    st.markdown("## About the Parkinson Subtype Predictor")

    st.markdown(
        """
        A web app that predicts Parkinson's disease progression subtype --
        fast or slow -- from the trajectories of clinical scores. It is a
        research and demonstration tool. The predictions are **not clinically
        validated** and do not replace medical judgment.
        """
    )

    st.markdown("### How the models work")
    st.markdown(
        """
        Four methods compete in this app on the same task.

        - **Random Forest** -- ensemble of 500 decision trees
        - **XGBoost** -- gradient-boosted trees
        - **Logistic Regression** with L1 regularization
        - **Likelihood Ratio** -- Tom's method, fits per-subtype slope
          distributions via Linear Mixed Effects and computes a log-likelihood
          ratio per score

        For Random Forest / XGBoost / Logistic Regression we extract two
        features per clinical score per patient: the **slope** of a linear
        regression across all visits and the **intercept** at the time of
        diagnosis. Patients with only one visit are routed to separate
        baseline-only models. All three ML models are isotonically calibrated
        via 5-fold cross-validation, so the output probabilities can be
        interpreted directly (70% means roughly 7 out of 10 comparable cases
        turn out to be of that class).
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
        Missing score values are imputed with the median of the training
        cohort. To stay transparent, the app shows the **expected AUC** for
        each classifier at the current missingness and follow-up level for the
        selected patient, based on a 2D simulation (missingness × follow-up)
        on the PPMI dataset.
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

    st.markdown("### Code and data")
    st.markdown(
        """
        - Training data: PPMI ([ppmi-info.org](https://www.ppmi-info.org))
        - Code: [github.com/cl-poehl/parkinson-subtype-predictor](https://github.com/cl-poehl/parkinson-subtype-predictor)
        - External validation in preparation on the LuxPARK cohort
        """
    )
