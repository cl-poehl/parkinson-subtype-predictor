"""Shared helpers for demo.py and batch.py."""
import io
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

from src.constants import SCORE_RANGES, get_model_paths
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all
from src.shap_utils import get_shap


def build_template(active_scores):
    cols = ["patno", "disease_duration"] + active_scores
    sample = []
    for v, t in enumerate([0, 12, 24]):
        row = {"patno": "P001", "disease_duration": t}
        for s in active_scores:
            _, _, default = SCORE_RANGES[s]
            row[s] = default
        sample.append(row)
    return pd.DataFrame(sample, columns=cols).to_csv(index=False)


def run_predictions(df_in, score_mode, active_scores):
    """Returns (preds_df, shap_context). shap_context fasst Features und Modelle
    pro Modelltyp zusammen, damit der Cohort-SHAP nachher gerechnet werden kann."""
    df = df_in.copy()
    for s in active_scores:
        if s not in df.columns:
            df[s] = pd.NA

    visits_per_patient = df.groupby("patno").size()
    multi_ids = visits_per_patient[visits_per_patient >= 2].index
    single_ids = visits_per_patient[visits_per_patient == 1].index

    out = []
    shap_ctx = {}
    if len(multi_ids) > 0:
        multi = df[df["patno"].isin(multi_ids)]
        feats = extract_slope_intercept(multi, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=2))
        if models:
            preds = predict_all(models, feats)
            preds["model_type"] = "slope"
            out.append(preds)
            shap_ctx["slope"] = (feats, models)

    if len(single_ids) > 0:
        single = df[df["patno"].isin(single_ids)]
        feats = extract_baseline(single, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=1))
        if models:
            preds = predict_all(models, feats)
            preds["model_type"] = "baseline"
            out.append(preds)
            shap_ctx["baseline"] = (feats, models)

    if not out:
        return None, {}
    full = pd.concat(out).reset_index().rename(columns={"index": "patno"})
    return full, shap_ctx


def _signed_importance_bar(sv, max_display=12):
    """Horizontale Bars: pro Feature der mittlere SHAP-Wert. Positive nach rechts
    (Fast), negative nach links (Slow), eingefaerbt nach Richtung."""
    import numpy as np
    mean_shap = sv.values.mean(axis=0)
    abs_mean = np.abs(mean_shap)
    order = np.argsort(abs_mean)[::-1][:max_display]
    feat_names = [sv.feature_names[i] for i in order]
    means = mean_shap[order]

    df = pd.DataFrame({"feature": feat_names, "mean_shap": means})
    df["direction"] = df["mean_shap"].apply(lambda x: "Fast" if x >= 0 else "Slow")

    # Achsen-Grenzen symmetrisch
    bound = max(abs_mean.max() * 1.15, 0.01) if len(abs_mean) else 0.01

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort=feat_names,
                    axis=alt.Axis(title=None, labelLimit=400)),
            x=alt.X("mean_shap:Q",
                    scale=alt.Scale(domain=[-bound, bound]),
                    axis=alt.Axis(title="Mean SHAP value   (← Slow      Fast →)")),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Slow", "Fast"], range=["#3b82f6", "#ef4444"]),
                legend=None,
            ),
            tooltip=["feature", alt.Tooltip("mean_shap:Q", format=".3f"), "direction"],
        )
        .properties(height=max(28 * len(feat_names), 200))
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
    st.altair_chart(chart + rule, use_container_width=True)


def render_results(preds, source_name, shap_ctx=None, score_mode="luxpark"):
    clf_cols = [c for c in preds.columns if c not in ("patno", "model_type")]
    consensus = preds[clf_cols].mean(axis=1)
    preds = preds.assign(consensus=consensus,
                          klasse=consensus.apply(lambda x: "Fast" if x >= 0.5 else "Slow"))

    n = len(preds)
    n_fast = int((preds["consensus"] >= 0.5).sum())
    n_slow = n - n_fast
    mean_conf = preds["consensus"].mean()

    st.markdown(f"### Results  \n*Source: {source_name}*")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients", n)
    m2.metric("Fast progression", n_fast)
    m3.metric("Slow progression", n_slow)
    m4.metric("Mean P(Fast)", f"{mean_conf*100:.0f}%")
    st.markdown("")

    if HAS_ALTAIR and n <= 200:
        chart_df = preds.sort_values("consensus").assign(idx=range(n))
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("patno:N", sort=chart_df["patno"].tolist(),
                        axis=alt.Axis(labelAngle=-40, title="Patient")),
                y=alt.Y("consensus:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%", title="P(Fast progression)")),
                color=alt.Color(
                    "klasse:N",
                    scale=alt.Scale(domain=["Slow", "Fast"], range=["#3b82f6", "#ef4444"]),
                    legend=alt.Legend(title="Class"),
                ),
                tooltip=["patno", alt.Tooltip("consensus:Q", format=".1%"), "klasse"],
            )
            .properties(height=240)
        )
        rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
            color="gray", strokeDash=[4, 4]
        ).encode(y="y:Q")
        st.altair_chart(chart + rule, use_container_width=True)

    pretty = preds.copy()
    for c in clf_cols:
        pretty[c] = pretty[c].apply(lambda x: f"{x*100:.1f}%")
    pretty["consensus"] = pretty["consensus"].apply(lambda x: f"{x*100:.1f}%")
    pretty = pretty.rename(columns={"consensus": "Consensus", "klasse": "Class",
                                      "model_type": "Model"})
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    buf = io.StringIO()
    preds.drop(columns=["klasse"]).to_csv(buf, index=False)
    st.download_button(
        "Download results as CSV", buf.getvalue(),
        file_name="subtype_predictions.csv", mime="text/csv",
    )

    # ---- SHAP: cohort-level feature importance per classifier
    if shap_ctx:
        st.markdown("### Feature importance across the cohort")
        st.caption(
            "For each feature, the average SHAP value across all patients. "
            "Bars to the right push the prediction towards **Fast progression** on "
            "average, bars to the left towards **Slow progression**. The further out, "
            "the larger the impact. Features are sorted by absolute impact."
        )
        clf_names = clf_cols
        clf_tabs = st.tabs(clf_names)
        for tab, clf_name in zip(clf_tabs, clf_names):
            with tab:
                sv = None
                for mtype, (feats, models) in shap_ctx.items():
                    if clf_name not in models or len(feats) == 0:
                        continue
                    sv = get_shap(models[clf_name], feats,
                                  f"{score_mode}_{clf_name}_{mtype}")
                    if sv is not None:
                        break
                if sv is None:
                    st.caption("No SHAP plot available.")
                    continue
                _signed_importance_bar(sv, max_display=12)
