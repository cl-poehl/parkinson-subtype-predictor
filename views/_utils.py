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


def patient_shap_bar(sv, patient_idx=0, max_display=None):
    """Horizontaler Bar-Chart der SHAP-Beitraege fuer einen einzelnen Patienten.
    max_display=None zeigt alle Features."""
    import numpy as np
    values = sv.values[patient_idx]
    abs_v = np.abs(values)
    order = np.argsort(abs_v)[::-1]
    if max_display is not None:
        order = order[:max_display]
    feat_names = [sv.feature_names[i] for i in order]
    vals = values[order]

    df = pd.DataFrame({"feature": feat_names, "shap": vals})
    df["direction"] = df["shap"].apply(lambda x: "Fast" if x >= 0 else "Slow")
    bound = max(abs_v.max() * 1.15, 0.01) if len(abs_v) else 0.01

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort=feat_names,
                    axis=alt.Axis(title=None, labelLimit=400)),
            x=alt.X("shap:Q",
                    scale=alt.Scale(domain=[-bound, bound]),
                    axis=alt.Axis(title="SHAP value   (← Slow      Fast →)")),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Slow", "Fast"], range=["#3b82f6", "#ef4444"]),
                legend=None,
            ),
            tooltip=["feature", alt.Tooltip("shap:Q", format=".3f"), "direction"],
        )
        .properties(height=max(26 * len(feat_names), 200))
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

    # Uebersichts-Diagramm: pro Patient die P(Fast) aller drei Modelle als
    # eigene Punkte, sortiert nach Konsens.
    if HAS_ALTAIR and n <= 200:
        long_df = preds.melt(
            id_vars=["patno", "consensus", "klasse"],
            value_vars=clf_cols,
            var_name="Model",
            value_name="prob",
        )
        # Patient-Reihenfolge: nach Konsens absteigend, damit Fast links und Slow rechts? Lieber andersrum, klassische Lesung "links niedrig, rechts hoch"
        patno_order = preds.sort_values("consensus")["patno"].astype(str).tolist()
        long_df["patno"] = long_df["patno"].astype(str)

        st.caption(
            "Each patient column shows the probability of Fast progression "
            "predicted by all three models. Patients are sorted by consensus, "
            "from most Slow on the left to most Fast on the right. Dashed line "
            "at 50% is the classification threshold."
        )

        points = (
            alt.Chart(long_df)
            .mark_point(filled=True, size=90, opacity=0.85)
            .encode(
                x=alt.X("patno:N", sort=patno_order,
                        axis=alt.Axis(labelAngle=-40, title="Patient")),
                y=alt.Y("prob:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%", title="P(Fast progression)")),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(
                        domain=["Random Forest", "XGBoost", "Logistic Regression"],
                        range=["#10b981", "#f97316", "#6366f1"],
                    ),
                    legend=alt.Legend(title="Model", orient="top"),
                ),
                shape=alt.Shape("Model:N", legend=None),
                tooltip=["patno", "Model", alt.Tooltip("prob:Q", format=".1%")],
            )
        )
        threshold = (
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(color="gray", strokeDash=[5, 5])
            .encode(y="y:Q")
        )
        st.altair_chart((points + threshold).properties(height=300),
                        use_container_width=True)
        st.markdown("")

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

    # ---- SHAP: per Patient ausgewaehlt
    if shap_ctx:
        st.markdown("### Why this prediction? Per-patient explanation")
        st.caption(
            "Pick a patient to see which features pushed the model towards "
            "**Fast** (red, right) or **Slow** (blue, left). The further out from "
            "zero, the larger the influence."
        )

        # Welcher Patient ist in welcher mtype-Gruppe?
        patient_lookup = {}
        for mtype, (feats, models) in shap_ctx.items():
            for pos, patno in enumerate(feats.index):
                patient_lookup[str(patno)] = (mtype, pos)

        ordered_ids = list(preds["patno"].astype(str).unique())
        selected = st.selectbox("Patient", options=ordered_ids,
                                 key=f"shap_patient_{source_name}")
        sel_row = preds[preds["patno"].astype(str) == selected].iloc[0]
        sel_consensus = float(sel_row["consensus"])
        sel_class = "Fast" if sel_consensus >= 0.5 else "Slow"
        sel_color = "#ef4444" if sel_class == "Fast" else "#3b82f6"

        st.markdown(
            f"<small>Consensus for **{selected}**: "
            f"<b style='color:{sel_color}'>{sel_consensus*100:.1f}% Fast</b> "
            f"({sel_class} progression)</small>",
            unsafe_allow_html=True,
        )

        mtype, patient_idx = patient_lookup.get(selected, (None, None))
        if mtype is None:
            st.caption("No SHAP context for this patient.")
        else:
            feats, models = shap_ctx[mtype]
            clf_tabs = st.tabs(clf_cols)
            for tab, clf_name in zip(clf_tabs, clf_cols):
                with tab:
                    if clf_name not in models:
                        st.caption("Model not available.")
                        continue
                    sv = get_shap(models[clf_name], feats,
                                  f"{score_mode}_{clf_name}_{mtype}")
                    if sv is None:
                        continue
                    patient_shap_bar(sv, patient_idx=patient_idx, max_display=None)
