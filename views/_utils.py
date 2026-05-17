"""Geteilte Helper fuer demo.py und batch.py."""
import io
import pandas as pd
import streamlit as st

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

from src.constants import SCORE_RANGES, get_model_paths
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all


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
    df = df_in.copy()
    for s in active_scores:
        if s not in df.columns:
            df[s] = pd.NA

    visits_per_patient = df.groupby("patno").size()
    multi_ids = visits_per_patient[visits_per_patient >= 2].index
    single_ids = visits_per_patient[visits_per_patient == 1].index

    out = []
    if len(multi_ids) > 0:
        multi = df[df["patno"].isin(multi_ids)]
        feats = extract_slope_intercept(multi, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=2))
        if models:
            preds = predict_all(models, feats)
            preds["model_type"] = "slope"
            out.append(preds)

    if len(single_ids) > 0:
        single = df[df["patno"].isin(single_ids)]
        feats = extract_baseline(single, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=1))
        if models:
            preds = predict_all(models, feats)
            preds["model_type"] = "baseline"
            out.append(preds)

    if not out:
        return None
    return pd.concat(out).reset_index().rename(columns={"index": "patno"})


def render_results(preds, source_name):
    clf_cols = [c for c in preds.columns if c not in ("patno", "model_type")]
    consensus = preds[clf_cols].mean(axis=1)
    preds = preds.assign(consensus=consensus,
                          klasse=consensus.apply(lambda x: "Fast" if x >= 0.5 else "Slow"))

    n = len(preds)
    n_fast = int((preds["consensus"] >= 0.5).sum())
    n_slow = n - n_fast
    mean_conf = preds["consensus"].mean()

    st.markdown(f"### Ergebnisse  \n*Quelle: {source_name}*")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patienten", n)
    m2.metric("Fast Progression", n_fast)
    m3.metric("Slow Progression", n_slow)
    m4.metric("Mittlere Fast-Wahrscheinlichkeit", f"{mean_conf*100:.0f}%")
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
                        axis=alt.Axis(format="%", title="P(Fast Progression)")),
                color=alt.Color(
                    "klasse:N",
                    scale=alt.Scale(domain=["Slow", "Fast"], range=["#3b82f6", "#ef4444"]),
                    legend=alt.Legend(title="Klassifikation"),
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
    pretty = pretty.rename(columns={"consensus": "Konsens", "klasse": "Klasse",
                                      "model_type": "Modelltyp"})
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    buf = io.StringIO()
    preds.drop(columns=["klasse"]).to_csv(buf, index=False)
    st.download_button(
        "Ergebnisse als CSV", buf.getvalue(),
        file_name="subtype_predictions.csv", mime="text/csv",
    )
