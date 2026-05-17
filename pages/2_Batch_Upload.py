"""Batch-Vorhersage, modernes Layout mit Tabs und visualisierten Ergebnissen."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pandas as pd
import streamlit as st

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

from src.constants import (
    SCORE_LABELS, SCORE_RANGES, SCORES_LUXPARK,
    get_score_set, get_model_paths,
)
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all

st.set_page_config(page_title="Batch Upload", layout="wide")

hcol1, hcol2 = st.columns([3, 2], vertical_alignment="bottom")
with hcol1:
    st.title("Batch-Vorhersage")
    st.caption("Sage Subtypen fuer mehrere Patienten gleichzeitig vorher.")
with hcol2:
    score_mode = st.segmented_control(
        "Score-Set",
        options=["luxpark", "full"],
        format_func=lambda x: {"luxpark": "17 Scores (LuxPARK)",
                                "full": "25 Scores (volles PPMI)"}[x],
        default="luxpark",
        key="score_mode",
        help="17 Scores ist die LuxPARK-kompatible Schnittmenge. 25 Scores nutzt "
             "zusaetzlich die PPMI-Batterie.",
    )
active_scores = get_score_set(score_mode)

DEMO_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "demo_patients.csv")


def build_template():
    cols = ["patno", "disease_duration"] + active_scores
    sample = []
    for v, t in enumerate([0, 12, 24]):
        row = {"patno": "P001", "disease_duration": t}
        for s in active_scores:
            _, _, default = SCORE_RANGES[s]
            row[s] = default
        sample.append(row)
    return pd.DataFrame(sample, columns=cols).to_csv(index=False)


def run_predictions(df_in):
    """Predictions berechnen, Multi-Visit und Single-Visit getrennt."""
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
    """Ergebnis-Sektion: Summary, Chart, Tabelle, Download."""
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

    # Chart: sortierte Patienten nach Konsens-Wahrscheinlichkeit
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

    # Detail-Tabelle, schoen formatiert
    pretty = preds.copy()
    for c in clf_cols:
        pretty[c] = pretty[c].apply(lambda x: f"{x*100:.1f}%")
    pretty["consensus"] = pretty["consensus"].apply(lambda x: f"{x*100:.1f}%")
    pretty = pretty.rename(columns={"consensus": "Konsens", "klasse": "Klasse",
                                      "model_type": "Modelltyp"})
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    # Download
    buf = io.StringIO()
    preds.drop(columns=["klasse"]).to_csv(buf, index=False)
    st.download_button(
        "Ergebnisse als CSV", buf.getvalue(),
        file_name="subtype_predictions.csv", mime="text/csv",
    )


# ---- Tabs: Demo vs. Eigene Daten
tab_demo, tab_csv = st.tabs(["Demo-Daten", "Eigene CSV"])

with tab_demo:
    dcol1, dcol2 = st.columns([3, 2])
    with dcol1:
        st.markdown("**Sechs synthetische Patienten zum schnellen Testen.**")
        st.write(
            "Drei Fast- und drei Slow-Progressors, jeweils mit drei Visits ueber "
            "vier Jahre. Die Werte sind plausibel gewaehlt, aber komplett erfunden. "
            "Damit kannst du die Webapp ausprobieren, ohne eigene Daten zu haben."
        )
        run_demo = st.button("Demo durchrechnen", type="primary", use_container_width=True)
    with dcol2:
        # Vorschau der ersten Zeilen
        if os.path.exists(DEMO_CSV_PATH):
            preview = pd.read_csv(DEMO_CSV_PATH).head(6)
            st.caption("Vorschau der Demo-Daten")
            st.dataframe(preview[["patno", "disease_duration", "UPDRS3_on",
                                    "UPDRS1", "UPDRS2", "MOCA"]],
                          use_container_width=True, hide_index=True, height=210)

    if run_demo:
        if not os.path.exists(DEMO_CSV_PATH):
            st.error(f"Demo-CSV nicht gefunden unter {DEMO_CSV_PATH}.")
        else:
            with st.spinner("Berechne Predictions ..."):
                preds = run_predictions(pd.read_csv(DEMO_CSV_PATH))
            if preds is None:
                st.error("Keine Modelle gefunden.")
            else:
                render_results(preds, "Demo-Patienten")

with tab_csv:
    st.markdown(
        "**Lade eine CSV mit deinen eigenen Patientendaten hoch.** Eine Zeile pro Visit, "
        "mehrere Visits pro Patient sind moeglich. Fehlende Score-Werte sind erlaubt."
    )

    tcol1, tcol2 = st.columns([2, 3])
    with tcol1:
        st.download_button(
            "Leere CSV-Vorlage",
            data=build_template(),
            file_name=f"vorlage_{score_mode}.csv",
            mime="text/csv",
            use_container_width=True,
            help=f"Vorlage mit den {len(active_scores)} aktiven Score-Spalten und einem "
                 f"Beispielpatient (P001) zum Orientieren.",
        )
        st.caption(
            "Spalten: `patno`, `disease_duration`, plus die Scores. "
            "Vorlage runterladen, in Excel ausfuellen, hier wieder hochladen."
        )
    with tcol2:
        uploaded = st.file_uploader("CSV hochladen", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Datei gelesen: {len(df)} Zeilen, {df['patno'].nunique()} Patienten")
        with st.expander("Daten ansehen"):
            st.dataframe(df, use_container_width=True, hide_index=True)

        with st.spinner("Berechne Predictions ..."):
            preds = run_predictions(df)
        if preds is None:
            st.error("Keine Modelle gefunden.")
        else:
            render_results(preds, uploaded.name)
