"""Batch-Modus: CSV hochladen und Predictions zurueckgeben."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pandas as pd
import streamlit as st

from src.constants import (
    SCORE_LABELS, SCORE_RANGES, SCORES_LUXPARK,
    get_score_set, get_model_paths,
)
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all

st.set_page_config(page_title="Batch Upload", layout="wide")
st.title("Batch-Vorhersage per CSV")

# Sidebar Toggle
with st.sidebar:
    st.markdown("##### Konfiguration")
    score_mode = st.radio(
        "Score-Set",
        options=["luxpark", "full"],
        format_func=lambda x: {"luxpark": "17 Scores (LuxPARK-kompatibel)",
                                "full": "25 Scores (voller PPMI-Umfang)"}[x],
        key="score_mode",
    )
    active_scores = get_score_set(score_mode)
    st.caption(f"Aktiv: **{len(active_scores)} Scores**")

st.markdown(
    """
Lade eine CSV hoch mit einer Zeile pro Visit. Mehrere Visits pro Patient sind moeglich.
Patienten mit nur einer Visit werden automatisch mit dem Single-Visit-Modell gerechnet,
der Rest mit dem Slope-Modell.

**Spalten.**

- `patno` -- Patienten-ID, frei waehlbar (z.B. P001, P002). Wird nur genutzt, um Visits
  demselben Patienten zuzuordnen.
- `disease_duration` -- Monate seit Diagnose.
- alle weiteren Spalten sind die Scores (siehe Vorlage).
"""
)


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


DEMO_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "demo_patients.csv")

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    st.download_button(
        "Leere Vorlage",
        data=build_template(),
        file_name=f"vorlage_{score_mode}.csv",
        mime="text/csv",
        help=f"CSV-Vorlage mit den {len(active_scores)} aktiven Scores und einem "
             f"Beispielpatienten (P001) mit 3 Visits.",
    )
with col_b:
    demo_bytes = ""
    if os.path.exists(DEMO_CSV_PATH):
        with open(DEMO_CSV_PATH) as fh:
            demo_bytes = fh.read()
    st.download_button(
        "Synthetische Demo-Daten",
        data=demo_bytes,
        file_name="demo_patients.csv",
        mime="text/csv",
        help="Sechs synthetische Patienten (3 Fast, 3 Slow) mit je 3 Visits. "
             "Enthaelt die 17 LuxPARK-kompatiblen Scores, im 25-Modus werden die "
             "zusaetzlichen Scores als fehlend behandelt.",
    )
with col_c:
    st.caption("Demo-Daten enthalten 6 erfundene Patienten (3 Fast, 3 Slow). "
               "Damit kann man die App testen, ohne echte Daten zu brauchen.")

uploaded = st.file_uploader("CSV hochladen", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"Eingelesen: {len(df)} Zeilen, {df['patno'].nunique()} Patienten")
    st.dataframe(df.head())

    # Fehlende Spalten aus der CSV durch NaN ergaenzen
    for s in active_scores:
        if s not in df.columns:
            df[s] = pd.NA

    if st.button("Predictions berechnen", type="primary"):
        visits_per_patient = df.groupby("patno").size()
        multi_visit_ids = visits_per_patient[visits_per_patient >= 2].index
        single_visit_ids = visits_per_patient[visits_per_patient == 1].index

        results = []
        if len(multi_visit_ids) > 0:
            multi = df[df["patno"].isin(multi_visit_ids)]
            feats = extract_slope_intercept(multi, active_scores)
            models = load_models(get_model_paths(score_mode, n_visits=2))
            if models:
                preds = predict_all(models, feats)
                preds["model_type"] = "slope"
                results.append(preds)

        if len(single_visit_ids) > 0:
            single = df[df["patno"].isin(single_visit_ids)]
            feats = extract_baseline(single, active_scores)
            models = load_models(get_model_paths(score_mode, n_visits=1))
            if models:
                preds = predict_all(models, feats)
                preds["model_type"] = "baseline"
                results.append(preds)

        if not results:
            st.error("Keine Modelle gefunden. Trainings-Skript muss noch laufen gelassen werden.")
        else:
            full = pd.concat(results).reset_index().rename(columns={"index": "patno"})
            st.write(f"Predictions fuer {len(full)} Patienten:")
            st.dataframe(full)

            buf = io.StringIO()
            full.to_csv(buf, index=False)
            st.download_button("Ergebnis-CSV herunterladen", buf.getvalue(),
                               file_name="subtype_predictions.csv", mime="text/csv")
