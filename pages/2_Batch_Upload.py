"""Batch-Modus: CSV hochladen und Predictions zurueckgeben."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pandas as pd
import streamlit as st

from src.constants import SCORE_LABELS, SCORE_RANGES, MODEL_FILES, MODEL_FILES_BASELINE
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all

st.set_page_config(page_title="Batch Upload", layout="wide")
st.title("Batch-Vorhersage per CSV")

st.markdown(
    """
Lade eine CSV hoch mit einer Zeile pro Visit. Mehrere Visits pro Patient sind moeglich.
Patienten mit nur einer Visit werden automatisch mit dem Single-Visit-Modell gerechnet,
der Rest mit dem Slope-Modell.

**Spalten.**

- `patno` -- Patienten-ID, frei waehlbar (z.B. P001, P002, ... oder die echte ID
  aus eurem System). Wird nur genutzt, um Visits demselben Patienten zuzuordnen.
- `disease_duration` -- Monate seit Diagnose.
- alle weiteren Spalten sind die Scores (siehe Liste in der CSV-Vorlage).
"""
)

scores = list(SCORE_LABELS.keys())

# CSV-Vorlage zum Download anbieten
def build_template():
    """Vorlage mit Spaltenkopf und drei Beispielzeilen (ein Patient mit 3 Visits)."""
    cols = ["patno", "disease_duration"] + scores
    sample = []
    for v, t in enumerate([0, 12, 24]):
        row = {"patno": "P001", "disease_duration": t}
        for s in scores:
            _, _, default = SCORE_RANGES[s]
            row[s] = default
        sample.append(row)
    return pd.DataFrame(sample, columns=cols).to_csv(index=False)

col_a, col_b = st.columns([1, 3])
with col_a:
    st.download_button(
        "Leere CSV-Vorlage herunterladen",
        data=build_template(),
        file_name="vorlage_subtype_prediction.csv",
        mime="text/csv",
        help="CSV mit allen Spalten und einem Beispielpatient (P001) mit 3 Visits. "
             "Werte einfach ueberschreiben und um eigene Patienten erweitern.",
    )
with col_b:
    st.caption("Die Vorlage enthaelt einen Beispielpatient (P001) mit drei Visits. "
               "Werte sind nur Platzhalter -- bitte mit den realen Daten ueberschreiben.")

uploaded = st.file_uploader("CSV hochladen", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"Eingelesen: {len(df)} Zeilen, {df['patno'].nunique()} Patienten")
    st.dataframe(df.head())

    if st.button("Predictions berechnen", type="primary"):
        # Pro Patient pruefen ob >=2 Visits, dann entsprechend Slope- oder Baseline-Modell
        visits_per_patient = df.groupby("patno").size()
        multi_visit_ids = visits_per_patient[visits_per_patient >= 2].index
        single_visit_ids = visits_per_patient[visits_per_patient == 1].index

        results = []
        if len(multi_visit_ids) > 0:
            multi = df[df["patno"].isin(multi_visit_ids)]
            feats = extract_slope_intercept(multi, scores)
            models = load_models(MODEL_FILES)
            if models:
                preds = predict_all(models, feats)
                preds["model_type"] = "slope"
                results.append(preds)

        if len(single_visit_ids) > 0:
            single = df[df["patno"].isin(single_visit_ids)]
            feats = extract_baseline(single, scores)
            models = load_models(MODEL_FILES_BASELINE)
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
