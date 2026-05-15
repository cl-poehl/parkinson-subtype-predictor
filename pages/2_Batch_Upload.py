"""Batch-Modus: CSV hochladen und Predictions zurueckgeben."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pandas as pd
import streamlit as st

from src.constants import SCORE_LABELS, MODEL_FILES, MODEL_FILES_BASELINE
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all

st.set_page_config(page_title="Batch Upload", layout="wide")
st.title("Batch-Vorhersage per CSV")

st.markdown(
    """
Lade eine CSV hoch mit den Spalten `patno`, `disease_duration` (Monate seit Diagnose),
und den Scores in den Spalten unten. Mehrere Visits pro Patient sind moeglich (eine
Zeile pro Visit). Patienten mit nur einer Visit werden mit dem Single-Visit-Modell
gerechnet, der Rest mit dem Slope-Modell.
"""
)

scores = list(SCORE_LABELS.keys())
st.code("patno,disease_duration," + ",".join(scores), language="text")

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
