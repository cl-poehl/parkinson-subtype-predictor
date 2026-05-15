"""Einzelpatient-Eingabemaske."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.constants import SCORE_LABELS, SCORE_RANGES, MODEL_FILES, MODEL_FILES_BASELINE
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all

st.set_page_config(page_title="Einzelpatient", layout="wide")
st.title("Einzelpatient-Vorhersage")

n_visits = st.slider("Anzahl Visits", min_value=1, max_value=6, value=2,
                     help="Bei nur 1 Visit wird das Single-Visit-Modell genutzt (geringere Genauigkeit).")

st.subheader("Visit-Zeitpunkte und Scores")

# Tabellarische Eingabe ueber Spalten je Visit
visit_data = []
cols = st.columns(n_visits)
for v, col in enumerate(cols):
    with col:
        st.markdown(f"**Visit {v+1}**")
        time_default = v * 12.0
        time_val = st.number_input(f"Disease Duration (Monate)", min_value=0.0, max_value=300.0,
                                    value=time_default, step=1.0, key=f"time_{v}")
        values = {"patno": "P1", "disease_duration": time_val}
        for score, (lo, hi, default) in SCORE_RANGES.items():
            values[score] = st.number_input(f"{score}", min_value=float(lo), max_value=float(hi),
                                            value=float(default), step=1.0, key=f"{score}_{v}")
        visit_data.append(values)

run = st.button("Vorhersage berechnen", type="primary")

if run:
    df = pd.DataFrame(visit_data)
    scores = list(SCORE_RANGES.keys())

    if n_visits >= 2:
        feats = extract_slope_intercept(df, scores)
        models = load_models(MODEL_FILES)
        used_model = "Slope-Modell (Slopes + Intercepts)"
    else:
        feats = extract_baseline(df, scores)
        models = load_models(MODEL_FILES_BASELINE)
        used_model = "Single-Visit-Modell (Baseline-Scores)"

    if not models:
        st.error("Keine Modelle gefunden. Trainings-Skript muss noch laufen gelassen werden "
                 "(scripts/train_models.py).")
    else:
        preds = predict_all(models, feats)
        st.markdown(f"### Vorhersage ({used_model})")
        for clf_name in preds.columns:
            p_fast = preds[clf_name].iloc[0]
            label = "Fast Progression" if p_fast >= 0.5 else "Slow Progression"
            st.metric(label=clf_name, value=f"{p_fast*100:.1f}% Fast",
                      delta=label, delta_color="off")
            st.progress(float(p_fast))

        st.divider()
        st.markdown("### Konsens")
        mean_fast = preds.mean(axis=1).iloc[0]
        consensus = "Fast Progression" if mean_fast >= 0.5 else "Slow Progression"
        st.markdown(f"**{consensus}** (Mittelwert {mean_fast*100:.1f}% Fast)")

        # TODO SHAP-Plot
        st.info("SHAP-Plot kommt noch.")
