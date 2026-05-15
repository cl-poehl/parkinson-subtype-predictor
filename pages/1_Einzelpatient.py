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

# Presets: ein Fast- und ein Slow-Beispiel zum schnellen Testen.
# Werte sind plausibel gewaehlt, aber synthetisch.
PRESETS = {
    "fast": {
        "n_visits": 3,
        "visits": [
            {"time": 0,  "UPDRS3_off": 32, "UPDRS3_on": 24, "UPDRS1": 12, "UPDRS2": 14, "UPDRS4": 0,
             "MOCA": 26, "SCOPA": 10, "RBDScr": 5, "VFT_phon_f": 11, "JLO": 24,
             "HY_off": 2, "HY_on": 2, "AXSC_off": 4, "AXSC_on": 3, "PIGD_off": 5, "PIGD_on": 4, "LEDD": 300},
            {"time": 24, "UPDRS3_off": 40, "UPDRS3_on": 30, "UPDRS1": 18, "UPDRS2": 20, "UPDRS4": 1,
             "MOCA": 24, "SCOPA": 14, "RBDScr": 6, "VFT_phon_f": 10, "JLO": 23,
             "HY_off": 2, "HY_on": 2, "AXSC_off": 6, "AXSC_on": 4, "PIGD_off": 7, "PIGD_on": 5, "LEDD": 600},
            {"time": 48, "UPDRS3_off": 48, "UPDRS3_on": 36, "UPDRS1": 24, "UPDRS2": 28, "UPDRS4": 3,
             "MOCA": 21, "SCOPA": 18, "RBDScr": 8, "VFT_phon_f": 9, "JLO": 21,
             "HY_off": 3, "HY_on": 2, "AXSC_off": 9, "AXSC_on": 6, "PIGD_off": 10, "PIGD_on": 7, "LEDD": 950},
        ],
    },
    "slow": {
        "n_visits": 3,
        "visits": [
            {"time": 0,  "UPDRS3_off": 20, "UPDRS3_on": 14, "UPDRS1": 5, "UPDRS2": 7, "UPDRS4": 0,
             "MOCA": 29, "SCOPA": 5, "RBDScr": 2, "VFT_phon_f": 15, "JLO": 29,
             "HY_off": 1, "HY_on": 1, "AXSC_off": 2, "AXSC_on": 1, "PIGD_off": 1, "PIGD_on": 1, "LEDD": 150},
            {"time": 24, "UPDRS3_off": 22, "UPDRS3_on": 15, "UPDRS1": 6, "UPDRS2": 8, "UPDRS4": 0,
             "MOCA": 28, "SCOPA": 6, "RBDScr": 3, "VFT_phon_f": 14, "JLO": 28,
             "HY_off": 2, "HY_on": 1, "AXSC_off": 2, "AXSC_on": 1, "PIGD_off": 2, "PIGD_on": 1, "LEDD": 250},
            {"time": 48, "UPDRS3_off": 24, "UPDRS3_on": 16, "UPDRS1": 7, "UPDRS2": 9, "UPDRS4": 0,
             "MOCA": 28, "SCOPA": 7, "RBDScr": 3, "VFT_phon_f": 14, "JLO": 28,
             "HY_off": 2, "HY_on": 1, "AXSC_off": 2, "AXSC_on": 2, "PIGD_off": 2, "PIGD_on": 2, "LEDD": 350},
        ],
    },
}


def apply_preset(name):
    """Werte aus dem Preset in den Session State schreiben.
    Streamlit-Widgets lesen daraus bei der naechsten Rerun."""
    preset = PRESETS[name]
    st.session_state["n_visits"] = preset["n_visits"]
    for v, vd in enumerate(preset["visits"]):
        st.session_state[f"time_{v}"] = float(vd["time"])
        for score in SCORE_RANGES:
            if score in vd:
                st.session_state[f"{score}_{v}"] = float(vd[score])


st.markdown("### Beispiele")
cb1, cb2, _ = st.columns([1, 1, 3])
with cb1:
    if st.button("Fast-Progressor laden", help="Synthetischer Patient, deutliche Progression"):
        apply_preset("fast")
        st.rerun()
with cb2:
    if st.button("Slow-Progressor laden", help="Synthetischer Patient, stabiler Verlauf"):
        apply_preset("slow")
        st.rerun()

st.divider()

n_visits = st.slider("Anzahl Visits", min_value=1, max_value=6, value=2, key="n_visits",
                     help="Bei nur 1 Visit wird das Single-Visit-Modell genutzt (geringere Genauigkeit).")

st.subheader("Visit-Zeitpunkte und Scores")

# Tabellarische Eingabe ueber Spalten je Visit
visit_data = []
cols = st.columns(n_visits)
for v, col in enumerate(cols):
    with col:
        st.markdown(f"**Visit {v+1}**")
        time_default = v * 12.0
        time_val = st.number_input("Disease Duration (Monate)", min_value=0.0, max_value=300.0,
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
