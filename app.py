"""Parkinson Subtype Predictor - Haupt-App mit Top-Tabs."""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from src.constants import get_score_set
from views import single_patient, demo, batch, about

st.set_page_config(
    page_title="Parkinson Subtype Predictor",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
    section[data-testid="stMain"] > div.block-container {max-width: 1400px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Kopfzeile
hcol1, hcol2 = st.columns([3, 2], vertical_alignment="bottom")
with hcol1:
    st.title("Parkinson Subtype Predictor")
    st.caption(
        "Vorhersage von Parkinson-Progressionssubtypen (fast vs. slow) auf Basis "
        "klinischer Scores, trainiert auf der PPMI-Kohorte."
    )
with hcol2:
    score_mode = st.segmented_control(
        "Score-Set",
        options=["luxpark", "full"],
        format_func=lambda x: {"luxpark": "Standard (17 Scores)",
                                "full": "Erweitert (25 Scores)"}[x],
        default="luxpark",
        key="score_mode",
        help="Standard: klinische Routine-Scores. Erweitert: zusaetzlich die "
             "PPMI-Forschungsbatterie.",
    )
active_scores = get_score_set(score_mode)

# ---- Haupt-Tabs (Mehr erfahren als eigener Tab, prominenter als ein Popover)
tab_single, tab_batch, tab_demo, tab_about = st.tabs([
    "Einzelpatient",
    "Mehrere Patienten",
    "Demo",
    "Mehr erfahren",
])

with tab_single:
    single_patient.render(score_mode, active_scores)

with tab_batch:
    batch.render(score_mode, active_scores)

with tab_demo:
    demo.render(score_mode, active_scores)

with tab_about:
    about.render(score_mode, active_scores)
