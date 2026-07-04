"""Parkinson Subtype Predictor - main app with top-tab navigation."""
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

    div[data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1.2rem;
        padding-bottom: 0;
    }
    button[data-baseweb="tab"] {
        height: 56px !important;
        padding: 0 28px !important;
        background: transparent !important;
        border-radius: 10px 10px 0 0 !important;
        transition: background 0.15s, color 0.15s;
    }
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
        font-size: 1.05rem !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"]:hover {
        background: #f3f4f6 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: #eef2ff !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div[data-testid="stMarkdownContainer"] p {
        font-weight: 700 !important;
        color: #4338ca !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #4338ca !important;
        height: 3px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Kopfzeile
hcol1, hcol2 = st.columns([3, 2], vertical_alignment="bottom")
with hcol1:
    st.title("Parkinson Subtype Predictor")
    st.caption(
        "Predicting Parkinson's disease progression subtypes (fast vs. slow) "
        "from clinical score trajectories, trained on the PPMI cohort."
    )
with hcol2:
    score_mode = st.segmented_control(
        "Score set",
        options=["luxpark", "full"],
        format_func=lambda x: {"luxpark": "Standard (17 scores)",
                                "full": "Extended (25 scores)"}[x],
        default="luxpark",
        key="score_mode",
        help="Standard: 17 clinical routine scores. Extended: adds the PPMI "
             "research battery (HVLT, SDM, LNS, VFT semantic, SEADL, ESS, GDS) "
             "for slightly higher accuracy, but rarely available in clinical "
             "routine.",
    )
active_scores = get_score_set(score_mode)

# Imputer hard-coded auf kNN (k=5). Methodisch begruendet: kNN vermeidet den
# Klassen-Bias den Median/Mean durch das 4.5:1 slow:fast PPMI-Verhaeltnis
# einbringen wuerden. Empirische Sensitivitaets-Analyse ueber 8 Imputer
# (median, mean, kNN, MICE, missForest, +indicator-Varianten, native NaN,
# SoftImpute) zeigt AUC-Unterschiede <= 0.013 -- die Wahl ist statistisch
# nicht signifikant. Details in der About-Sektion 'Imputation method
# sensitivity'.

# ---- Haupt-Tabs
tab_single, tab_batch, tab_demo, tab_about = st.tabs([
    "Single patient",
    "Batch",
    "Demo",
    "About",
])

with tab_single:
    single_patient.render(score_mode, active_scores)

with tab_batch:
    batch.render(score_mode, active_scores)

with tab_demo:
    demo.render(score_mode, active_scores)

with tab_about:
    about.render(score_mode, active_scores)
