"""Parkinson Subtype Predictor - Haupt-App mit Top-Tabs."""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from src.constants import get_score_set
from views import single_patient, demo, batch

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
hcol1, hcol2, hcol3 = st.columns([3, 1, 2], vertical_alignment="bottom")
with hcol1:
    st.title("Parkinson Subtype Predictor")
    st.caption(
        "Vorhersage von Parkinson-Progressionssubtypen (fast vs. slow) auf Basis "
        "klinischer Scores, trainiert auf der PPMI-Kohorte."
    )
with hcol2:
    with st.popover("Mehr erfahren", use_container_width=True):
        st.markdown(
            """
            ### Was diese App macht
            Sie sagt voraus, ob ein Parkinson-Patient eher schnell (fast) oder langsam
            (slow) progredient ist, basierend auf den Verlaeufen klinischer Scores.

            ### Methodik kurz
            Drei Klassifikatoren -- Random Forest, XGBoost und Logistische Regression --
            wurden auf der PPMI-Kohorte (n=409 Patienten) trainiert. Features sind die
            Slopes und Intercepts der Score-Verlaeufe pro Patient. Alle Modelle sind
            via isotonischer Cross-Validation-Kalibrierung kalibriert, sodass die
            ausgegebenen Wahrscheinlichkeiten interpretierbar sind. Auf dem Hold-out-Test
            erreichen XGBoost und Random Forest AUCs von etwa 0.94 bis 0.95, die
            Logistische Regression etwa 0.88.

            ### Score-Sets
            **Standard (17 Scores)**: die klinischen Routine-Scores, die in den meisten
            PD-Kliniken erhoben werden. Diese 17 ueberlappen mit der LuxPARK-Kohorte
            und dienen der externen Validierung.

            **Erweitert (25 Scores)**: zusaetzlich die PPMI-Forschungsbatterie
            (HVLT, SDM, LNS, VFT semantic, SEADL, ESS, GDS). Etwas hoehere Genauigkeit,
            in der klinischen Routine aber selten verfuegbar.

            ### Was die App mit unvollstaendigen Daten macht
            Fehlende Score-Werte werden mit dem Median des Trainingssets imputiert,
            sodass Vorhersagen auch bei luekenhaften Daten moeglich sind. Die
            angezeigte erwartete AUC pro Klassifikator basiert auf einer
            Simulation und zeigt, wie zuverlaessig das Modell bei der aktuellen
            Missingness- und Follow-Up-Konstellation klassifiziert.

            ### Was die App NICHT ist
            Ein Forschungs- und Demonstrationstool. Die Vorhersagen sind **nicht**
            klinisch validiert und sollen keine medizinische Entscheidung ersetzen.

            ### Datenquellen und Code
            Trainingsdaten: PPMI (Parkinson's Progression Markers Initiative).
            Code: github.com/cl-poehl/parkinson-subtype-predictor
            """
        )
with hcol3:
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

# ---- Haupt-Tabs
tab_single, tab_batch, tab_demo = st.tabs([
    "Einzelpatient",
    "Mehrere Patienten",
    "Demo",
])

with tab_single:
    single_patient.render(score_mode, active_scores)

with tab_batch:
    batch.render(score_mode, active_scores)

with tab_demo:
    demo.render(score_mode, active_scores)
