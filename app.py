"""Parkinson Subtype Predictor - Hauptseite."""
import streamlit as st

st.set_page_config(
    page_title="Parkinson Subtype Predictor",
    page_icon=":brain:",
    layout="wide",
)

st.title("Parkinson Subtype Predictor")

st.markdown(
    """
Web-App zur Vorhersage des Parkinson-Progressionssubtyps (fast vs. slow) auf Basis
klinischer Scores.

Trainiert auf der PPMI-Kohorte mit drei Klassifikatoren (Random Forest, XGBoost,
Logistic Regression).

**Funktionen** (siehe Sidebar links):

- **Einzelpatient** -- manuelle Eingabe der Scores eines Patienten, mit Wahrscheinlichkeiten
  pro Modell und SHAP-Erklaerung.
- **Batch** -- CSV-Upload mit mehreren Patienten, Predictions als CSV-Download.

**Hinweis.** Die Vorhersage benoetigt fuer das volle Slope-Modell mindestens zwei Visits
pro Patient. Bei nur einer Visite wird automatisch das Single-Visit-Modell verwendet
(geringere Genauigkeit).
"""
)

st.info(
    "Dies ist ein Forschungstool. Die Vorhersagen sind nicht fuer die klinische "
    "Entscheidungsfindung validiert.",
    icon=":material/info:",
)
