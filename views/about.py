"""Mehr-erfahren-Tab: Hintergrund zur App, Methodik, Disclaimer."""
import streamlit as st


def render(*_):
    st.markdown("## Ueber den Parkinson Subtype Predictor")

    st.markdown(
        """
        Eine Web-App zur Vorhersage des Parkinson-Progressionssubtyps -- fast
        progredient oder slow progredient -- aus den Verlaeufen klinischer
        Scores. Sie dient als Forschungs- und Demonstrationstool. Die
        Vorhersagen sind **nicht klinisch validiert** und ersetzen keine
        aerztliche Beurteilung.
        """
    )

    st.markdown("### Wie die Modelle funktionieren")
    st.markdown(
        """
        Drei verschiedene Klassifikatoren wurden auf der PPMI-Kohorte trainiert
        (Parkinson's Progression Markers Initiative, n=409 Patienten).

        - **Random Forest** -- Ensemble aus 500 Entscheidungsbaeumen
        - **XGBoost** -- Gradient-Boosted Trees
        - **Logistische Regression** mit L1-Regularisierung

        Die Eingabe-Features sind pro Score zwei Werte: der **Slope**
        (Steigung der Linearen Regression ueber alle Visits) und der
        **Intercept** (extrapolierter Wert zum Zeitpunkt der Diagnose). Bei
        Patienten mit nur einer Visit kommt stattdessen ein Single-Visit-
        Modell zum Einsatz, das auf absoluten Score-Werten arbeitet --
        weniger genau, aber besser als nichts.

        Alle Modelle sind via isotonischer Cross-Validation-Kalibrierung
        kalibriert, sodass die Wahrscheinlichkeiten valide interpretierbar
        sind: 70 Prozent heisst, in 70 Prozent der vergleichbaren Faelle
        bewahrheitet sich die Klasse.
        """
    )

    st.markdown("### Genauigkeit auf PPMI")
    g1, g2, g3 = st.columns(3)
    g1.metric("Random Forest AUC", "0.95")
    g2.metric("XGBoost AUC", "0.95")
    g3.metric("Logistische Regression AUC", "0.88")
    st.caption("Auf 10-fold Cross-Validation mit GroupKFold nach Patient.")

    st.markdown("### Score-Sets")
    st.markdown(
        """
        - **Standard (17 Scores)** -- klinische Routine-Scores, die in den
          meisten PD-Kliniken erhoben werden. Diese 17 ueberlappen mit der
          LuxPARK-Kohorte (Luxemburg) und sind unsere Basis fuer die externe
          Validierung.
        - **Erweitert (25 Scores)** -- zusaetzlich die PPMI-Forschungs-
          batterie (HVLT, SDM, LNS, VFT semantic, SEADL, ESS, GDS). Etwas
          hoehere Genauigkeit, in der Routine aber selten verfuegbar.
        """
    )

    st.markdown("### Umgang mit fehlenden Werten")
    st.markdown(
        """
        Fehlende Score-Werte werden mit dem Median des Trainingssets imputiert.
        Damit sind Vorhersagen auch bei luekenhaften Daten moeglich. Die
        Verlaesslichkeit sinkt aber mit zunehmender Missingness, was die App
        transparent macht: pro Klassifikator zeigt sie die **erwartete AUC**
        bei der aktuellen Missingness- und Follow-Up-Konstellation, basierend
        auf einer Simulation auf dem PPMI-Datensatz.
        """
    )

    st.markdown("### Disclaimer")
    st.info(
        "Diese App ist ein Forschungs- und Demonstrationstool. Die Vorhersagen "
        "sind nicht klinisch validiert und sollen keine medizinische "
        "Entscheidung ersetzen.",
        icon=":material/info:",
    )

    st.markdown("### Code und Daten")
    st.markdown(
        """
        - Trainingsdaten: PPMI ([ppmi-info.org](https://www.ppmi-info.org))
        - Code: [github.com/cl-poehl/parkinson-subtype-predictor](https://github.com/cl-poehl/parkinson-subtype-predictor)
        - Externe Validierung in Vorbereitung auf der LuxPARK-Kohorte
        """
    )
