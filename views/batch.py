"""Batch-Tab: CSV-Upload mit Vorlage."""
import pandas as pd
import streamlit as st

from views._utils import build_template, run_predictions, render_results


def render(score_mode, active_scores):
    st.write(
        "Lade eine CSV mit deinen eigenen Patientendaten hoch. Eine Zeile pro Visit, "
        "mehrere Visits pro Patient sind moeglich. Fehlende Score-Werte sind erlaubt."
    )

    tcol1, tcol2 = st.columns([2, 3])
    with tcol1:
        st.download_button(
            "Leere CSV-Vorlage",
            data=build_template(active_scores),
            file_name=f"vorlage_{score_mode}.csv",
            mime="text/csv",
            use_container_width=True,
            help=f"Vorlage mit den {len(active_scores)} aktiven Score-Spalten und "
                 f"einem Beispielpatient (P001) zum Orientieren.",
        )
        st.caption(
            "Spalten: `patno`, `disease_duration`, plus die Scores. "
            "Vorlage runterladen, in Excel ausfuellen, hier wieder hochladen."
        )
    with tcol2:
        uploaded = st.file_uploader("CSV hochladen", type=["csv"],
                                     label_visibility="collapsed")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Datei gelesen: {len(df)} Zeilen, {df['patno'].nunique()} Patienten")
        with st.expander("Daten ansehen"):
            st.dataframe(df, use_container_width=True, hide_index=True)

        with st.spinner("Berechne Predictions ..."):
            preds = run_predictions(df, score_mode, active_scores)
        if preds is None:
            st.error("Keine Modelle gefunden.")
        else:
            render_results(preds, uploaded.name)
