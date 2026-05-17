"""Demo-Tab: zeigt synthetische Patienten direkt mit Predictions."""
import os
import pandas as pd
import streamlit as st

from views._utils import run_predictions, render_results

DEMO_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "demo_patients.csv")


def render(score_mode, active_scores):
    st.write(
        "Sechs synthetische Patienten zum schnellen Testen der App ohne eigene Daten. "
        "Drei Fast- und drei Slow-Progressors, je mit drei Visits ueber vier Jahre. "
        "Werte sind plausibel gewaehlt, aber komplett erfunden."
    )

    if not os.path.exists(DEMO_CSV_PATH):
        st.error(f"Demo-CSV nicht gefunden unter {DEMO_CSV_PATH}.")
        return

    # Rohdaten optional einklappen
    with st.expander("Demo-Daten ansehen"):
        st.dataframe(pd.read_csv(DEMO_CSV_PATH), use_container_width=True, hide_index=True)

    with st.spinner("Berechne Predictions ..."):
        preds = run_predictions(pd.read_csv(DEMO_CSV_PATH), score_mode, active_scores)
    if preds is None:
        st.error("Keine Modelle gefunden.")
    else:
        render_results(preds, "Demo-Patienten")
