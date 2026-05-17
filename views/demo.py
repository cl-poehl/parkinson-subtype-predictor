"""Demo tab: show synthetic patients, then compute predictions on click."""
import os
import pandas as pd
import streamlit as st

from views._utils import run_predictions, render_results

DEMO_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "demo_patients.csv")


def render(score_mode, active_scores):
    st.write(
        "Six synthetic patients to try out the app without your own data. "
        "Three fast progressors and three slow progressors, each with three "
        "visits over four years. The values are plausible but completely "
        "made up."
    )

    if not os.path.exists(DEMO_CSV_PATH):
        st.error(f"Demo CSV not found at {DEMO_CSV_PATH}.")
        return

    df = pd.read_csv(DEMO_CSV_PATH)

    st.markdown("##### Demo data")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("")
    run = st.button("Compute prediction", type="primary",
                    use_container_width=True, key="demo_run")

    if run:
        with st.spinner("Computing predictions ..."):
            preds = run_predictions(df, score_mode, active_scores)
        if preds is None:
            st.error("No models found.")
        else:
            render_results(preds, "Demo patients")
