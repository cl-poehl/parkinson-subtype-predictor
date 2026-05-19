"""Demo tab: show synthetic patients, compute predictions on click, persist results."""
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
        "visits over four years. Patient sparseness varies on purpose, so you "
        "can see how the model handles different levels of missingness."
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

    state_key = "demo_results"
    if run:
        with st.spinner("Computing predictions ..."):
            preds, shap_ctx, patient_stats, source_df = run_predictions(
                df, score_mode, active_scores
            )
        if preds is None:
            st.error("No models found.")
            st.session_state[state_key] = None
        else:
            st.session_state[state_key] = {
                "preds": preds, "shap_ctx": shap_ctx,
                "patient_stats": patient_stats, "source_df": source_df,
                "active_scores": active_scores,
                "score_mode": score_mode, "source": "Demo patients",
            }

    cached = st.session_state.get(state_key)
    if cached is not None:
        if cached["score_mode"] != score_mode:
            st.info("Score set changed since last run. Click *Compute prediction* "
                    "to refresh.")
        else:
            render_results(
                cached["preds"], cached["source"],
                shap_ctx=cached["shap_ctx"], score_mode=score_mode,
                patient_stats=cached.get("patient_stats"),
                source_df=cached.get("source_df"),
                active_scores=cached.get("active_scores"),
            )
