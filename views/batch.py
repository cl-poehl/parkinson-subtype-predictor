"""Batch tab: CSV upload with template."""
import pandas as pd
import streamlit as st

from views._utils import build_template, run_predictions, render_results


def render(score_mode, active_scores):
    st.write(
        "Upload a CSV with your own patient data. One row per visit, multiple "
        "visits per patient are supported. Missing score values are allowed."
    )

    tcol1, tcol2 = st.columns([2, 3])
    with tcol1:
        st.download_button(
            "Empty CSV template",
            data=build_template(active_scores),
            file_name=f"template_{score_mode}.csv",
            mime="text/csv",
            use_container_width=True,
            help=f"Template with the {len(active_scores)} active score columns and "
                 f"one example patient (P001) to illustrate the format.",
        )
        st.caption(
            "Columns: `patno`, `disease_duration`, plus the scores. "
            "Download the template, fill it in Excel, upload here."
        )
    with tcol2:
        uploaded = st.file_uploader("CSV file", type=["csv"],
                                     label_visibility="collapsed")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"File read: {len(df)} rows, {df['patno'].nunique()} patients")
        with st.expander("Preview data"):
            st.dataframe(df, use_container_width=True, hide_index=True)

        with st.spinner("Computing predictions ..."):
            preds, shap_ctx = run_predictions(df, score_mode, active_scores)
        if preds is None:
            st.error("No models found.")
        else:
            render_results(preds, uploaded.name, shap_ctx=shap_ctx,
                           score_mode=score_mode)
