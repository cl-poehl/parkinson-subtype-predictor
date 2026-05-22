"""Batch tab: CSV upload with template, persistent results in session state."""
import pandas as pd
import streamlit as st

from views._utils import build_template, run_predictions, render_results


def render(score_mode, active_scores, imputer="knn"):
    st.write(
        "Upload a CSV with your own patient data. One row per visit, multiple "
        "visits per patient are supported. Missing score values are allowed."
    )
    with st.expander(":material/menu_book: **CSV format conventions** "
                       "(columns, units, off/on medication)",
                       expanded=False):
        st.markdown(
            """
            **Required columns:**

            - `patno` -- patient identifier (string or integer). Multiple
              rows with the same `patno` are interpreted as repeat
              visits of the same patient.
            - `disease_duration` -- visit time in **months since PD
              diagnosis** (matches PPMI's `Disease_duration`). Same
              patient at months 0, 12, 24 = three visits.

            **Score columns** (one per clinical score). Use the
              exact codes from the empty template -- examples:

            - `UPDRS3_off`, `UPDRS3_on` -- MDS-UPDRS Part III, off
              and on medication respectively. Off = >=12 h since
              last levodopa, on = within 1 h after dose.
            - `MOCA`, `SCOPA`, `RBDScr`, `JLO` -- self-explanatory
              clinical scores.
            - `HY_off`, `HY_on`, `AXSC_off`, `AXSC_on`, `PIGD_off`,
              `PIGD_on` -- Hoehn-Yahr stage and axial/PIGD sub-scores,
              both medication states.
            - `LEDD` -- Levodopa Equivalent Daily Dose in mg/day.

            **Missing values** are allowed and encoded as empty cells.
            The pipeline marks each derived slope/intercept feature
            with a data-quality tag (measured, low-quality from 2 visits,
            or imputed from 0-1 visits) and renders this in the SHAP
            plot.

            **Score set** (selected in the page header):

            - *Standard (17)* -- the LuxPARK-compatible subset; routine
              clinical scores. Use this if your CSV contains the
              standard PPMI minimum.
            - *Extended (25)* -- adds the PPMI research battery (LNS,
              VFT_sem_sum, HVLT_DR, HVLT_IR, SDM, SEADL, ESS, GDS).
              Slightly higher AUC on PPMI but rarely all measured in
              routine clinics.

            Click 'Empty CSV template' to download a skeleton with the
            currently-active columns and one example patient row.
            """
        )

    tcol1, tcol2 = st.columns([2, 3])
    with tcol1:
        st.download_button(
            "Empty CSV template",
            data=build_template(active_scores),
            file_name=f"template_{score_mode}.csv",
            mime="text/csv",
            width="stretch",
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

    state_key = "batch_results"
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"File read: {len(df)} rows, {df['patno'].nunique()} patients")
        with st.expander("Preview data"):
            st.dataframe(df, width="stretch", hide_index=True)

        # Re-Run wenn neuer Upload oder anderer Modus
        cached = st.session_state.get(state_key)
        needs_run = (
            cached is None
            or cached.get("file_id") != uploaded.file_id
            or cached.get("score_mode") != score_mode
        )
        if needs_run:
            with st.spinner("Computing predictions ..."):
                preds, shap_ctx, patient_stats, source_df = run_predictions(
                    df, score_mode, active_scores, imputer=imputer
                )
            if preds is None:
                st.error("No models found.")
                st.session_state[state_key] = None
            else:
                st.session_state[state_key] = {
                    "preds": preds, "shap_ctx": shap_ctx,
                    "patient_stats": patient_stats, "source_df": source_df,
                    "active_scores": active_scores,
                    "score_mode": score_mode, "source": uploaded.name,
                    "file_id": uploaded.file_id,
                }

    cached = st.session_state.get(state_key)
    if cached is not None:
        render_results(
            cached["preds"], cached["source"],
            shap_ctx=cached["shap_ctx"], score_mode=cached["score_mode"],
            patient_stats=cached.get("patient_stats"),
            source_df=cached.get("source_df"),
            active_scores=cached.get("active_scores"),
        )
