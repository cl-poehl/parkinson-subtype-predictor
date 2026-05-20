"""Single-patient view: form-based input, then full per-patient analytics
(LR-method, bootstrap CIs, score trajectories, percentiles, imputed-flag SHAP)."""
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import SCORE_LABELS, SCORE_RANGES, SCORE_GROUPS
from views._utils import run_predictions, render_results
from src.counterfactuals import single_feature_counterfactuals


def _empty_visit_data(n):
    return [{s: None for s in SCORE_RANGES} for _ in range(n)]


def _default_visit_times(n):
    return [float(v * 12) for v in range(n)]


def _to_python(val):
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def render(score_mode, active_scores):
    if "n_visits" not in st.session_state:
        st.session_state.n_visits = 3
        st.session_state.visit_data = _empty_visit_data(3)
        st.session_state.visit_times = _default_visit_times(3)

    st.write(
        "Enter the clinical scores of a single patient across one or more "
        "visits. Empty fields are treated as missing -- the model can handle "
        "incomplete data, but prediction reliability drops as missingness "
        "increases. If you just want to try the app, check the Demo tab."
    )

    # ---- Number of visits
    st.subheader("Patient data")
    n_visits = st.slider(
        "Number of visits",
        min_value=1, max_value=6,
        value=st.session_state.n_visits,
        help="Number of documented clinic visits.",
    )
    if n_visits != len(st.session_state.visit_data):
        cur, cur_t = st.session_state.visit_data, st.session_state.visit_times
        if n_visits > len(cur):
            for i in range(len(cur), n_visits):
                cur.append({s: None for s in SCORE_RANGES})
                cur_t.append(float(i * 12))
        else:
            cur = cur[:n_visits]
            cur_t = cur_t[:n_visits]
        st.session_state.visit_data = cur
        st.session_state.visit_times = cur_t
        for grp in SCORE_GROUPS:
            st.session_state.pop(f"editor_{grp}", None)
    st.session_state.n_visits = n_visits

    st.markdown("**Visit timepoints** _(months since diagnosis)_")
    time_cols = st.columns(n_visits)
    for v, col in enumerate(time_cols):
        with col:
            new_t = st.number_input(
                f"Visit {v+1}",
                min_value=0.0, max_value=300.0,
                value=float(st.session_state.visit_times[v]),
                step=1.0, key=f"time_input_{v}",
            )
            st.session_state.visit_times[v] = new_t

    st.markdown("")

    # ---- Scores als Spreadsheet pro Gruppe, gefiltert nach aktivem Score-Set
    st.subheader("Clinical scores")
    st.caption(
        "Empty cells are treated as missing. Double-click a cell to edit. "
        "Values outside the typical range are automatically clamped."
    )

    clamp_warnings = []

    for group_name, group_scores in SCORE_GROUPS.items():
        visible_scores = [s for s in group_scores if s in active_scores]
        if not visible_scores:
            continue
        with st.expander(group_name, expanded=True):
            row_labels = [
                f"{SCORE_LABELS[s]} [{int(SCORE_RANGES[s][0])}-{int(SCORE_RANGES[s][1])}]"
                for s in visible_scores
            ]
            data = {
                f"Visit {v+1}": [
                    st.session_state.visit_data[v][s]
                    if st.session_state.visit_data[v][s] is not None else np.nan
                    for s in visible_scores
                ]
                for v in range(n_visits)
            }
            df = pd.DataFrame(data, index=row_labels, dtype="float64")
            df.index.name = "Score [range]"
            col_config = {
                f"Visit {v+1}": st.column_config.NumberColumn(
                    f"Visit {v+1}", min_value=0.0, step=1.0, format="%.1f",
                )
                for v in range(n_visits)
            }
            edited = st.data_editor(
                df, key=f"editor_{group_name}_{score_mode}", num_rows="fixed",
                column_config=col_config, use_container_width=True,
            )
            for v in range(n_visits):
                col = f"Visit {v+1}"
                for s, val in zip(visible_scores, edited[col].values):
                    py = _to_python(val)
                    if py is not None:
                        lo, hi, _ = SCORE_RANGES[s]
                        if py < lo or py > hi:
                            clamped = max(float(lo), min(float(hi), py))
                            clamp_warnings.append(
                                f"{SCORE_LABELS[s]} (Visit {v+1}): {py:g} -> {clamped:g}"
                            )
                            py = clamped
                    st.session_state.visit_data[v][s] = py

    if clamp_warnings:
        st.warning(
            "Some values were outside the valid range and were clamped:\n"
            + "\n".join(f"- {w}" for w in clamp_warnings)
        )

    st.markdown("")
    run = st.button("Compute prediction", type="primary",
                    use_container_width=True, key="single_run")

    state_key = "single_results"
    if run:
        # Build df aus den Visit-Daten
        rows = []
        for v in range(n_visits):
            row = {"patno": "P1", "disease_duration": st.session_state.visit_times[v]}
            for s in active_scores:
                row[s] = st.session_state.visit_data[v][s]
            rows.append(row)
        df_pred = pd.DataFrame(rows)
        with st.spinner("Computing prediction ..."):
            preds, shap_ctx, patient_stats, source_df = run_predictions(
                df_pred, score_mode, active_scores
            )
        if preds is None:
            st.error("No models found. Has the training script been run?")
            st.session_state[state_key] = None
        else:
            st.session_state[state_key] = {
                "preds": preds, "shap_ctx": shap_ctx,
                "patient_stats": patient_stats, "source_df": source_df,
                "active_scores": active_scores, "score_mode": score_mode,
            }

    cached = st.session_state.get(state_key)
    if cached is not None:
        if cached["score_mode"] != score_mode:
            st.info("Score set changed since last run. Click *Compute prediction* "
                    "to refresh.")
        else:
            render_results(
                cached["preds"], "Single patient",
                shap_ctx=cached["shap_ctx"], score_mode=score_mode,
                patient_stats=cached.get("patient_stats"),
                source_df=cached.get("source_df"),
                active_scores=cached.get("active_scores"),
            )
