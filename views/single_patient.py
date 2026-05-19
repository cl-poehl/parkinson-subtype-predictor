"""Single-patient view, rendered as a tab from app.py."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import (
    SCORE_LABELS, SCORE_RANGES, SCORE_GROUPS,
    get_score_set, get_model_paths,
)
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all
from src.reliability import expected_auc, reliability_label
from src.shap_utils import get_shap
from views._utils import patient_shap_bar


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

    # ---- Scores als Spreadsheet pro Gruppe
    st.subheader("Clinical scores")
    st.caption(
        "Empty cells are treated as missing. Double-click a cell to edit."
    )

    # Score-Range-Validierung sammelt Out-of-Range-Werte und clampt sie
    clamp_warnings = []

    for group_name, group_scores in SCORE_GROUPS.items():
        visible_scores = [s for s in group_scores if s in active_scores]
        if not visible_scores:
            continue
        with st.expander(group_name, expanded=True):
            # Range im Label sichtbar
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

    if not run:
        return

    rows = []
    for v in range(n_visits):
        row = {"patno": "P1", "disease_duration": st.session_state.visit_times[v]}
        for s in active_scores:
            row[s] = st.session_state.visit_data[v][s]
        rows.append(row)
    df_pred = pd.DataFrame(rows)
    score_cells = df_pred[active_scores]
    missing_rate = score_cells.isna().sum().sum() / score_cells.size

    if n_visits >= 2:
        feats = extract_slope_intercept(df_pred, active_scores)
        used_model = "Slope model (multiple visits)"
    else:
        feats = extract_baseline(df_pred, active_scores)
        used_model = "Single-visit model"

    models = load_models(get_model_paths(score_mode, n_visits))
    if not models:
        st.error("No models found. Has the training script been run?")
        return

    preds = predict_all(models, feats)
    fu = (max(st.session_state.visit_times) - min(st.session_state.visit_times)
          if n_visits >= 2 else 0)
    st.markdown(f"### Prediction  \n*Model: {used_model}* &nbsp;|&nbsp; "
                f"*Score set: {len(active_scores)} scores* &nbsp;|&nbsp; "
                f"*Missing: {missing_rate*100:.0f}%* &nbsp;|&nbsp; "
                f"*Follow-up: {fu:.0f} mo.*")

    if missing_rate > 0.5:
        st.warning(
            "More than half of the score values are missing. The prediction "
            "relies mostly on median-imputed values from the training cohort."
        )

    cols = st.columns(len(preds.columns))
    auc_sources = set()
    for col, clf_name in zip(cols, preds.columns):
        p_fast = float(preds[clf_name].iloc[0])
        label = "Fast progression" if p_fast >= 0.5 else "Slow progression"
        auc, source = expected_auc(clf_name, "slopes+intercepts", missing_rate, fu,
                                    score_mode=score_mode)
        if source:
            auc_sources.add(source)
        rel_text, rel_color = reliability_label(auc)
        with col:
            st.metric(label=clf_name, value=f"{p_fast*100:.1f}%",
                      delta=label, delta_color="off")
            st.progress(p_fast)
            if auc is not None:
                rel_text_en = {"hoch": "high", "mittel": "medium",
                                "niedrig": "low", "unbekannt": "unknown"}.get(rel_text, rel_text)
                st.markdown(
                    f"<small>Expected AUC: "
                    f"<b style='color:{rel_color}'>{auc:.2f}</b> "
                    f"({rel_text_en})</small>",
                    unsafe_allow_html=True,
                )

    for src in auc_sources:
        if src and src.endswith("_luxpark.csv"):
            st.caption("Expected AUC from a simulation on the 17 standard scores.")
        elif src and src.endswith("_full.csv"):
            st.caption("Expected AUC from a simulation on the 25 extended scores.")
        elif src:
            st.caption(
                "Expected AUC from an older simulation with 5 core scores "
                "(approximation)."
            )

    st.divider()
    mean_fast = float(preds.mean(axis=1).iloc[0])
    consensus = "Fast progression" if mean_fast >= 0.5 else "Slow progression"
    st.markdown(f"#### Consensus: **{consensus}** ({mean_fast*100:.1f}% Fast on average)")

    # ---- SHAP: pro Klassifikator ein einfacher Bar-Chart mit der Richtung
    st.markdown("### Why this prediction?")
    st.caption(
        "For each feature, how much it pushed the model towards **Fast progression** "
        "(red, to the right) or **Slow progression** (blue, to the left) for this "
        "patient. Bars further out from zero had more influence on the prediction."
    )

    clf_tabs = st.tabs(list(preds.columns))
    for tab, clf_name in zip(clf_tabs, preds.columns):
        with tab:
            sv = get_shap(models[clf_name], feats, f"{score_mode}_{clf_name}_{n_visits}")
            if sv is None:
                st.caption("No SHAP plot available for this model.")
                continue
            patient_shap_bar(sv, patient_idx=0, max_display=10)
