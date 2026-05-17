"""Einzelpatient-View. Wird aus app.py heraus als Tab gerendert."""
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
        "Trage die klinischen Scores eines Patienten ueber eine oder mehrere Visits ein. "
        "Felder, die leer bleiben, werden als fehlend behandelt. Das Modell kann auch "
        "mit unvollstaendigen Daten umgehen, aber die Verlaesslichkeit der Vorhersage "
        "sinkt mit zunehmender Missingness. Wenn du die App erst einmal ausprobieren "
        "willst, schau in den Demo-Tab."
    )

    # ---- Anzahl Visits
    st.subheader("Patientendaten")
    n_visits = st.slider(
        "Anzahl Visits",
        min_value=1, max_value=6,
        value=st.session_state.n_visits,
        help="Anzahl der dokumentierten Klinikbesuche.",
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

    st.markdown("**Zeitpunkte der Visits** _(Monate seit Diagnose)_")
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
    st.subheader("Klinische Scores")
    st.caption(
        "Leere Zellen werden als fehlende Werte behandelt. Score-Bezeichnungen "
        "siehe linke Spalte."
    )

    for group_name, group_scores in SCORE_GROUPS.items():
        visible_scores = [s for s in group_scores if s in active_scores]
        if not visible_scores:
            continue
        with st.expander(group_name, expanded=True):
            data = {
                f"Visit {v+1}": [
                    st.session_state.visit_data[v][s]
                    if st.session_state.visit_data[v][s] is not None else np.nan
                    for s in visible_scores
                ]
                for v in range(n_visits)
            }
            df = pd.DataFrame(data, index=[SCORE_LABELS[s] for s in visible_scores],
                              dtype="float64")
            df.index.name = "Score"
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
                    st.session_state.visit_data[v][s] = _to_python(val)

    st.markdown("")
    run = st.button("Vorhersage berechnen", type="primary",
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
        used_model = "Slope-Modell (mehrere Visits)"
    else:
        feats = extract_baseline(df_pred, active_scores)
        used_model = "Single-Visit-Modell"

    models = load_models(get_model_paths(score_mode, n_visits))
    if not models:
        st.error("Keine Modelle gefunden. Trainings-Skript noch nicht gelaufen?")
        return

    preds = predict_all(models, feats)
    fu = (max(st.session_state.visit_times) - min(st.session_state.visit_times)
          if n_visits >= 2 else 0)
    st.markdown(f"### Vorhersage  \n*Modell: {used_model}* &nbsp;|&nbsp; "
                f"*Score-Set: {len(active_scores)} Scores* &nbsp;|&nbsp; "
                f"*Anteil fehlend: {missing_rate*100:.0f}%* &nbsp;|&nbsp; "
                f"*Follow-Up: {fu:.0f} Mon.*")

    if missing_rate > 0.5:
        st.warning(
            "Mehr als die Haelfte der Score-Werte fehlen. Vorhersage stuetzt sich "
            "groesstenteils auf imputierte Mittelwerte."
        )

    cols = st.columns(len(preds.columns))
    auc_sources = set()
    for col, clf_name in zip(cols, preds.columns):
        p_fast = float(preds[clf_name].iloc[0])
        label = "Fast Progression" if p_fast >= 0.5 else "Slow Progression"
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
                st.markdown(
                    f"<small>Erwartete AUC: "
                    f"<b style='color:{rel_color}'>{auc:.2f}</b> "
                    f"({rel_text})</small>",
                    unsafe_allow_html=True,
                )

    for src in auc_sources:
        if src and src.endswith("_luxpark.csv"):
            st.caption("Erwartete AUC aus Simulation auf den 17 Standard-Scores.")
        elif src and src.endswith("_full.csv"):
            st.caption("Erwartete AUC aus Simulation auf den 25 erweiterten Scores.")
        elif src:
            st.caption(
                "Erwartete AUC aus aelterer Simulation mit 5 Kern-Scores (Naeherung)."
            )

    st.divider()
    mean_fast = float(preds.mean(axis=1).iloc[0])
    consensus = "Fast Progression" if mean_fast >= 0.5 else "Slow Progression"
    st.markdown(f"#### Konsens: **{consensus}** ({mean_fast*100:.1f}% Fast im Mittel)")

    st.info("SHAP-Plot folgt im naechsten Schritt.")
