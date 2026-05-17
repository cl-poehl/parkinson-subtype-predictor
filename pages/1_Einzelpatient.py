"""Einzelpatient-Eingabemaske, Spreadsheet-Layout, Missingness-Default,
17/25-Score-Toggle."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.constants import (
    SCORE_LABELS, SCORE_RANGES, SCORE_GROUPS, SCORES_LUXPARK,
    get_score_set, get_model_paths,
)
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all
from src.reliability import expected_auc, reliability_label

st.set_page_config(page_title="Einzelpatient", layout="wide")
st.title("Einzelpatient-Vorhersage")

# ---- Sidebar: Score-Modus Toggle (global fuer alle Seiten)
with st.sidebar:
    st.markdown("##### Konfiguration")
    score_mode = st.radio(
        "Score-Set",
        options=["luxpark", "full"],
        format_func=lambda x: {"luxpark": "17 Scores (LuxPARK-kompatibel)",
                                "full": "25 Scores (voller PPMI-Umfang)"}[x],
        key="score_mode",
        help="17 Scores entspricht der LuxPARK-Validierung. 25 Scores nutzt zusaetzlich "
             "die PPMI-spezifische kognitive Batterie sowie SEADL, ESS und GDS, was "
             "die AUC im Schnitt leicht erhoeht, aber die Generalisierbarkeit reduziert.",
    )
    active_scores = get_score_set(score_mode)
    st.caption(f"Aktiv: **{len(active_scores)} Scores**")

st.write(
    "Trage die klinischen Scores eines Patienten ueber eine oder mehrere Visits ein. "
    "Felder, die leer bleiben, werden als fehlend behandelt. Das Modell kann auch "
    "mit unvollstaendigen Daten umgehen, aber die Verlaesslichkeit der Vorhersage "
    "sinkt mit zunehmender Missingness."
)

# ---- Beispiele (Preset-Werte sind nur in den 17 LuxPARK-Scores definiert,
# wuerden im 25-Modus halt nur diese 17 fuellen)
PRESETS = {
    "fast": {
        "n_visits": 3,
        "visits": [
            {"time": 0,  "UPDRS3_off": 32, "UPDRS3_on": 24, "UPDRS1": 12, "UPDRS2": 14, "UPDRS4": 0,
             "MOCA": 26, "SCOPA": 10, "RBDScr": 5, "VFT_phon_f": 11, "JLO": 24,
             "HY_off": 2, "HY_on": 2, "AXSC_off": 4, "AXSC_on": 3, "PIGD_off": 5, "PIGD_on": 4, "LEDD": 300},
            {"time": 24, "UPDRS3_off": 40, "UPDRS3_on": 30, "UPDRS1": 18, "UPDRS2": 20, "UPDRS4": 1,
             "MOCA": None, "SCOPA": 14, "RBDScr": None, "VFT_phon_f": None, "JLO": None,
             "HY_off": 2, "HY_on": 2, "AXSC_off": 6, "AXSC_on": 4, "PIGD_off": 7, "PIGD_on": 5, "LEDD": 600},
            {"time": 48, "UPDRS3_off": 48, "UPDRS3_on": 36, "UPDRS1": 24, "UPDRS2": 28, "UPDRS4": 3,
             "MOCA": 21, "SCOPA": 18, "RBDScr": 8, "VFT_phon_f": 9, "JLO": 21,
             "HY_off": 3, "HY_on": 2, "AXSC_off": 9, "AXSC_on": 6, "PIGD_off": 10, "PIGD_on": 7, "LEDD": 950},
        ],
    },
    "slow": {
        "n_visits": 3,
        "visits": [
            {"time": 0,  "UPDRS3_off": None, "UPDRS3_on": 14, "UPDRS1": 5, "UPDRS2": 7, "UPDRS4": 0,
             "MOCA": 29, "SCOPA": None, "RBDScr": None, "VFT_phon_f": None, "JLO": None,
             "HY_off": None, "HY_on": 1, "AXSC_off": None, "AXSC_on": None,
             "PIGD_off": None, "PIGD_on": None, "LEDD": 150},
            {"time": 24, "UPDRS3_off": None, "UPDRS3_on": 15, "UPDRS1": None, "UPDRS2": 8, "UPDRS4": 0,
             "MOCA": None, "SCOPA": None, "RBDScr": None, "VFT_phon_f": None, "JLO": None,
             "HY_off": None, "HY_on": 1, "AXSC_off": None, "AXSC_on": None,
             "PIGD_off": None, "PIGD_on": None, "LEDD": 250},
            {"time": 48, "UPDRS3_off": None, "UPDRS3_on": 16, "UPDRS1": 7, "UPDRS2": 9, "UPDRS4": 0,
             "MOCA": 28, "SCOPA": None, "RBDScr": None, "VFT_phon_f": None, "JLO": None,
             "HY_off": None, "HY_on": 1, "AXSC_off": None, "AXSC_on": None,
             "PIGD_off": None, "PIGD_on": None, "LEDD": 350},
        ],
    },
}


def empty_visit_data(n):
    return [{s: None for s in SCORE_RANGES} for _ in range(n)]


def default_visit_times(n):
    return [float(v * 12) for v in range(n)]


if "n_visits" not in st.session_state:
    st.session_state.n_visits = 3
    st.session_state.visit_data = empty_visit_data(3)
    st.session_state.visit_times = default_visit_times(3)


def apply_preset(name):
    p = PRESETS[name]
    n = p["n_visits"]
    st.session_state.n_visits = n
    st.session_state.visit_times = [float(v["time"]) for v in p["visits"]]
    st.session_state.visit_data = [
        {s: (None if v.get(s) is None else float(v[s])) for s in SCORE_RANGES}
        for v in p["visits"]
    ]
    for grp in SCORE_GROUPS:
        st.session_state.pop(f"editor_{grp}", None)


# ---- UI: Beispiele
with st.container(border=True):
    bcol1, bcol2 = st.columns([1, 3])
    with bcol1:
        st.markdown("##### Beispielpatienten")
    with bcol2:
        st.caption(
            "Synthetische Beispielpatienten zum Ausprobieren. Werte sind realistisch "
            "gewaehlt, beinhalten typische Luecken aus dem Klinik-Alltag, und koennen "
            "danach beliebig angepasst werden."
        )
    pcol1, pcol2, _ = st.columns([1.3, 1.3, 2])
    with pcol1:
        if st.button("Fast-Progressor", use_container_width=True, type="secondary",
                     help="Synthetischer Patient mit deutlich steigender Symptomatik."):
            apply_preset("fast")
            st.rerun()
    with pcol2:
        if st.button("Slow-Progressor", use_container_width=True, type="secondary",
                     help="Synthetischer Patient mit stabilem Verlauf, sparsam getrackt."):
            apply_preset("slow")
            st.rerun()

st.markdown("")

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

# ---- Scores als Spreadsheet pro Gruppe, gefiltert nach aktivem Score-Set
st.subheader("Klinische Scores")
st.caption(
    "Leere Zellen werden als fehlende Werte behandelt. Score-Bezeichnungen siehe "
    "linke Spalte."
)


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


for group_name, group_scores in SCORE_GROUPS.items():
    visible_scores = [s for s in group_scores if s in active_scores]
    if not visible_scores:
        continue
    with st.expander(group_name, expanded=True):
        data = {
            f"Visit {v+1}": [
                st.session_state.visit_data[v][s] if st.session_state.visit_data[v][s] is not None else np.nan
                for s in visible_scores
            ]
            for v in range(n_visits)
        }
        df = pd.DataFrame(data, index=[SCORE_LABELS[s] for s in visible_scores], dtype="float64")
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

# ---- Vorhersage
st.markdown("")
run = st.button("Vorhersage berechnen", type="primary", use_container_width=True)

if run:
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
        model_type_for_lookup = "slopes+intercepts"
    else:
        feats = extract_baseline(df_pred, active_scores)
        used_model = "Single-Visit-Modell"
        model_type_for_lookup = "slopes+intercepts"

    models = load_models(get_model_paths(score_mode, n_visits))

    if not models:
        st.error("Keine Modelle gefunden. Trainings-Skript noch nicht gelaufen?")
    else:
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
            auc, source = expected_auc(clf_name, model_type_for_lookup, missing_rate, fu,
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

        # Quellen-Hinweis pro Tabellenquelle
        for src in auc_sources:
            if src and src.endswith("_luxpark.csv"):
                st.caption("Erwartete AUC basiert auf einer Simulation auf den 17 LuxPARK-"
                           "kompatiblen Scores.")
            elif src and src.endswith("_full.csv"):
                st.caption("Erwartete AUC basiert auf einer Simulation auf den 25 PPMI-Scores.")
            elif src:
                st.caption(
                    "Erwartete AUC basiert auf einer aelteren Simulation mit 5 Kern-Scores "
                    "(Naeherung). Die webapp-spezifische Simulation laeuft noch, danach werden "
                    "die Zahlen praeziser."
                )

        st.divider()
        mean_fast = float(preds.mean(axis=1).iloc[0])
        consensus = "Fast Progression" if mean_fast >= 0.5 else "Slow Progression"
        st.markdown(f"#### Konsens: **{consensus}** ({mean_fast*100:.1f}% Fast im Mittel)")

        st.info("SHAP-Plot folgt im naechsten Schritt.")
