"""Einzelpatient-Eingabemaske, Spreadsheet-Layout mit Missing-Values als Default."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.constants import SCORE_LABELS, SCORE_RANGES, MODEL_FILES, MODEL_FILES_BASELINE
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all

st.set_page_config(page_title="Einzelpatient", layout="wide")
st.title("Einzelpatient-Vorhersage")

st.write(
    "Trage die klinischen Scores eines Patienten ueber eine oder mehrere Visits ein. "
    "Felder, die leer bleiben, werden als fehlend behandelt -- das Modell kann auch "
    "mit unvollstaendigen Daten umgehen, allerdings sinkt mit zunehmender Missingness "
    "die Vorhersage-Verlaesslichkeit. Bei mindestens zwei Visits laeuft das Slope-Modell, "
    "bei nur einer Visit das Single-Visit-Modell."
)

# ---- Score-Gruppierung
SCORE_GROUPS = {
    "Motorische Symptome": [
        "UPDRS3_off", "UPDRS3_on", "UPDRS2", "UPDRS4",
        "HY_off", "HY_on", "AXSC_off", "AXSC_on", "PIGD_off", "PIGD_on",
    ],
    "Kognition": ["MOCA", "VFT_phon_f", "JLO"],
    "Nicht-Motorische Symptome": ["UPDRS1", "SCOPA", "RBDScr"],
    "Medikation": ["LEDD"],
}

# ---- Beispiele (mit realistischer Missingness, None = fehlend)
PRESETS = {
    "fast": {
        # gut dokumentierter Patient, ein paar Luecken bei Visit 2 (kognitive Batterie
        # nicht jedes Jahr neu erhoben)
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
        # spaerlich getrackter Patient, nur die ON-State-Messung, kein Off, kein
        # Cognitive Battery, kein RBD/SCOPA -- realistisch fuer eine Routine-Klinik
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
    """Alle Score-Felder sind initial leer (None)."""
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


# ---- UI: Beispiele in eigenem Container
with st.container(border=True):
    bcol1, bcol2 = st.columns([1, 3])
    with bcol1:
        st.markdown("##### Beispielpatienten")
    with bcol2:
        st.caption(
            "Wenn du die App testen moechtest ohne echte Daten, lade einen "
            "synthetischen Beispielpatient. Die Beispiele enthalten bewusst auch "
            "fehlende Werte, wie sie in der Routine-Klinik typisch sind. Werte "
            "lassen sich danach noch beliebig anpassen."
        )
    pcol1, pcol2, _ = st.columns([1.3, 1.3, 2])
    with pcol1:
        if st.button("Fast-Progressor", use_container_width=True, type="secondary",
                     help="Synthetischer Patient mit deutlich steigender Symptomatik. "
                          "Gut dokumentiert, einzelne Visits haben Luecken in der "
                          "kognitiven Batterie."):
            apply_preset("fast")
            st.rerun()
    with pcol2:
        if st.button("Slow-Progressor", use_container_width=True, type="secondary",
                     help="Synthetischer Patient mit stabilem Verlauf. Sparsam "
                          "getrackt, viele Scores fehlen -- typisches Beispiel fuer "
                          "ein Modell-Verhalten bei hoher Missingness."):
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

# ---- Visit-Zeitpunkte
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
    "Leere Zellen werden als fehlende Werte behandelt. Score-Bezeichnungen siehe "
    "linke Spalte. Doppelklick in eine Zelle zum Bearbeiten."
)

def _to_python(val):
    """Konvertiere data_editor-Value (kann NaN sein) in None oder float."""
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
    with st.expander(group_name, expanded=True):
        # DataFrame aufbauen, leere Werte als pd.NA, damit data_editor sie als leere Zelle zeigt
        data = {
            f"Visit {v+1}": [
                st.session_state.visit_data[v][s] if st.session_state.visit_data[v][s] is not None else np.nan
                for s in group_scores
            ]
            for v in range(n_visits)
        }
        df = pd.DataFrame(data, index=[SCORE_LABELS[s] for s in group_scores], dtype="float64")
        df.index.name = "Score"

        col_config = {
            f"Visit {v+1}": st.column_config.NumberColumn(
                f"Visit {v+1}", min_value=0.0, step=1.0, format="%.1f",
            )
            for v in range(n_visits)
        }
        edited = st.data_editor(
            df, key=f"editor_{group_name}", num_rows="fixed",
            column_config=col_config, use_container_width=True,
        )
        for v in range(n_visits):
            col = f"Visit {v+1}"
            for s, val in zip(group_scores, edited[col].values):
                st.session_state.visit_data[v][s] = _to_python(val)

# ---- Vorhersage
st.markdown("")
run = st.button("Vorhersage berechnen", type="primary", use_container_width=True)

if run:
    rows = []
    for v in range(n_visits):
        row = {"patno": "P1", "disease_duration": st.session_state.visit_times[v]}
        for s in SCORE_RANGES:
            row[s] = st.session_state.visit_data[v][s]  # None bleibt None -> wird zu NaN
        rows.append(row)
    df_pred = pd.DataFrame(rows)
    scores_list = list(SCORE_RANGES.keys())

    # Missingness-Anteil als Anhaltspunkt fuer den Nutzer
    score_cells = df_pred[scores_list]
    missing_rate = score_cells.isna().sum().sum() / score_cells.size

    if n_visits >= 2:
        feats = extract_slope_intercept(df_pred, scores_list)
        models = load_models(MODEL_FILES)
        used_model = "Slope-Modell (mehrere Visits)"
    else:
        feats = extract_baseline(df_pred, scores_list)
        models = load_models(MODEL_FILES_BASELINE)
        used_model = "Single-Visit-Modell"

    if not models:
        st.error("Keine Modelle gefunden. Trainings-Skript noch nicht gelaufen?")
    else:
        preds = predict_all(models, feats)
        st.markdown(f"### Vorhersage  \n*Modell: {used_model}* &nbsp;&nbsp;|&nbsp;&nbsp; "
                    f"*Anteil fehlender Werte: {missing_rate*100:.0f}%*")

        if missing_rate > 0.5:
            st.warning(
                "Mehr als die Haelfte der Score-Werte fehlen. Die Vorhersage stuetzt "
                "sich groesstenteils auf imputierte Mittelwerte aus dem Trainingsset "
                "und ist entsprechend unsicher."
            )

        cols = st.columns(len(preds.columns))
        for col, clf_name in zip(cols, preds.columns):
            p_fast = float(preds[clf_name].iloc[0])
            label = "Fast Progression" if p_fast >= 0.5 else "Slow Progression"
            with col:
                st.metric(label=clf_name, value=f"{p_fast*100:.1f}%",
                          delta=label, delta_color="off")
                st.progress(p_fast)

        st.divider()
        mean_fast = float(preds.mean(axis=1).iloc[0])
        consensus = "Fast Progression" if mean_fast >= 0.5 else "Slow Progression"
        st.markdown(f"#### Konsens: **{consensus}** ({mean_fast*100:.1f}% Fast im Mittel)")

        st.info("SHAP-Plot folgt im naechsten Schritt.")
