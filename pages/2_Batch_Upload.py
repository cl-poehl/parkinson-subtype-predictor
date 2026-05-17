"""Batch-Modus: CSV hochladen und Predictions zurueckgeben.
Alternativ: synthetische Demo-Daten direkt durchrechnen, ohne Download."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import pandas as pd
import streamlit as st

from src.constants import (
    SCORE_LABELS, SCORE_RANGES, SCORES_LUXPARK,
    get_score_set, get_model_paths,
)
from src.features import extract_slope_intercept, extract_baseline
from src.inference import load_models, predict_all

st.set_page_config(page_title="Batch Upload", layout="wide")
st.title("Batch-Vorhersage per CSV")

# Sidebar Toggle
with st.sidebar:
    st.markdown("##### Konfiguration")
    score_mode = st.radio(
        "Score-Set",
        options=["luxpark", "full"],
        format_func=lambda x: {"luxpark": "17 Scores (LuxPARK-kompatibel)",
                                "full": "25 Scores (voller PPMI-Umfang)"}[x],
        key="score_mode",
    )
    active_scores = get_score_set(score_mode)
    st.caption(f"Aktiv: **{len(active_scores)} Scores**")

st.markdown(
    """
Lade eine CSV hoch mit einer Zeile pro Visit. Patienten mit mehreren Visits werden
mit dem Slope-Modell vorhergesagt, Patienten mit nur einer Visit mit dem
Single-Visit-Modell.

Spalten: `patno`, `disease_duration` (Monate seit Diagnose), und die Scores.
Fehlende Werte sind erlaubt, leere Zellen oder fehlende Spalten werden als
Missing-Values behandelt.
"""
)

DEMO_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "demo_patients.csv")


def build_template():
    cols = ["patno", "disease_duration"] + active_scores
    sample = []
    for v, t in enumerate([0, 12, 24]):
        row = {"patno": "P001", "disease_duration": t}
        for s in active_scores:
            _, _, default = SCORE_RANGES[s]
            row[s] = default
        sample.append(row)
    return pd.DataFrame(sample, columns=cols).to_csv(index=False)


def run_predictions(df_in):
    """Predictions auf einem DataFrame berechnen, ergibt eine einheitliche
    Output-Tabelle. Behandelt Multi-Visit und Single-Visit getrennt."""
    df = df_in.copy()
    # Fehlende Score-Spalten als NaN ergaenzen
    for s in active_scores:
        if s not in df.columns:
            df[s] = pd.NA

    visits_per_patient = df.groupby("patno").size()
    multi_ids = visits_per_patient[visits_per_patient >= 2].index
    single_ids = visits_per_patient[visits_per_patient == 1].index

    out = []
    if len(multi_ids) > 0:
        multi = df[df["patno"].isin(multi_ids)]
        feats = extract_slope_intercept(multi, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=2))
        if models:
            preds = predict_all(models, feats)
            preds["model_type"] = "slope"
            out.append(preds)

    if len(single_ids) > 0:
        single = df[df["patno"].isin(single_ids)]
        feats = extract_baseline(single, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=1))
        if models:
            preds = predict_all(models, feats)
            preds["model_type"] = "baseline"
            out.append(preds)

    if not out:
        return None
    return pd.concat(out).reset_index().rename(columns={"index": "patno"})


def show_results(preds, source_name):
    st.markdown(f"### Ergebnisse fuer {source_name}")
    st.write(f"Predictions fuer {len(preds)} Patienten")

    # Hauptmodelle in Prozent runden fuer Anzeige
    clf_cols = [c for c in preds.columns if c not in ("patno", "model_type")]
    pretty = preds.copy()
    for c in clf_cols:
        pretty[c] = (pretty[c] * 100).round(1)
    pretty["Konsens"] = (preds[clf_cols].mean(axis=1) * 100).round(1)
    pretty["Klasse"] = pretty["Konsens"].apply(lambda x: "Fast" if x >= 50 else "Slow")

    st.dataframe(pretty, use_container_width=True)

    buf = io.StringIO()
    preds.to_csv(buf, index=False)
    st.download_button("Ergebnisse als CSV", buf.getvalue(),
                       file_name="subtype_predictions.csv", mime="text/csv")


# ---- Demo-Daten direkt durchrechnen
with st.container(border=True):
    dcol1, dcol2 = st.columns([1.5, 3])
    with dcol1:
        st.markdown("##### Demo-Daten testen")
        run_demo = st.button("Demo-Daten ausfuehren", use_container_width=True,
                              type="secondary",
                              help="Laedt sechs synthetische Patienten und rechnet "
                                   "die Predictions direkt durch.")
    with dcol2:
        st.caption(
            "Schneller Test mit sechs erfundenen Patienten (3 Fast, 3 Slow, je 3 Visits "
            "ueber 4 Jahre). Komplett synthetische Werte, keine echten Daten. Klick auf "
            "den Button rechnet die Vorhersagen direkt durch -- ohne Download- oder "
            "Upload-Schritt."
        )

# ---- Eigene CSV
st.markdown("##### Eigene CSV hochladen")

tcol1, tcol2 = st.columns([1, 3])
with tcol1:
    st.download_button(
        "Leere Vorlage herunterladen",
        data=build_template(),
        file_name=f"vorlage_{score_mode}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with tcol2:
    st.caption(f"CSV-Vorlage mit den {len(active_scores)} aktiven Score-Spalten und einem "
               f"Beispielpatient (P001) mit drei Visits.")

uploaded = st.file_uploader("CSV-Datei", type=["csv"], label_visibility="collapsed")

# ---- Ergebnisse rendern
if run_demo:
    if os.path.exists(DEMO_CSV_PATH):
        demo_df = pd.read_csv(DEMO_CSV_PATH)
        preds = run_predictions(demo_df)
        if preds is None:
            st.error("Keine Modelle gefunden.")
        else:
            show_results(preds, "Demo-Daten")
    else:
        st.error(f"Demo-CSV nicht gefunden unter {DEMO_CSV_PATH}.")

elif uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"Eingelesen: {len(df)} Zeilen, {df['patno'].nunique()} Patienten")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Predictions berechnen", type="primary"):
        preds = run_predictions(df)
        if preds is None:
            st.error("Keine Modelle gefunden.")
        else:
            show_results(preds, uploaded.name)
