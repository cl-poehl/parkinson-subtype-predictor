"""Shared helpers fuer demo.py und batch.py.
Sammelt pro Patient: Predictions (alle Klassifikatoren + LR-Methode), Per-Fold-
Vorhersagen fuer CI, Missingness/Follow-Up, Visit-Liste, Imputations-Flags,
Perzentile gegenueber PPMI-Subtyp-Verteilungen.
Rendert Uebersicht plus Detail-Drilldown pro Patient."""
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import SCORE_LABELS, SCORE_RANGES, get_model_paths
from src.features import extract_slope_intercept, extract_baseline, imputation_flags
from src.inference import load_models, predict_all, predict_all_with_folds
from src.lr_method import (
    lr_predict_from_slopes, percentile_in_subtype, get_reference,
)
from src.reliability import expected_auc, expected_auc_ci, reliability_label
from src.shap_utils import get_shap


# ----------------------- Templates --------------------------
def build_template(active_scores):
    cols = ["patno", "disease_duration"] + active_scores
    sample = []
    for v, t in enumerate([0, 12, 24]):
        row = {"patno": "P001", "disease_duration": t}
        for s in active_scores:
            _, _, default = SCORE_RANGES[s]
            row[s] = default
        sample.append(row)
    return pd.DataFrame(sample, columns=cols).to_csv(index=False)


# ----------------------- Prediction pipeline ----------------
def _per_patient_meta(df, active_scores):
    """Pro Patient Missingness, Follow-Up, Visit-Liste, Anzahl Visits."""
    meta = {}
    for patno, group in df.groupby("patno"):
        score_cells = group[active_scores]
        miss = float(score_cells.isna().sum().sum() / max(score_cells.size, 1))
        times = group["disease_duration"].sort_values().tolist()
        meta[str(patno)] = {
            "missing": miss,
            "follow_up": float(max(times) - min(times)) if len(times) >= 2 else 0.0,
            "visit_times": [float(t) for t in times],
            "n_visits": len(times),
        }
    return meta


def _compute_lr_predictions(df_slope, score_mode):
    """LR-Methode pro Patient. df_slope ist der OLS-Slope-Feature-DataFrame
    (index = patno, columns = '<score>_slope', '<score>_intercept').
    Returns dict {patno: lr_result_dict}."""
    out = {}
    for patno, row in df_slope.iterrows():
        slopes = {}
        for col in df_slope.columns:
            if col.endswith("_slope"):
                score = col[:-6]
                v = row[col]
                if not pd.isna(v):
                    slopes[score] = float(v)
        result = lr_predict_from_slopes(slopes, score_mode)
        out[str(patno)] = result
    return out


def run_predictions(df_in, score_mode, active_scores):
    """Vollstaendige Vorhersage-Pipeline pro Patient.

    Returns (preds, shap_ctx, patient_stats, source_df).
    - preds: pd.DataFrame mit patno, model_type, P(Fast) je Klassifikator,
             LR-Methode als zusaetzliche Spalte 'Likelihood Ratio'.
    - shap_ctx: {mtype: (feats, models)} fuer SHAP-Berechnung.
    - patient_stats: {patno: {missing, follow_up, visit_times, n_visits,
                              imputed: {feat: bool}, folds: {clf: array},
                              lr_method: dict}}.
    - source_df: das Originaldatum (Visit-Zeilen) fuer Trajektorien-Plot.
    """
    df = df_in.copy()
    for s in active_scores:
        if s not in df.columns:
            df[s] = pd.NA

    patient_stats = _per_patient_meta(df, active_scores)

    visits_per_patient = df.groupby("patno").size()
    multi_ids = visits_per_patient[visits_per_patient >= 2].index
    single_ids = visits_per_patient[visits_per_patient == 1].index

    out = []
    shap_ctx = {}

    if len(multi_ids) > 0:
        multi = df[df["patno"].isin(multi_ids)]
        feats = extract_slope_intercept(multi, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=2))
        if models:
            mean_df, folds = predict_all_with_folds(models, feats)
            mean_df["model_type"] = "slope"
            mean_df["patno"] = mean_df.index.astype(str)
            out.append(mean_df.reset_index(drop=True))
            shap_ctx["slope"] = (feats, models)

            # Folds pro Patient zuordnen
            for pos, patno in enumerate(feats.index):
                patient_stats[str(patno)]["folds"] = {
                    clf: folds[clf][pos] for clf in folds
                }

            # Imputations-Flags fuer Slope-Modell
            imp = imputation_flags(multi, active_scores, mode="slope")
            for patno, ff in imp.items():
                patient_stats[str(patno)]["imputed"] = ff

            # LR-Methode
            lr_results = _compute_lr_predictions(feats, score_mode)
            # in das preds-DataFrame integrieren als zusaetzliche Spalte
            last = out[-1]
            last["Likelihood Ratio"] = [
                (lr_results.get(str(p)) or {}).get("p_fast", np.nan)
                for p in last["patno"]
            ]
            for patno, res in lr_results.items():
                if patno in patient_stats:
                    patient_stats[patno]["lr_method"] = res

    if len(single_ids) > 0:
        single = df[df["patno"].isin(single_ids)]
        feats = extract_baseline(single, active_scores)
        models = load_models(get_model_paths(score_mode, n_visits=1))
        if models:
            mean_df, folds = predict_all_with_folds(models, feats)
            mean_df["model_type"] = "baseline"
            mean_df["patno"] = mean_df.index.astype(str)
            # Baseline-Modell hat keine LR-Schaetzung (LR braucht Slope -> >=2 Visits)
            mean_df["Likelihood Ratio"] = np.nan
            out.append(mean_df.reset_index(drop=True))
            shap_ctx["baseline"] = (feats, models)

            for pos, patno in enumerate(feats.index):
                patient_stats[str(patno)]["folds"] = {
                    clf: folds[clf][pos] for clf in folds
                }

            imp = imputation_flags(single, active_scores, mode="baseline")
            for patno, ff in imp.items():
                patient_stats[str(patno)]["imputed"] = ff
            # Keine LR-Methode fuer Single-Visit
            for patno in single["patno"].astype(str).unique():
                if patno in patient_stats:
                    patient_stats[patno]["lr_method"] = None

    if not out:
        return None, {}, {}, df_in

    full = pd.concat(out, ignore_index=True)
    return full, shap_ctx, patient_stats, df_in


# ----------------------- SHAP-Bar ---------------------------
def patient_shap_bar(sv, patient_idx=0, imputed_lookup=None, max_display=None):
    """SHAP-Beitraege als horizontale Bars. imputed_lookup: dict feature_name -> bool,
    Features die imputiert wurden bekommen ein '(imputed)' im Label."""
    values = sv.values[patient_idx]
    abs_v = np.abs(values)
    order = np.argsort(abs_v)[::-1]
    if max_display is not None:
        order = order[:max_display]
    feat_names = [sv.feature_names[i] for i in order]

    # Original-Feature-Codes brauchen wir, um imputed-Lookup zu machen.
    # sv.feature_names enthaelt bereits den huebsche Form, aber die mappen
    # nicht 1:1 zurueck. Wir holen die Codes aus den Daten.
    # Heuristik: Pretty-Name muss zurueckgemappt werden. Wir nehmen direkt
    # aus den ORIGINAL feature-Cols der zugehoerigen X-Matrix (gespeichert
    # in sv.data shape n_samples x n_features). Aber wir haben die Codes
    # nicht im Explainer gespeichert. Daher: imputed_lookup als parallele
    # Liste in derselben Reihenfolge wie sv.feature_names erwarten.

    vals = values[order]
    df = pd.DataFrame({"feature": feat_names, "shap": vals})

    # Marker fuer imputierte Features
    if imputed_lookup is not None:
        marks = []
        for i in order:
            name = sv.feature_names[i]
            marks.append(" (imputed)" if imputed_lookup.get(name, False) else "")
        df["feature"] = [f + m for f, m in zip(df["feature"], marks)]
        df["imputed"] = [imputed_lookup.get(sv.feature_names[i], False)
                          for i in order]
    else:
        df["imputed"] = False

    df["direction"] = df["shap"].apply(lambda x: "Fast" if x >= 0 else "Slow")
    bound = max(abs_v.max() * 1.15, 0.01) if len(abs_v) else 0.01

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y("feature:N", sort=df["feature"].tolist(),
                    axis=alt.Axis(title=None, labelLimit=400)),
            x=alt.X("shap:Q",
                    scale=alt.Scale(domain=[-bound, bound]),
                    axis=alt.Axis(title="SHAP value   (← Slow      Fast →)")),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Slow", "Fast"], range=["#3b82f6", "#ef4444"]),
                legend=None,
            ),
            opacity=alt.condition("datum.imputed", alt.value(0.45), alt.value(1.0)),
            tooltip=["feature", alt.Tooltip("shap:Q", format=".3f"),
                     "direction", "imputed"],
        )
        .properties(height=max(26 * len(df), 200))
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="black").encode(x="x:Q")
    st.altair_chart(chart + rule, use_container_width=True)


# ----------------------- Score-Trajektorien -----------------
def score_trajectory_plot(source_df, patno, active_scores):
    """Line-Chart Grid: ein kleines Diagramm pro Score, x = Disease Duration,
    y = Score-Wert, Punkte fuer einzelne Visits."""
    patient_rows = source_df[source_df["patno"].astype(str) == str(patno)].copy()
    if patient_rows.empty:
        st.caption("No visit data available for this patient.")
        return

    long_rows = []
    for _, row in patient_rows.iterrows():
        for s in active_scores:
            val = row.get(s)
            if pd.isna(val):
                continue
            long_rows.append({
                "Score": SCORE_LABELS.get(s, s),
                "code": s,
                "Disease duration (months)": float(row["disease_duration"]),
                "Value": float(val),
            })
    if not long_rows:
        st.caption("No measured scores for this patient.")
        return
    long_df = pd.DataFrame(long_rows)

    # Reihenfolge nach Score-Liste
    score_order = [SCORE_LABELS.get(s, s) for s in active_scores
                   if SCORE_LABELS.get(s, s) in long_df["Score"].unique()]

    chart = (
        alt.Chart(long_df)
        .mark_line(point=alt.OverlayMarkDef(size=60))
        .encode(
            x=alt.X("Disease duration (months):Q",
                    axis=alt.Axis(format="d")),
            y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
            color=alt.value("#4338ca"),
            tooltip=["Score", "Disease duration (months)", "Value"],
        )
        .properties(width=240, height=130)
        .facet(
            facet=alt.Facet("Score:N", sort=score_order,
                             header=alt.Header(labelFontSize=11)),
            columns=4,
        )
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart, use_container_width=False)


# ----------------------- Visit-Liste, CI, Perzentile ----------
def _ci_from_folds(folds_array, ci=0.95):
    """95%-Konfidenzintervall des Mittelwerts ueber die K=5 CV-Folds:
    mean ± z * std/sqrt(K) (z=1.96 fuer 95%). Liefert (lo, hi) in P(Fast)-Einheiten,
    geclippt auf [0, 1]. Bei einer Punktwolke ohne Streuung (alle Folds gleich)
    ist die Range null."""
    if folds_array is None or len(folds_array) == 0:
        return (np.nan, np.nan)
    arr = np.asarray(folds_array, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    mean = float(arr.mean())
    if len(arr) < 2:
        return (mean, mean)
    z = 1.96 if abs(ci - 0.95) < 1e-6 else float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + ci / 2))
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return (max(0.0, mean - z * se), min(1.0, mean + z * se))


def _fold_range(folds_array):
    """Volles min-max als zusaetzliche Info im Detail-Panel."""
    if folds_array is None or len(folds_array) == 0:
        return (np.nan, np.nan)
    return float(np.nanmin(folds_array)), float(np.nanmax(folds_array))


def _confidence_range(folds_array):
    """95% CI der Confidence max(p,1-p) ueber die Folds, basierend auf dem
    95%-CI der P(Fast). Wenn die CI 0.5 straddlet, liegt die Untergrenze
    bei 0.5 (Confidence kann nicht kleiner als ein Muenzwurf sein)."""
    if folds_array is None or len(folds_array) == 0:
        return (np.nan, np.nan)
    p_lo, p_hi = _ci_from_folds(folds_array)
    if p_lo >= 0.5:
        return (p_lo, p_hi)
    if p_hi <= 0.5:
        return (1 - p_hi, 1 - p_lo)
    return (0.5, max(p_hi, 1 - p_lo))


def _percentile_panel(reference, slopes_dict, score_mode):
    """Zeigt fuer jeden Score mit Slope: Perzentil im fast vs slow Subtyp.
    Tabelle mit Score, Slope, Perzentil-fast, Perzentil-slow."""
    rows = []
    for score, slope in slopes_dict.items():
        if slope is None or np.isnan(slope):
            continue
        pf = percentile_in_subtype(reference, score, slope, subtype=1, dist_kind="slope")
        ps = percentile_in_subtype(reference, score, slope, subtype=2, dist_kind="slope")
        rows.append({
            "Score": SCORE_LABELS.get(score, score),
            "Patient slope": slope,
            "Percentile (Fast cohort)": pf,
            "Percentile (Slow cohort)": ps,
        })
    if not rows:
        st.caption("No slopes available for percentile lookup.")
        return
    df = pd.DataFrame(rows)
    df_show = df.copy()
    df_show["Patient slope"] = df_show["Patient slope"].apply(
        lambda x: f"{x:+.3f}" if pd.notna(x) else "—"
    )
    for c in ["Percentile (Fast cohort)", "Percentile (Slow cohort)"]:
        df_show[c] = df_show[c].apply(
            lambda x: f"{x:.0f}th" if pd.notna(x) else "—"
        )
    st.dataframe(df_show, use_container_width=True, hide_index=True)


# ----------------------- Hauptansicht -----------------------
def render_results(preds, source_name, shap_ctx=None, score_mode="luxpark",
                    patient_stats=None, source_df=None, active_scores=None):
    """Komplette Ergebnis-Sektion."""
    clf_cols = [c for c in preds.columns if c not in (
        "patno", "model_type", "Likelihood Ratio"
    )]
    has_lr = "Likelihood Ratio" in preds.columns
    all_method_cols = clf_cols + (["Likelihood Ratio"] if has_lr else [])

    # Konsens: Mittelwert aller verfuegbaren Methoden (NaN-safe)
    consensus = preds[all_method_cols].mean(axis=1, skipna=True)
    preds = preds.assign(consensus=consensus,
                          klasse=consensus.apply(
                              lambda x: "Fast" if x >= 0.5 else "Slow"))

    n = len(preds)
    n_fast = int((preds["consensus"] >= 0.5).sum())
    n_slow = n - n_fast
    mean_conf_score = preds["consensus"].apply(lambda x: max(x, 1 - x)).mean()

    st.markdown(f"### Results  \n*Source: {source_name}*")

    # Cohort-Header und Overview-Chart machen nur bei mehreren Patienten Sinn.
    if n > 1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patients", n)
        m2.metric("Fast progression", n_fast)
        m3.metric("Slow progression", n_slow)
        m4.metric("Mean confidence", f"{mean_conf_score*100:.0f}%",
                   help="Average certainty in the predicted class across all "
                        "patients. High = the cohort is classified decisively "
                        "(not 50/50 'don't know').")
        st.markdown("")

    # ---- Uebersichts-Chart: Confidence pro Patient pro Modell
    if n > 1 and n <= 200:
        method_palette = {
            "Random Forest": "#10b981",
            "XGBoost": "#f97316",
            "Logistic Regression": "#6366f1",
            "Likelihood Ratio": "#a855f7",
        }
        long_rows = []
        for _, row in preds.iterrows():
            patno = str(row["patno"])
            for c in all_method_cols:
                p = row[c]
                if pd.isna(p):
                    continue
                p = float(p)
                # CI nur fuer ML-Modelle (LR hat keine Folds)
                folds = (patient_stats or {}).get(patno, {}).get("folds", {})
                if c in folds and len(folds[c]) > 0:
                    conf_lo, conf_hi = _confidence_range(folds[c])
                    # Min/Max ueber die Folds in Confidence-Space als Whisker-Punkte
                    fold_confs = [max(f, 1 - f) for f in folds[c]]
                    conf_min = float(min(fold_confs))
                    conf_max = float(max(fold_confs))
                else:
                    conf_lo, conf_hi = max(p, 1 - p), max(p, 1 - p)
                    conf_min = conf_max = max(p, 1 - p)
                long_rows.append({
                    "patno": patno, "Method": c,
                    "prob": p, "confidence": max(p, 1 - p),
                    "conf_lo": conf_lo, "conf_hi": conf_hi,
                    "conf_min": conf_min, "conf_max": conf_max,
                    "predicted_class": "Fast" if p >= 0.5 else "Slow",
                })
        long_df = pd.DataFrame(long_rows)
        # Reihenfolge der Patienten wie im Input (preds), nicht nach Konsens sortiert
        patno_order = preds["patno"].astype(str).drop_duplicates().tolist()

        st.caption(
            "Per patient, how certain each model is about its prediction "
            "(50% = coin flip, 100% = absolutely sure). Box-plot-style "
            "display: the **filled symbol** is the mean across the K=5 "
            "CalibratedClassifierCV folds, the **thick bar** is the 95% "
            "confidence interval of that mean (mean ± 1.96·std/√K), the **two "
            "open circles** above and below mark the min and max across the "
            "folds (\"whiskers\"). Likelihood Ratio has no fold-based spread "
            "(single fit on the full PPMI cohort). Symbol shape of the filled "
            "mean = predicted class, color = method. Patients in input order."
        )

        method_order = [m for m in
                         ["Random Forest", "XGBoost", "Logistic Regression",
                          "Likelihood Ratio"]
                         if m in long_df["Method"].unique()]

        errorbars = (
            alt.Chart(long_df)
            .mark_errorbar(thickness=1.5)
            .encode(
                x=alt.X("patno:N", sort=patno_order),
                y=alt.Y("conf_lo:Q",
                        scale=alt.Scale(domain=[0.5, 1.0]),
                        title=None, axis=None),
                y2=alt.Y2("conf_hi:Q", title=None),
                color=alt.Color("Method:N",
                                scale=alt.Scale(domain=method_order,
                                                 range=[method_palette[m]
                                                         for m in method_order]),
                                legend=None),
                xOffset=alt.XOffset("Method:N"),
            )
        )
        # Whisker-Punkte fuer min und max ueber die Folds (Box-Plot-aehnliche
        # Darstellung)
        whisker_min = (
            alt.Chart(long_df)
            .mark_point(filled=False, size=40, strokeWidth=1.5, opacity=0.7)
            .encode(
                x=alt.X("patno:N", sort=patno_order),
                y=alt.Y("conf_min:Q", title=None, axis=None),
                color=alt.Color("Method:N",
                                scale=alt.Scale(domain=method_order,
                                                 range=[method_palette[m]
                                                         for m in method_order]),
                                legend=None),
                xOffset=alt.XOffset("Method:N"),
                tooltip=["patno", "Method",
                         alt.Tooltip("conf_min:Q", format=".1%", title="Min")],
            )
        )
        whisker_max = whisker_min.encode(
            y=alt.Y("conf_max:Q", title=None, axis=None),
            tooltip=["patno", "Method",
                     alt.Tooltip("conf_max:Q", format=".1%", title="Max")],
        )
        points = (
            alt.Chart(long_df)
            .mark_point(filled=True, size=110, opacity=0.9)
            .encode(
                x=alt.X("patno:N", sort=patno_order,
                        axis=alt.Axis(labelAngle=-40, title="Patient")),
                y=alt.Y("confidence:Q",
                        scale=alt.Scale(domain=[0.5, 1.0]),
                        axis=alt.Axis(format="%",
                                       title="Certainty in predicted class")),
                color=alt.Color(
                    "Method:N",
                    scale=alt.Scale(domain=method_order,
                                     range=[method_palette[m] for m in method_order]),
                    legend=alt.Legend(title="Method", orient="top"),
                ),
                shape=alt.Shape(
                    "predicted_class:N",
                    scale=alt.Scale(domain=["Fast", "Slow"],
                                     range=["circle", "square"]),
                    legend=alt.Legend(title="Predicted class", orient="top",
                                       symbolFillColor="#374151",
                                       symbolStrokeColor="#374151"),
                ),
                xOffset=alt.XOffset("Method:N"),
                tooltip=["patno", "Method",
                         alt.Tooltip("predicted_class:N", title="Class"),
                         alt.Tooltip("prob:Q", format=".1%", title="P(Fast)"),
                         alt.Tooltip("confidence:Q", format=".1%"),
                         alt.Tooltip("conf_lo:Q", format=".1%", title="CI low"),
                         alt.Tooltip("conf_hi:Q", format=".1%", title="CI high")],
            )
        )
        st.altair_chart(
            (errorbars + whisker_min + whisker_max + points)
            .properties(height=320),
            use_container_width=True,
        )
        st.markdown("")

    # ---- Tabelle: nur bei >1 Patient sinnvoll, sonst redundant zum Detail-Panel
    if n > 1:
        pretty_cols = ["patno", "klasse", "consensus"] + all_method_cols
        if "model_type" in preds.columns:
            pretty_cols.append("model_type")
        pretty = preds[pretty_cols].copy()
        for c in all_method_cols:
            pretty[c] = pretty[c].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
            )
        pretty["consensus"] = pretty["consensus"].apply(lambda x: f"{x*100:.1f}%")
        pretty = pretty.rename(columns={
            "patno": "Patient", "consensus": "Consensus",
            "klasse": "Class", "model_type": "Model type"
        })
        st.dataframe(pretty, use_container_width=True, hide_index=True)

        buf = io.StringIO()
        preds.drop(columns=["klasse"]).to_csv(buf, index=False)
        st.download_button(
            "Download results as CSV", buf.getvalue(),
            file_name="subtype_predictions.csv", mime="text/csv",
        )

    # ---- Per-Patient Detail
    if not shap_ctx:
        return
    ordered_ids = list(preds["patno"].astype(str).unique())
    if n > 1:
        st.markdown("### Per-patient detail")
        st.caption(
            "Choose a patient to inspect their visit trajectories, model "
            "confidence and reliability, percentile position in the PPMI "
            "cohort, and SHAP-based explanation of the prediction."
        )
        selected = st.selectbox("Patient", options=ordered_ids,
                                 key=f"detail_patient_{source_name}")
    else:
        # Bei nur einem Patient kein Dropdown, direkt rendern
        selected = ordered_ids[0]

    sel_row = preds[preds["patno"].astype(str) == selected].iloc[0]
    sel_consensus = float(sel_row["consensus"])
    sel_class = "Fast" if sel_consensus >= 0.5 else "Slow"
    sel_color = "#ef4444" if sel_class == "Fast" else "#3b82f6"

    stats = (patient_stats or {}).get(selected, {})
    miss = stats.get("missing", 0)
    fu = stats.get("follow_up", 0)
    visit_times = stats.get("visit_times", [])
    n_visits = stats.get("n_visits", 0)

    # Patient-Summary-Header
    st.markdown(
        f"**{selected}** — Consensus: "
        f"<b style='color:{sel_color}'>{sel_consensus*100:.1f}% Fast</b> "
        f"({sel_class} progression)",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<small>Visits: **{n_visits}** at "
        f"{', '.join(f'{int(t)} mo' for t in visit_times)} &nbsp;|&nbsp; "
        f"Follow-up: **{fu:.0f} months** &nbsp;|&nbsp; "
        f"Missing scores: **{miss*100:.0f}%**</small>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ---- Score-Trajektorien
    if source_df is not None and active_scores is not None:
        st.markdown("##### Score trajectories")
        st.caption("One small chart per measured score. Filled scores only -- "
                    "unmeasured scores are not shown.")
        score_trajectory_plot(source_df, selected, active_scores)
        st.markdown("")

    # ---- Methoden-Detail mit Confidence + Bootstrap-CI + Expected AUC
    st.markdown("##### Predictions per method")
    methods_to_show = [m for m in all_method_cols if pd.notna(sel_row[m])]
    metric_cols = st.columns(len(methods_to_show))
    folds = stats.get("folds", {})
    for mcol, name in zip(metric_cols, methods_to_show):
        p = float(sel_row[name])
        conf = max(p, 1 - p)
        cls = "Fast" if p >= 0.5 else "Slow"
        cls_color = "#ef4444" if cls == "Fast" else "#3b82f6"
        with mcol:
            st.markdown(f"**{name}**")
            st.markdown(
                f"Predicted: <b style='color:{cls_color}'>{cls}</b> &nbsp; "
                f"<small>(P(Fast) = {p*100:.1f}%)</small>",
                unsafe_allow_html=True,
            )
            if name in folds:
                # CI auf Confidence-Skala (konsistent mit dem Overview-Chart)
                conf_lo, conf_hi = _confidence_range(folds[name])
                # min/max ueber Folds, ebenfalls in Confidence-Skala
                fold_confs = [max(f, 1 - f) for f in folds[name]]
                fmin, fmax = min(fold_confs), max(fold_confs)
                st.markdown(
                    f"Confidence: **{conf*100:.0f}%**  \n"
                    f"95% CI across folds: **{conf_lo*100:.1f}% – {conf_hi*100:.1f}%**  \n"
                    f"<small>min–max across folds: {fmin*100:.1f}% – {fmax*100:.1f}%</small>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"Confidence: **{conf*100:.0f}%**")
            auc, _ = expected_auc(name, "slopes+intercepts", miss, fu,
                                   score_mode=score_mode)
            ci_mean, ci_lo, ci_hi = expected_auc_ci(name, miss,
                                                     score_mode=score_mode)
            if auc is not None:
                rel_de, rel_color = reliability_label(auc)
                rel_en = {"hoch": "high", "mittel": "medium",
                           "niedrig": "low"}.get(rel_de, rel_de)
                if ci_lo is not None and ci_hi is not None:
                    st.markdown(
                        f"Expected AUC: "
                        f"<b style='color:{rel_color}'>{auc:.2f}</b> "
                        f"<small>[{ci_lo:.2f}, {ci_hi:.2f}]</small> ({rel_en})",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"Expected AUC: "
                        f"<b style='color:{rel_color}'>{auc:.2f}</b> ({rel_en})",
                        unsafe_allow_html=True,
                    )
    st.caption(
        "**Confidence** = how decided the model is about THIS patient. **CV-"
        "fold range** = spread of P(Fast) across the 5 calibration folds, a "
        "rough estimate of model-split variance. **Expected AUC** = how "
        "reliable the model is on average at this data quality (from the "
        "missingness × follow-up simulation), bracketed by the 95% bootstrap "
        "confidence interval at the current missingness level (1000 patient-"
        "level resamples). Likelihood Ratio is shown without fold range "
        "because it uses a single fit on the full PPMI cohort."
    )
    st.markdown("")

    # ---- Perzentil-Position
    st.markdown("##### Position in the PPMI cohort")
    st.caption(
        "For each measured score, the percentile of the patient's slope in the "
        "PPMI fast-progressor distribution and in the slow-progressor "
        "distribution. A high percentile means the patient's slope is steeper "
        "than most patients of that subtype; a low percentile means flatter."
    )
    reference = get_reference(score_mode)
    mtype, patient_idx = None, None
    patient_slopes = {}
    # Slope-Modus zuerst pruefen (gibt Slopes fuer Perzentile her)
    if "slope" in shap_ctx:
        feats_slope, _ = shap_ctx["slope"]
        idx_str = [str(x) for x in feats_slope.index]
        if selected in idx_str:
            pos = idx_str.index(selected)
            row = feats_slope.iloc[pos]
            for col in feats_slope.columns:
                if col.endswith("_slope") and pd.notna(row[col]):
                    patient_slopes[col[:-6]] = float(row[col])
            mtype = "slope"
            patient_idx = pos
    if mtype is None and "baseline" in shap_ctx:
        feats_base, _ = shap_ctx["baseline"]
        idx_str = [str(x) for x in feats_base.index]
        if selected in idx_str:
            mtype = "baseline"
            patient_idx = idx_str.index(selected)

    if patient_slopes:
        _percentile_panel(reference, patient_slopes, score_mode)
    else:
        st.caption("No slopes available (single-visit patient).")
    st.markdown("")

    # ---- SHAP-Bar pro Methode
    sh_head_l, sh_head_r = st.columns([4, 1], vertical_alignment="center")
    with sh_head_l:
        st.markdown("##### Why this prediction? Feature contributions")
    with sh_head_r:
        with st.popover(":material/info: Method", use_container_width=True):
            st.markdown(
                "SHAP values are averaged across all K=5 folds of the "
                "`CalibratedClassifierCV` ensemble, so the attribution is "
                "consistent with the ensemble's averaged prediction.\n\n"
                "Each fold's SHAP comes from the **underlying classifier**'s "
                "output space (probability for Random Forest, log-odds for "
                "XGBoost and Logistic Regression). Because the displayed "
                "probability also passes through an isotonic calibration step, "
                "the base value plus the sum of SHAP contributions does **not** "
                "exactly equal the displayed calibrated probability. The "
                "**relative direction and magnitude** of each feature's push "
                "are correctly attributed -- this is what the bars show.\n\n"
                "Faded bars mark features that were **imputed** (the score "
                "had to be filled in because the patient didn't have enough "
                "real measurements for it)."
            )
    st.caption(
        "Bars to the right (red) pushed the prediction towards **Fast**, "
        "bars to the left (blue) towards **Slow**."
    )
    if mtype is None:
        st.caption("No SHAP context for this patient.")
        return

    feats, models = shap_ctx[mtype]
    # Imputed-Flags fuer diesen Patient (Keys sind Feature-Codes wie 'MOCA_slope').
    imp_codes = stats.get("imputed", {})
    # SHAP feature_names sind pretty (z.B. 'MoCA (slope)'). Wir bauen ein Mapping.
    pretty_to_code = {}
    for col in feats.columns:
        if col.endswith("_slope"):
            base = col[:-6]
            pretty_to_code[f"{SCORE_LABELS.get(base, base)} (slope)"] = col
        elif col.endswith("_intercept"):
            base = col[:-10]
            pretty_to_code[f"{SCORE_LABELS.get(base, base)} (intercept)"] = col
        else:
            pretty_to_code[SCORE_LABELS.get(col, col)] = col
    imputed_lookup = {pretty: imp_codes.get(code, False)
                       for pretty, code in pretty_to_code.items()}

    # ML-Tabs (LR-Methode hat keine SHAP)
    ml_methods = [m for m in clf_cols if m in models]
    if not ml_methods:
        st.caption("SHAP not available for this model type.")
        return
    clf_tabs = st.tabs(ml_methods)
    for tab, clf_name in zip(clf_tabs, ml_methods):
        with tab:
            sv = get_shap(models[clf_name], feats,
                          f"{score_mode}_{clf_name}_{mtype}")
            if sv is None:
                continue
            patient_shap_bar(sv, patient_idx=patient_idx,
                              imputed_lookup=imputed_lookup, max_display=None)
