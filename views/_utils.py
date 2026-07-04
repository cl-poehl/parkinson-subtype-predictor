"""Shared helpers for demo.py and batch.py.
Collects per patient: predictions (all classifiers + LR method), per-fold
predictions for CI, missingness/follow-up, visit list, imputation flags,
percentiles against the PPMI subtype distributions.
Renders an overview plus a per-patient detail drilldown."""
import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.constants import (
    SCORE_LABELS, SCORE_RANGES, get_model_paths, get_conformal_paths,
    core_presence_route, fallback_scores, get_fallback_model_paths,
    get_fallback_conformal_paths, get_vennabers_paths,
    get_fallback_vennabers_paths,
)
from src.conformal import load_conformal_set, predict_sets
from src import vennabers
from src.features import (
    extract_slope_intercept, extract_baseline, imputation_flags,
    feature_reliability,
)
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
    """Per patient: missingness, follow-up, visit list, number of visits."""
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
    """LR method per patient. df_slope is the OLS slope feature DataFrame
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


def _store_vennabers(va_states, models, mean_df, patnos, patient_stats):
    """Per-patient Venn-Abers calibrated probability interval [p_lo, p_hi] (+ the
    merged point) for class Fast, from the deployed model's score, stored in
    patient_stats[patno]['va'][clf]."""
    for clf_name in models:
        if clf_name not in va_states or clf_name not in mean_df.columns:
            continue
        intervals = vennabers.predict_intervals(va_states[clf_name],
                                                 mean_df[clf_name].values)
        for pos, patno in enumerate(patnos):
            patient_stats[str(patno)].setdefault("va", {})[clf_name] = intervals[pos]


def run_predictions(df_in, score_mode, active_scores, imputer="knn"):
    """Complete prediction pipeline per patient.

    Returns (preds, shap_ctx, patient_stats, source_df).
    - preds: pd.DataFrame with patno, model_type, P(Fast) per classifier,
             LR method as an additional column 'Likelihood Ratio'.
    - shap_ctx: {mtype: (feats, models)} for SHAP computation.
    - patient_stats: {patno: {missing, follow_up, visit_times, n_visits,
                              imputed: {feat: bool}, folds: {clf: array},
                              lr_method: dict}}.
    - source_df: the original data (visit rows) for the trajectory plot.
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
        models = load_models(get_model_paths(score_mode, n_visits=2, imputer=imputer))
        conformals = load_conformal_set(get_conformal_paths(score_mode, n_visits=2, imputer=imputer))
        if models:
            mean_df, folds = predict_all_with_folds(models, feats)
            mean_df["model_type"] = "slope"
            mean_df["patno"] = mean_df.index.astype(str)
            out.append(mean_df.reset_index(drop=True))
            shap_ctx["slope"] = (feats, models)
            # Conformal prediction sets per model per patient
            for clf_name in models:
                if clf_name in conformals:
                    sets = predict_sets(conformals[clf_name], feats)
                    for pos, patno in enumerate(feats.index):
                        ps = patient_stats[str(patno)].setdefault("pred_sets", {})
                        ps[clf_name] = sets[pos] if sets else None

            # Assign folds per patient
            for pos, patno in enumerate(feats.index):
                patient_stats[str(patno)]["folds"] = {
                    clf: folds[clf][pos] for clf in folds
                }

            # Venn-Abers calibrated probability intervals
            va_states = vennabers.load_states(
                get_vennabers_paths(score_mode, n_visits=2, imputer=imputer))
            _store_vennabers(va_states, models, mean_df, list(feats.index),
                             patient_stats)

            # Imputation flags + reliability labels for the slope model
            imp = imputation_flags(multi, active_scores, mode="slope")
            for patno, ff in imp.items():
                patient_stats[str(patno)]["imputed"] = ff
            rel = feature_reliability(multi, active_scores, mode="slope")
            for patno, ll in rel.items():
                patient_stats[str(patno)]["reliability"] = ll

            # LR method
            lr_results = _compute_lr_predictions(feats, score_mode)
            # integrate into the preds DataFrame as an additional column
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
        models = load_models(get_model_paths(score_mode, n_visits=1, imputer=imputer))
        conformals = load_conformal_set(get_conformal_paths(score_mode, n_visits=1, imputer=imputer))
        if models:
            mean_df, folds = predict_all_with_folds(models, feats)
            mean_df["model_type"] = "baseline"
            mean_df["patno"] = mean_df.index.astype(str)
            mean_df["Likelihood Ratio"] = np.nan
            out.append(mean_df.reset_index(drop=True))
            shap_ctx["baseline"] = (feats, models)
            for clf_name in models:
                if clf_name in conformals:
                    sets = predict_sets(conformals[clf_name], feats)
                    for pos, patno in enumerate(feats.index):
                        ps = patient_stats[str(patno)].setdefault("pred_sets", {})
                        ps[clf_name] = sets[pos] if sets else None

            for pos, patno in enumerate(feats.index):
                patient_stats[str(patno)]["folds"] = {
                    clf: folds[clf][pos] for clf in folds
                }

            va_states = vennabers.load_states(
                get_vennabers_paths(score_mode, n_visits=1, imputer=imputer))
            _store_vennabers(va_states, models, mean_df, list(feats.index),
                             patient_stats)

            imp = imputation_flags(single, active_scores, mode="baseline")
            for patno, ff in imp.items():
                patient_stats[str(patno)]["imputed"] = ff
            rel = feature_reliability(single, active_scores, mode="baseline")
            for patno, ll in rel.items():
                patient_stats[str(patno)]["reliability"] = ll
            # No LR method for single-visit patients
            for patno in single["patno"].astype(str).unique():
                if patno in patient_stats:
                    patient_stats[patno]["lr_method"] = None

    if not out:
        return None, {}, {}, df_in

    full = pd.concat(out, ignore_index=True)
    _apply_core_fallback_routing(df, full, shap_ctx, patient_stats,
                                  score_mode, active_scores, imputer)
    return full, shap_ctx, patient_stats, df_in


def _apply_core_fallback_routing(df, full, shap_ctx, patient_stats,
                                  score_mode, active_scores, imputer):
    """Missing-core fallback routing (paper 3.4.2). Patients missing two or more
    MDS-UPDRS core parts are re-predicted with a model trained natively on the
    scores they actually have, overwriting their headline P(Fast), confidence
    folds and conformal set. The default full-feature SHAP context is kept (the
    auxiliary diagnostics need it); a route-specific context is added so the
    per-patient SHAP explains the prediction actually shown."""
    measured = {}
    for patno, g in df.groupby("patno"):
        meas = [s for s in active_scores
                if s in g.columns and g[s].notna().any()]
        r = core_presence_route(meas)
        if r is not None:
            measured[str(patno)] = r
    if not measured:
        return

    clf_cols = [c for c in full.columns
                if c not in ("patno", "model_type", "Likelihood Ratio")]
    full_idx = {str(p): i for i, p in enumerate(full["patno"])}

    # group routed patients by (visit class, pattern)
    groups = {}
    for patno, route in measured.items():
        nv = patient_stats.get(patno, {}).get("n_visits", 0)
        mtype = "slope" if nv >= 2 else "baseline"
        groups.setdefault((mtype, route), []).append(patno)

    for (mtype, route), patnos in groups.items():
        kept = fallback_scores(active_scores, route)
        gdf = df[df["patno"].astype(str).isin(patnos)]
        feats = (extract_slope_intercept(gdf, kept) if mtype == "slope"
                 else extract_baseline(gdf, kept))
        if feats.empty:
            continue
        n_visits = 2 if mtype == "slope" else 1
        models = load_models(get_fallback_model_paths(score_mode, n_visits, route,
                                                       imputer=imputer))
        if not models:
            continue
        conformals = load_conformal_set(
            get_fallback_conformal_paths(score_mode, n_visits, route, imputer=imputer))
        mean_df, folds = predict_all_with_folds(models, feats)
        sets_by_clf = {clf: predict_sets(conformals[clf], feats)
                       for clf in models if clf in conformals}
        va_states = vennabers.load_states(
            get_fallback_vennabers_paths(score_mode, n_visits, route, imputer=imputer))
        va_by_clf = {clf: vennabers.predict_intervals(va_states[clf],
                                                       mean_df[clf].values)
                     for clf in models if clf in va_states and clf in mean_df.columns}
        imp = imputation_flags(gdf, kept, mode=mtype)
        rel = feature_reliability(gdf, kept, mode=mtype)

        index_list = list(feats.index)
        for pos, patno in enumerate(index_list):
            sp = str(patno)
            row_i = full_idx.get(sp)
            if row_i is not None:
                for clf in clf_cols:
                    if clf in mean_df.columns:
                        full.at[row_i, clf] = float(mean_df.iloc[pos][clf])
            st_ = patient_stats.setdefault(sp, {})
            st_["route"] = route
            st_["route_scores"] = kept
            st_["folds"] = {clf: folds[clf][pos] for clf in folds}
            ps = st_.setdefault("pred_sets", {})
            for clf in models:
                if clf in sets_by_clf:
                    ps[clf] = sets_by_clf[clf][pos] if sets_by_clf[clf] else None
            va = st_.setdefault("va", {})
            for clf in va_by_clf:
                va[clf] = va_by_clf[clf][pos]
            if sp in imp:
                st_["imputed"] = imp[sp]
            if sp in rel:
                st_["reliability"] = rel[sp]
        shap_ctx[f"{mtype}::{route}"] = (feats, models)


# ----------------------- SHAP bar ---------------------------
def patient_shap_bar(sv, patient_idx=0, reliability_lookup=None,
                       max_display=None):
    """SHAP contributions as horizontal bars with a 3-level data-quality
    display per feature. reliability_lookup: dict feature_name -> str:
    'imputed' (kNN-filled), 'low' (exactly 2 measurements) or 'ok' (>=3).

    - 'ok'      bars: solid color, no stroke
    - 'low'     bars: 60% fill, thin dashed border
    - 'imputed' bars: 25% fill, thick dashed border
    """
    values = sv.values[patient_idx]
    abs_v = np.abs(values)
    order = np.argsort(abs_v)[::-1]
    if max_display is not None:
        order = order[:max_display]
    feat_names = [sv.feature_names[i] for i in order]
    vals = values[order]
    df = pd.DataFrame({"feature": feat_names, "shap": vals})

    if reliability_lookup is not None:
        rels = [reliability_lookup.get(sv.feature_names[i], "ok")
                for i in order]
        marks = []
        for r in rels:
            if r == "imputed":
                marks.append(" (imputed)")
            elif r == "low":
                marks.append(" (low-quality)")
            else:
                marks.append("")
        df["feature"] = [f + m for f, m in zip(df["feature"], marks)]
        df["reliability"] = rels
    else:
        df["reliability"] = "ok"

    df["direction"] = df["shap"].apply(lambda x: "Fast" if x >= 0 else "Slow")
    bound = max(abs_v.max() * 1.15, 0.01) if len(abs_v) else 0.01

    color_scale = alt.Scale(domain=["Slow", "Fast"],
                              range=["#3b82f6", "#ef4444"])
    base = alt.Chart(df).encode(
        y=alt.Y("feature:N", sort=df["feature"].tolist(),
                axis=alt.Axis(title=None, labelLimit=400)),
        x=alt.X("shap:Q",
                scale=alt.Scale(domain=[-bound, bound]),
                axis=alt.Axis(title="SHAP value   (← Slow      Fast →)")),
        color=alt.Color("direction:N", scale=color_scale, legend=None),
        tooltip=["feature", alt.Tooltip("shap:Q", format=".3f"),
                  "direction", "reliability"],
    )
    ok = base.transform_filter("datum.reliability === 'ok'").mark_bar(
        fillOpacity=1.0, strokeWidth=0)
    low = base.transform_filter("datum.reliability === 'low'").mark_bar(
        fillOpacity=0.60, stroke="#374151", strokeWidth=0.8,
        strokeDash=[2, 2])
    imputed = base.transform_filter("datum.reliability === 'imputed'").mark_bar(
        fillOpacity=0.25, stroke="#374151", strokeWidth=1.4,
        strokeDash=[4, 3])
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="black").encode(x="x:Q")
    chart = (ok + low + imputed + rule).properties(
        height=max(26 * len(df), 200))
    st.altair_chart(chart, width="stretch")


# ----------------------- Score trajectories -----------------
def score_trajectory_plot(source_df, patno, active_scores):
    """Line-chart grid: one small chart per score, x = disease duration,
    y = score value, points for individual visits."""
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

    # Order by the score list
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
    st.altair_chart(chart, width="content")


# ----------------------- Visit list, CI, percentiles ----------
def _ci_from_folds(folds_array, ci=0.95):
    """95% confidence interval of the mean across the K=5 CV folds:
    mean ± z * std/sqrt(K) (z=1.96 for 95%). Returns (lo, hi) in P(Fast) units,
    clipped to [0, 1]. For a point cloud without spread (all folds equal)
    the range is zero."""
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
    """Full min-max as additional info in the detail panel."""
    if folds_array is None or len(folds_array) == 0:
        return (np.nan, np.nan)
    return float(np.nanmin(folds_array)), float(np.nanmax(folds_array))


def _confidence_range(folds_array):
    """95% CI of the confidence max(p,1-p) across the folds, based on the
    95% CI of P(Fast). If the CI straddles 0.5, the lower bound is
    0.5 (confidence cannot be lower than a coin flip)."""
    if folds_array is None or len(folds_array) == 0:
        return (np.nan, np.nan)
    p_lo, p_hi = _ci_from_folds(folds_array)
    if p_lo >= 0.5:
        return (p_lo, p_hi)
    if p_hi <= 0.5:
        return (1 - p_hi, 1 - p_lo)
    return (0.5, max(p_hi, 1 - p_lo))


def _percentile_panel(reference, slopes_dict, score_mode):
    """Shows for each score with a slope: percentile in the fast vs slow subtype.
    Table with score, slope, percentile-fast, percentile-slow."""
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
    st.dataframe(df_show, width="stretch", hide_index=True)


# ----------------------- Main view -----------------------
# ----------------------- Cohort aggregate views -----------------------
def _predicted_class_map(preds):
    """patno (str) -> 'Fast'/'Slow' from the consensus."""
    return dict(zip(preds["patno"].astype(str), preds["klasse"]))


def _pretty_feature(code):
    if code.endswith("_slope"):
        return f"{SCORE_LABELS.get(code[:-6], code[:-6])} (slope)"
    if code.endswith("_intercept"):
        return f"{SCORE_LABELS.get(code[:-10], code[:-10])} (intercept)"
    return SCORE_LABELS.get(code, code)


def cohort_trajectory_plot(source_df, preds, active_scores, bin_width=12):
    """Mean score trajectories over Disease_duration, split by
    predicted subtype, with an SEM band. Visits are binned onto a fixed
    time grid (default 12 months) so that patients become comparable
    (timestamp binning)."""
    class_map = _predicted_class_map(preds)
    df = source_df.copy()
    df["patno"] = df["patno"].astype(str)
    df["Predicted"] = df["patno"].map(class_map)
    df = df[df["Predicted"].notna()]
    measured = [s for s in active_scores
                if s in df.columns and df[s].notna().any()]
    if df.empty or not measured:
        st.caption("No trajectory data available.")
        return

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        default = ([s for s in ["UPDRS2", "UPDRS3_on", "MOCA"] if s in measured]
                   or measured[:1])
        pick = st.multiselect(
            "Scores to plot", options=measured,
            format_func=lambda s: SCORE_LABELS.get(s, s),
            default=default, key="cohort_traj_scores")
    with c2:
        bin_width = st.selectbox("Bin width (months)", [6, 12, 24],
                                  index=1, key="cohort_traj_bin")
    with c3:
        min_n = st.selectbox("Min patients/bin", [1, 3, 5],
                              index=1, key="cohort_traj_minn",
                              help="Bins backed by fewer patients than this are "
                                   "hidden, so a single case is never drawn as a "
                                   "cohort trend.")
    if not pick:
        return

    # Bin onto a fixed grid; round() centers the bins on the grid.
    df["bin"] = (df["disease_duration"] / bin_width).round() * bin_width
    rows = []
    for s in pick:
        sub = df[["patno", "bin", "Predicted", s]].dropna(subset=[s])
        # Step 1: average per patient per bin -> the unit of observation is the
        # patient, not the individual visit (otherwise a patient with two
        # visits in the same bin counts twice and the SEM becomes artificially small).
        per_pat = (sub.groupby(["patno", "bin", "Predicted"])[s]
                   .mean().reset_index())
        for _, r in per_pat.iterrows():
            rows.append({"Score": SCORE_LABELS.get(s, s), "bin": float(r["bin"]),
                         "Predicted": r["Predicted"], "Value": float(r[s])})
    long_df = pd.DataFrame(rows)
    # Step 2: aggregate over patients; n = number of patients in the bin.
    agg = (long_df.groupby(["Score", "bin", "Predicted"])["Value"]
           .agg(mean="mean", sd="std", n="count").reset_index())
    n_dropped = int((agg["n"] < min_n).sum())
    agg = agg[agg["n"] >= min_n].copy()
    if agg.empty:
        st.caption(f"No time bin has ≥{min_n} patients for the selected scores. "
                    f"Lower the threshold or widen the bin.")
        return
    agg["sem"] = (agg["sd"] / np.sqrt(agg["n"].clip(lower=1))).fillna(0.0)
    agg["lo"] = agg["mean"] - agg["sem"]
    agg["hi"] = agg["mean"] + agg["sem"]

    color = alt.Color("Predicted:N",
                       scale=alt.Scale(domain=["Fast", "Slow"],
                                        range=["#ef4444", "#3b82f6"]),
                       legend=alt.Legend(title="Predicted subtype", orient="top"))
    base = alt.Chart(agg)
    band = base.mark_area(opacity=0.18).encode(
        x=alt.X("bin:Q", title="Disease duration (months, binned)"),
        y=alt.Y("lo:Q", title="Mean score ± SEM"), y2="hi:Q", color=color)
    line = base.mark_line(point=True).encode(
        x="bin:Q", y=alt.Y("mean:Q"), color=color,
        tooltip=["Score", alt.Tooltip("bin:Q", title="Month"), "Predicted",
                 alt.Tooltip("mean:Q", format=".2f"),
                 alt.Tooltip("n:Q", title="n visits")])
    chart = ((band + line).properties(height=190, width=260)
             .facet(facet=alt.Facet("Score:N", title=None), columns=2)
             .resolve_scale(y="independent"))
    st.altair_chart(chart, width="stretch")
    drop_txt = (f" {n_dropped} sparse bin(s) with <{min_n} patients were hidden."
                if n_dropped else "")
    st.caption(
        f"Mean clinical-score trajectory by **predicted** subtype. Visits are "
        f"binned to a {bin_width}-month grid; **each patient contributes one "
        f"averaged value per bin** (a patient is never counted twice), so the "
        f"shaded ±1 SEM band reflects spread across *patients*, not visits. "
        f"n in the tooltip is the patient count behind each point.{drop_txt} "
        f"Fast/Slow curves that diverge are face validity that the predicted "
        f"label tracks genuine progression, not just baseline level.")


def cohort_shap_beeswarm(shap_ctx, score_mode):
    """Aggregated SHAP beeswarm across the cohort: per feature a
    strip/jitter of the SHAP values of all patients, colored by
    feature value. Shows which scores drive the cohort predictions."""
    mtype = ("slope" if "slope" in shap_ctx
             else "baseline" if "baseline" in shap_ctx else None)
    if mtype is None:
        st.caption("No SHAP context available.")
        return
    feats, models = shap_ctx[mtype]
    ml_methods = [m for m in ["Random Forest", "XGBoost", "Logistic Regression"]
                  if m in models]
    if not ml_methods:
        st.caption("No SHAP-capable model in this run.")
        return
    if len(feats) < 2:
        st.caption("Cohort beeswarm needs ≥2 patients sharing a visit profile "
                    "(single-visit and multi-visit patients are explained "
                    "separately).")
        return
    clf_name = st.selectbox("Model", ml_methods, key="cohort_shap_model")
    sv = get_shap(models[clf_name], feats, f"{score_mode}_{clf_name}_{mtype}")
    if sv is None:
        st.caption("SHAP unavailable for this model.")
        return
    vals = sv.values
    fnames = list(sv.feature_names)
    fv = feats.values.astype(float)
    rows = []
    for j, fn in enumerate(fnames):
        col = fv[:, j]
        cmin, cmax = np.nanmin(col), np.nanmax(col)
        rng = (cmax - cmin) or 1.0
        pname = _pretty_feature(fn)
        for i in range(vals.shape[0]):
            rows.append({"feature": pname, "shap": float(vals[i, j]),
                         "fval": float((col[i] - cmin) / rng)})
    bdf = pd.DataFrame(rows)
    order = (bdf.assign(a=bdf["shap"].abs()).groupby("feature")["a"].mean()
             .sort_values(ascending=False).index.tolist())
    pts = (alt.Chart(bdf).transform_calculate(jitter="random()")
           .mark_circle(size=26, opacity=0.55).encode(
        y=alt.Y("feature:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=400)),
        x=alt.X("shap:Q", title="SHAP value   (← Slow      Fast →)"),
        yOffset=alt.YOffset("jitter:Q"),
        color=alt.Color("fval:Q",
                        scale=alt.Scale(scheme="redblue", reverse=True),
                        legend=alt.Legend(title="Feature value (low→high)",
                                           orient="right", gradientLength=120)),
        tooltip=["feature", alt.Tooltip("shap:Q", format=".3f")])
        .properties(height=max(30 * len(order), 220)))
    rule = (alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(color="black").encode(x="x:Q"))
    st.altair_chart(pts + rule, width="stretch")
    st.caption(
        f"Each dot is one patient's SHAP contribution for that feature "
        f"({clf_name}, {mtype} model, n={len(feats)} patients). Dots right of "
        f"zero push toward Fast, left toward Slow; color encodes whether the "
        f"feature value was low (blue) or high (red). Features are ordered by "
        f"mean |SHAP| — the cohort's strongest drivers sit at the top.")


def cohort_abstention_summary(patient_stats):
    """Conformal abstention overview: per model, how many patients get a
    decisive {Fast}/{Slow} vs. {Fast, Slow} = 'don't know'
    (90% coverage)."""
    cats = ["Decisive Fast", "Decisive Slow", "Abstain {Fast, Slow}", "Empty set"]
    methods = {}
    for stt in patient_stats.values():
        for m, cset in (stt.get("pred_sets") or {}).items():
            d = methods.setdefault(m, {c: 0 for c in cats})
            if cset is None:
                continue
            s = set(cset)
            if s == {"Fast", "Slow"}:
                d["Abstain {Fast, Slow}"] += 1
            elif s == {"Fast"}:
                d["Decisive Fast"] += 1
            elif s == {"Slow"}:
                d["Decisive Slow"] += 1
            else:
                d["Empty set"] += 1
    if not methods:
        st.caption("No conformal prediction sets available for this run.")
        return
    rows, totals = [], {}
    for m, d in methods.items():
        tot = sum(d.values()) or 1
        totals[m] = tot
        for c in cats:
            if d[c]:
                rows.append({"Method": m, "Outcome": c, "count": d[c],
                             "frac": d[c] / tot})
    cdf = pd.DataFrame(rows)
    palette = {"Decisive Fast": "#ef4444", "Decisive Slow": "#3b82f6",
               "Abstain {Fast, Slow}": "#9ca3af", "Empty set": "#1f2937"}
    chart = (alt.Chart(cdf).mark_bar().encode(
        y=alt.Y("Method:N", title=None),
        x=alt.X("frac:Q", stack="normalize", axis=alt.Axis(format="%"),
                title="Share of cohort"),
        color=alt.Color("Outcome:N",
                        scale=alt.Scale(domain=cats,
                                         range=[palette[c] for c in cats]),
                        legend=alt.Legend(title="90% prediction set", orient="top")),
        order=alt.Order("Outcome:N"),
        tooltip=["Method", "Outcome", "count",
                 alt.Tooltip("frac:Q", format=".0%")])
        .properties(height=28 * len(methods) + 40))
    st.altair_chart(chart, width="stretch")
    ab = {m: methods[m]["Abstain {Fast, Slow}"] / totals[m] for m in methods}
    worst = max(ab, key=ab.get)
    st.caption(
        f"At the 90% coverage level, the conformal predictor returns a single "
        f"decisive label for most patients and abstains ({{Fast, Slow}}) on the "
        f"borderline ones. Abstention rate ranges "
        f"{min(ab.values())*100:.0f}–{max(ab.values())*100:.0f}% across methods "
        f"(highest for {worst}). Abstained cases are exactly where the clinician "
        f"should not lean on the model.")


def cohort_dataquality_summary(patient_stats):
    """Data-quality overview per feature across the cohort: measured
    (≥3 visits) / low-quality (2 visits) / imputed (0-1 visit)."""
    labels = {"ok": "Measured (≥3 visits)", "low": "Low-quality (2 visits)",
              "imputed": "Imputed (0-1 visit)"}
    counts = {}
    for stt in patient_stats.values():
        for feat, q in (stt.get("reliability") or {}).items():
            d = counts.setdefault(feat, {"ok": 0, "low": 0, "imputed": 0})
            d[q if q in d else "ok"] += 1
    if not counts:
        st.caption("No data-quality information available.")
        return
    rows = []
    for feat, d in counts.items():
        tot = sum(d.values()) or 1
        for q in ("ok", "low", "imputed"):
            if d[q]:
                rows.append({"Feature": _pretty_feature(feat),
                             "Quality": labels[q], "count": d[q],
                             "frac": d[q] / tot, "imp": d["imputed"] / tot})
    qdf = pd.DataFrame(rows)
    order = (qdf.groupby("Feature")["imp"].first()
             .sort_values(ascending=False).index.tolist())
    dom = list(labels.values())
    chart = (alt.Chart(qdf).mark_bar().encode(
        y=alt.Y("Feature:N", sort=order, title=None,
                axis=alt.Axis(labelLimit=400)),
        x=alt.X("frac:Q", stack="normalize", axis=alt.Axis(format="%"),
                title="Share of patients"),
        color=alt.Color("Quality:N",
                        scale=alt.Scale(domain=dom,
                                         range=["#10b981", "#f59e0b", "#9ca3af"]),
                        legend=alt.Legend(title="Feature provenance", orient="top")),
        order=alt.Order("Quality:N"),
        tooltip=["Feature", "Quality", "count",
                 alt.Tooltip("frac:Q", format=".0%")])
        .properties(height=max(20 * len(order), 200)))
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Per feature, the share of the uploaded cohort for which the "
        "slope/intercept was directly measured, derived from only two visits, "
        "or kNN-imputed. Features are ordered by imputation rate — the ones "
        "most often filled in are where the model leans hardest on the "
        "missingness-handling pipeline.")


def render_cohort_panels(preds, patient_stats, shap_ctx, source_df,
                          active_scores, score_mode):
    """Aggregate views across the entire uploaded cohort (>1 patient)."""
    st.divider()
    st.markdown("### Cohort overview")
    st.caption("Aggregate views across all uploaded patients. Use the "
                "per-patient detail below to drill into any single case.")

    if source_df is not None and active_scores is not None:
        st.markdown("##### Mean score trajectories by predicted subtype")
        cohort_trajectory_plot(source_df, preds, active_scores)

    if patient_stats:
        st.markdown("##### Decisiveness — conformal prediction sets")
        cohort_abstention_summary(patient_stats)

    if shap_ctx:
        st.markdown("##### What drives the cohort? (SHAP beeswarm)")
        cohort_shap_beeswarm(shap_ctx, score_mode)

    if patient_stats:
        st.markdown("##### Data quality across the cohort")
        cohort_dataquality_summary(patient_stats)


def render_results(preds, source_name, shap_ctx=None, score_mode="luxpark",
                    patient_stats=None, source_df=None, active_scores=None):
    """Complete results section."""
    clf_cols = [c for c in preds.columns if c not in (
        "patno", "model_type", "Likelihood Ratio"
    )]
    has_lr = "Likelihood Ratio" in preds.columns
    all_method_cols = clf_cols + (["Likelihood Ratio"] if has_lr else [])

    # Consensus: mean of all available methods (NaN-safe)
    consensus = preds[all_method_cols].mean(axis=1, skipna=True)
    preds = preds.assign(consensus=consensus,
                          klasse=consensus.apply(
                              lambda x: "Fast" if x >= 0.5 else "Slow"))

    n = len(preds)
    n_fast = int((preds["consensus"] >= 0.5).sum())
    n_slow = n - n_fast
    mean_conf_score = preds["consensus"].apply(lambda x: max(x, 1 - x)).mean()

    st.markdown(f"### Results  \n*Source: {source_name}*")

    # The cohort header and overview chart only make sense with multiple patients.
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

    # ---- Overview chart: confidence per patient per model
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
                # CI only for ML models (LR has no folds)
                folds = (patient_stats or {}).get(patno, {}).get("folds", {})
                if c in folds and len(folds[c]) > 0:
                    conf_lo, conf_hi = _confidence_range(folds[c])
                    # Min/max across the folds in confidence space as whisker points
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
        # Patient order as in the input (preds), not sorted by consensus
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

        Y_AXIS = alt.Axis(format="%", title="Certainty in predicted class")
        Y_SCALE = alt.Scale(domain=[0.5, 1.0])
        errorbars = (
            alt.Chart(long_df)
            .mark_errorbar(thickness=1.5)
            .encode(
                x=alt.X("patno:N", sort=patno_order),
                y=alt.Y("conf_lo:Q", scale=Y_SCALE, axis=Y_AXIS),
                y2=alt.Y2("conf_hi:Q"),
                color=alt.Color("Method:N",
                                scale=alt.Scale(domain=method_order,
                                                 range=[method_palette[m]
                                                         for m in method_order]),
                                legend=None),
                xOffset=alt.XOffset("Method:N"),
            )
        )
        # Whisker points for min and max across the folds (box-plot-like
        # display)
        whisker_min = (
            alt.Chart(long_df)
            .mark_point(filled=False, size=40, strokeWidth=1.5, opacity=0.7)
            .encode(
                x=alt.X("patno:N", sort=patno_order),
                y=alt.Y("conf_min:Q", scale=Y_SCALE, axis=Y_AXIS),
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
            y=alt.Y("conf_max:Q", scale=Y_SCALE, axis=Y_AXIS),
            tooltip=["patno", "Method",
                     alt.Tooltip("conf_max:Q", format=".1%", title="Max")],
        )
        points = (
            alt.Chart(long_df)
            .mark_point(filled=True, size=110, opacity=0.9)
            .encode(
                x=alt.X("patno:N", sort=patno_order,
                        axis=alt.Axis(labelAngle=-40, title="Patient")),
                y=alt.Y("confidence:Q", scale=Y_SCALE, axis=Y_AXIS),
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
            width="stretch",
        )
        st.markdown("")

    # ---- Table: only meaningful with >1 patient, otherwise redundant with the detail panel
    if n > 1:
        st.caption(
            "Each method column shows **P(Fast progression)** -- the raw "
            "model probability that the patient is a fast progressor. The "
            "**Consensus** column is the mean across all available methods, "
            "**Class** is the resulting prediction (threshold 50%)."
        )
        pretty_cols = ["patno", "klasse", "consensus"] + all_method_cols
        if "model_type" in preds.columns:
            pretty_cols.append("model_type")
        pretty = preds[pretty_cols].copy()
        for c in all_method_cols:
            pretty[c] = pretty[c].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
            )
        pretty["consensus"] = pretty["consensus"].apply(lambda x: f"{x*100:.1f}%")
        # Append ' P(Fast)' suffix to the method columns for clarity
        rename_map = {
            "patno": "Patient",
            "consensus": "Consensus P(Fast)",
            "klasse": "Class",
            "model_type": "Model type",
        }
        for m in all_method_cols:
            rename_map[m] = f"{m} P(Fast)"
        pretty = pretty.rename(columns=rename_map)
        st.dataframe(pretty, width="stretch", hide_index=True)

        buf = io.StringIO()
        preds.drop(columns=["klasse"]).to_csv(buf, index=False)
        st.download_button(
            "Download results as CSV", buf.getvalue(),
            file_name="subtype_predictions.csv", mime="text/csv",
        )

    # ---- Cohort aggregate views (only meaningful with multiple patients)
    if n > 1:
        render_cohort_panels(preds, patient_stats or {}, shap_ctx or {},
                              source_df, active_scores, score_mode)

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
        selected = ordered_ids[0]

    with st.expander(":material/menu_book: **How to read this patient's "
                       "detail panel** (what each block tells you)",
                       expanded=False):
        st.markdown(
            """
            The blocks below answer different questions about this
            patient's prediction:

            **1. Score trajectories.** The raw clinical scores plotted
            against disease duration. Lets you visually confirm whether
            the patient looks like a clear fast/slow case or a
            borderline one.

            **2. Predictions per method.** Four cards, one per method
            (Random Forest, XGBoost, Logistic Regression, Reference
            Likelihood Ratio). Each card shows:

            - **90% Set** -- the Conformal prediction set
              (Vovk et al. 2005). If a single label, the model is
              decisive; if `{Fast, Slow}`, the model defers. The set is
              guaranteed to contain the true label in ≥90% of PPMI
              patients (under exchangeability).
            - **P(Fast)** -- the isotonically calibrated probability.
              Mean it as in 'in 100 PPMI patients with this prediction,
              about P would be Fast'.
            - **Confidence** = max(P, 1-P), with the 95% CI across
              the 5 calibration folds in brackets. Wide CI = model
              folds disagree; narrow = stable prediction.
            - **Expected AUC** = approximate discrimination of this
              method on PPMI patients with this missingness and
              follow-up profile. Low expected AUC = the method tends
              to do poorly in this regime, take the prediction with
              caution.

            **3. Likelihood Ratio per-score breakdown** (only if LR is
            available). Shows which clinical scores drove the LR
            method's prediction up or down. The LR has a known weakness:
            when one score is far in standard-deviation units from the
            Slow mean, its log10(LR) saturates at the +/-1.3 cap, which
            can override the joint signal from all other scores. This
            is why RF and XGBoost are often more robust on borderline
            patients.

            **4. Position in the PPMI cohort.** For each measured score,
            the patient's slope is compared with the distribution of
            slopes in Fast and Slow PPMI patients. A 75th-percentile
            position in the Fast distribution means 'this patient's
            slope is steeper than 75% of PPMI Fast patients' -- so
            the slope is faster than typical Fast.

            **5. Why this prediction? (SHAP).** SHAP values quantify
            each feature's contribution to *this patient's* prediction,
            relative to a baseline prediction averaged over PPMI. Red
            bars push toward Fast, blue toward Slow. **Bar styling
            encodes data quality**: solid = >=3 visits (statistically
            sound slope), dashed thin = exactly 2 visits (degenerate
            slope, mathematically OK but no residual information),
            dashed thick with light fill = imputed (kNN-filled from
            other patients).

            **6. Scientific context for this prediction.** Five
            diagnostics that put the prediction in a research-grade
            context: calibration anchor (PPMI patients with similar
            predictions, what fraction were Fast?), threshold table
            (where would this patient flip class?), noise robustness
            (P(Fast) range under 10% input perturbation), survival
            prediction (months to Hoehn-Yahr 3 from a Cox model with
            c-index 0.87), and simple-baseline comparison
            (UPDRS3-only / MoCA-only models).

            **7. Counterfactuals.** For each feature, the smallest
            change that would flip the predicted class -- helps to
            answer 'what would have to be different about this patient
            for the model to say the opposite?'
            """
        )

    sel_row = preds[preds["patno"].astype(str) == selected].iloc[0]
    sel_consensus = float(sel_row["consensus"])
    sel_class = "Fast" if sel_consensus >= 0.5 else "Slow"
    sel_color = "#ef4444" if sel_class == "Fast" else "#3b82f6"

    stats = (patient_stats or {}).get(selected, {})
    miss = stats.get("missing", 0)
    fu = stats.get("follow_up", 0)
    visit_times = stats.get("visit_times", [])
    n_visits = stats.get("n_visits", 0)

    # Patient summary header
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
    _route = stats.get("route")
    if _route:
        _kept_core = {"coreI": "MDS-UPDRS I", "coreII": "MDS-UPDRS II",
                      "coreIII": "MDS-UPDRS III",
                      "coreNone": "no MDS-UPDRS core part"}.get(_route, _route)
        st.info(
            f":material/alt_route: **Fallback model used.** Two or more "
            f"MDS-UPDRS core parts are missing for this patient ({_kept_core} "
            f"remains), so instead of imputing them the prediction, confidence "
            f"and conformal set above come from a model **trained natively on "
            f"the scores actually present** — which is significantly more "
            f"accurate in this regime (paper §3.4.2). The SHAP explanation "
            f"below is from that same fallback model."
        )
    st.markdown("")

    # ---- Score trajectories
    if source_df is not None and active_scores is not None:
        st.markdown("##### Score trajectories")
        st.caption("One small chart per measured score. Filled scores only -- "
                    "unmeasured scores are not shown.")
        score_trajectory_plot(source_df, selected, active_scores)
        st.markdown("")

    # ---- Method detail with confidence + bootstrap CI + expected AUC
    st.markdown("##### Predictions per method")
    st.caption(
        "This prediction comes from the model trained on the **full** PPMI "
        "cohort (all patients). The accuracy/calibration numbers on the About "
        "page are a separate, **conservative cross-validated estimate** of how "
        "this deployed model generalises — they are not produced by the model "
        "making this prediction. The **Venn-Abers** interval next to each "
        "P(Fast) is the calibrated probability interval for *this* patient "
        "(distribution-free, from the same calibration set as the conformal "
        "set); the small fold-stability range is only an informal cue, not the "
        "10-fold reporting CV."
    )
    methods_to_show = [m for m in all_method_cols if pd.notna(sel_row[m])]
    metric_cols = st.columns(len(methods_to_show))
    folds = stats.get("folds", {})
    pred_sets = stats.get("pred_sets", {})
    for mcol, name in zip(metric_cols, methods_to_show):
        p = float(sel_row[name])
        conf = max(p, 1 - p)
        cls = "Fast" if p >= 0.5 else "Slow"
        cls_color = "#ef4444" if cls == "Fast" else "#3b82f6"
        # Conformal Prediction Set
        cset = pred_sets.get(name)
        with mcol:
            st.markdown(f"**{name}**")
            if cset is not None:
                if len(cset) == 1:
                    set_color = "#ef4444" if cset[0] == "Fast" else "#3b82f6"
                    st.markdown(
                        f"90% Set: "
                        f"<b style='color:{set_color}'>"
                        f"{{ {cset[0]} }}</b>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "90% Set: <b style='color:#9ca3af'>"
                        "{ Fast, Slow }</b> &nbsp;<small>(uncertain)</small>",
                        unsafe_allow_html=True,
                    )
            va_iv = stats.get("va", {}).get(name)
            if va_iv is not None:
                vlo, vhi, _vp = va_iv
                pfast_html = (f"P(Fast) = {p*100:.1f}% "
                              f"<small>[Venn-Abers {vlo*100:.0f}–{vhi*100:.0f}%]"
                              f"</small>, predicted ")
            else:
                pfast_html = f"P(Fast) = {p*100:.1f}%, predicted "
            st.markdown(
                f"<small>{pfast_html}"
                f"<b style='color:{cls_color}'>{cls}</b></small>",
                unsafe_allow_html=True,
            )
            # 5-fold spread is an informal stability cue, not a calibrated CI.
            if name in folds:
                conf_lo, conf_hi = _confidence_range(folds[name])
                st.markdown(
                    f"Confidence: **{conf*100:.0f}%** "
                    f"<small>(fold stability {conf_lo*100:.0f}–{conf_hi*100:.0f}%)"
                    f"</small>",
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
        "**Reading guide for each card** (research conventions):\n\n"
        "- **90% Set** is the *Conformal prediction set* (Vovk et al. "
        "2005; MAPIE Split-Conformal, LAC conformity score). When the "
        "model is decisive it returns a single label; when it cannot "
        "discriminate it returns `{Fast, Slow}` and defers. Across "
        "PPMI patients this set covers the true label in ≥90% of "
        "cases (assuming exchangeability of calibration and test data).\n"
        "- **P(Fast)** is the isotonically calibrated probability "
        "that the patient is a Fast progressor, with the **Venn-Abers** "
        "calibrated probability interval (Vovk & Petej 2014) in "
        "brackets: a distribution-free interval, fit on the same "
        "calibration split as the conformal set, whose width reflects "
        "how tightly the calibration data pin down this probability. "
        "Frequentist interpretation: of 100 PPMI patients with this "
        "prediction, about P would actually be Fast.\n"
        "- **Confidence** = max(P, 1-P): how decisively the model "
        "places the patient on one side of 0.5. The small *fold-stability* "
        "range is the spread across the deployed model's 5 internal "
        "calibration folds -- an informal cue to whether the folds agreed, "
        "not a calibrated interval (the Venn-Abers interval above is the "
        "calibrated uncertainty).\n"
        "- **Expected AUC** is the cross-validated discriminative "
        "AUC of this method on PPMI patients matched to this "
        "patient's missingness and follow-up profile (1000-bootstrap "
        "CI). Low expected AUC indicates the method tends to "
        "underperform in this regime -- treat the prediction with "
        "appropriate caution.\n\n"
        "*The Reference Likelihood Ratio method is fit once on the "
        "full PPMI cohort, so it has neither a fold-based confidence "
        "CI nor a Conformal set.*"
    )
    st.markdown("")

    # ---- Likelihood-Ratio Per-Score-Breakdown
    lr_res = stats.get("lr_method")
    if lr_res and lr_res.get("per_score"):
        _lr_breakdown_panel(lr_res, score_mode)
        st.markdown("")

    # ---- Percentile position
    st.markdown("##### Position in the PPMI cohort")
    st.caption(
        "**How to read.** For each score, this patient's slope is "
        "compared against the distribution of slopes in PPMI Fast and "
        "PPMI Slow patients separately. The two percentile values "
        "answer 'where does this patient sit within each subtype?'. "
        "A patient who tracks the typical Slow trajectory will have a "
        "near-median percentile in the Slow column (~50) and a low "
        "percentile in the Fast column (<25, slower than most Fast). "
        "A patient on the borderline -- e.g. SLOW with a slightly "
        "steep PIGD slope -- might land at the 60th percentile of "
        "Slow but already the 30th percentile of Fast. Useful to "
        "spot which specific scores drive a borderline classification.\n\n"
        "Replaces the original raw-slope display: percentiles "
        "normalise across scores that have very different natural "
        "ranges (UPDRS-3 spans 0-132 while MoCA spans 0-30)."
    )
    reference = get_reference(score_mode)
    mtype, patient_idx = None, None
    patient_slopes = {}
    # Check the slope mode first (it provides the slopes for percentiles)
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

    # ---- SHAP bar per method
    sh_head_l, sh_head_r = st.columns([4, 1], vertical_alignment="center")
    with sh_head_l:
        st.markdown("##### Why this prediction? Feature contributions")
    with sh_head_r:
        with st.popover(":material/info: Method", width="stretch"):
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

    # Missing-core fallback (paper 3.4.2): explain with the model that actually
    # produced the prediction shown above, not the full-feature model.
    route = stats.get("route")
    ctx_key = f"{mtype}::{route}" if route and f"{mtype}::{route}" in shap_ctx else mtype
    feats, models = shap_ctx[ctx_key]
    if route:
        idx_str = [str(x) for x in feats.index]
        if selected in idx_str:
            patient_idx = idx_str.index(selected)
    # Reliability lookup per feature (keys are codes like 'MOCA_slope',
    # values are 'imputed' | 'low' | 'ok').
    rel_codes = stats.get("reliability", {})
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
    reliability_lookup = {pretty: rel_codes.get(code, "ok")
                            for pretty, code in pretty_to_code.items()}

    # ML tabs (the LR method has no SHAP)
    ml_methods = [m for m in clf_cols if m in models]
    if not ml_methods:
        st.caption("SHAP not available for this model type.")
        return
    n_ok = sum(1 for v in reliability_lookup.values() if v == "ok")
    n_low = sum(1 for v in reliability_lookup.values() if v == "low")
    n_imp = sum(1 for v in reliability_lookup.values() if v == "imputed")
    total = len(reliability_lookup)
    # Inline legend with the 3 quality levels
    st.markdown(
        f"<small><b>Data quality for this patient:</b> "
        f"&nbsp; <span style='display:inline-block;width:14px;height:14px;"
        f"background:#9ca3af;vertical-align:middle;'></span> "
        f"<b>{n_ok}</b> measured (≥3 visits, solid bars) &nbsp; | &nbsp; "
        f"<span style='display:inline-block;width:14px;height:14px;"
        f"background:#9ca3af;opacity:0.6;border:1px dashed #374151;"
        f"vertical-align:middle;'></span> "
        f"<b>{n_low}</b> low-quality (exactly 2 visits, dashed thin "
        f"border) &nbsp; | &nbsp; "
        f"<span style='display:inline-block;width:14px;height:14px;"
        f"background:#9ca3af;opacity:0.25;border:1.5px dashed #374151;"
        f"vertical-align:middle;'></span> "
        f"<b>{n_imp}</b> imputed (0-1 visit, kNN-filled, dashed thick "
        f"border) &nbsp; -- of {total} features total.</small>",
        unsafe_allow_html=True,
    )
    clf_tabs = st.tabs(ml_methods)
    for tab, clf_name in zip(clf_tabs, ml_methods):
        with tab:
            sv = get_shap(models[clf_name], feats,
                          f"{score_mode}_{clf_name}_{ctx_key}")
            if sv is None:
                continue
            patient_shap_bar(sv, patient_idx=patient_idx,
                              reliability_lookup=reliability_lookup,
                              max_display=None)

    # ---- Patient diagnostics: calibration anchor, thresholds, noise,
    # survival, baselines
    st.markdown("##### Scientific context for this prediction")
    st.caption(
        "Five diagnostics that put the patient's prediction in context, "
        "based on the published PPMI cross-validation: (a) **Calibration "
        "anchor** -- of patients with similar predictions in PPMI, how many "
        "actually were Fast? (b) **Threshold table** -- at which cutoff "
        "would this patient flip class? (c) **Noise robustness** -- how "
        "stable is the prediction under realistic measurement noise? "
        "(d) **Time-to-milestone** -- expected months until H&Y stage 3 "
        "from a Cox model. (e) **Simple-baseline comparison** -- what "
        "would a single-feature model say?"
    )
    _patient_diagnostics_panel(selected, sel_row, methods_to_show, clf_cols,
                                shap_ctx, score_mode)
    st.markdown("")

    # ---- Counterfactual Explanations
    st.markdown("##### What would change this prediction?")
    st.caption(
        "**Single-feature counterfactuals.** For each feature in turn, "
        "we ask: what is the smallest value-change to *only this "
        "feature* (holding all others at the patient's current values) "
        "that would flip the predicted class across the 0.5 threshold? "
        "The table shows the patient's current value, the target value "
        "needed for a flip, the absolute delta, and the relative change "
        "(delta as a fraction of the feature's range in PPMI).\n\n"
        "**Tabs** are per-classifier (RF / XGBoost / LogReg) because "
        "each model has a different decision boundary; counterfactual "
        "distances differ between models.\n\n"
        "**Reading.** A small relative change (e.g., 5%) means the "
        "prediction is *fragile* with respect to that feature -- a "
        "modest measurement difference could change the prediction. "
        "A large relative change (>=30%) means the prediction is robust "
        "to that feature. Sort by smallest relative change to identify "
        "the most prediction-driving features."
    )
    _counterfactual_panel(feats, patient_idx, models, ml_methods, score_mode,
                           mtype)


def _patient_diagnostics_panel(patno, sel_row, methods_to_show, clf_cols,
                                 shap_ctx, score_mode):
    """Three scientific diagnostics per patient:

    A) Calibration anchor: 'of X PPMI patients with a similar prediction,
       Y% were actually Fast'.
    B) Threshold table: at which threshold does the patient flip
       class?
    C) Noise robustness: 20 perturbations of the feature values, P(Fast)
       range and flip probability.
    """
    import os
    from src.clinical_metrics import optimal_threshold

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")

    # ---- (A) Calibration anchor
    cal_path = os.path.join(DATA_DIR, "ml_calibration_predictions.csv")
    lr_path = os.path.join(DATA_DIR, "lr_cv_predictions.csv")
    cal_df = pd.read_csv(cal_path) if os.path.exists(cal_path) else None
    lr_df = pd.read_csv(lr_path) if os.path.exists(lr_path) else None

    anchor_rows = []
    for name in methods_to_show:
        p = float(sel_row[name]) if pd.notna(sel_row[name]) else None
        if p is None:
            continue
        # Classifier -> internal name
        clf_key = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Logistic Regression": "logistic_regression",
            "Likelihood Ratio": "likelihood_ratio",
        }.get(name)
        if clf_key in ("random_forest", "xgboost", "logistic_regression") and cal_df is not None:
            sub = cal_df[(cal_df["score_set"] == score_mode) &
                          (cal_df["model_type"] == "slopes+intercepts") &
                          (cal_df["classifier"] == clf_key)]
        elif clf_key == "likelihood_ratio" and lr_df is not None:
            sub = lr_df[(lr_df["score_set"] == score_mode) &
                         (lr_df["model_type"] == "slopes+intercepts")]
        else:
            continue
        # Patients with a similar prediction (window +/- 0.05)
        near = sub[(sub["y_prob"] >= p - 0.05) & (sub["y_prob"] <= p + 0.05)]
        n_near = len(near)
        if n_near >= 5:
            rate = float(near["y_true"].mean())
            anchor_rows.append({
                "Method": name,
                "P(Fast) of this patient": f"{p*100:.1f}%",
                "PPMI patients with similar prediction": n_near,
                "Range used": f"{(p-0.05)*100:.0f}-{(p+0.05)*100:.0f}%",
                "Actual Fast rate in that group": f"{rate*100:.1f}%",
            })
        else:
            # Widen the window to +/- 0.10
            near = sub[(sub["y_prob"] >= p - 0.10) & (sub["y_prob"] <= p + 0.10)]
            n_near = len(near)
            if n_near >= 3:
                rate = float(near["y_true"].mean())
                anchor_rows.append({
                    "Method": name,
                    "P(Fast) of this patient": f"{p*100:.1f}%",
                    "PPMI patients with similar prediction": n_near,
                    "Range used": f"{(p-0.10)*100:.0f}-{(p+0.10)*100:.0f}%",
                    "Actual Fast rate in that group": f"{rate*100:.1f}%",
                })

    st.markdown("**(a) Calibration anchor** -- empirical Fast rate among "
                  "PPMI patients with predictions near the current one.")
    if anchor_rows:
        st.dataframe(pd.DataFrame(anchor_rows),
                      width="stretch", hide_index=True)
        st.caption(
            "If the actual Fast rate in similar-prediction PPMI patients "
            "is close to the patient's P(Fast), the model is well-"
            "calibrated for this region; large differences indicate "
            "calibration drift in this probability range."
        )
    else:
        st.caption("Not enough PPMI patients with similar predictions "
                    "to anchor (need >= 3 in +/- 0.10 window).")
    st.markdown("")

    # ---- (B) Threshold table per classifier
    st.markdown("**(b) Class at different decision thresholds** -- "
                  "where would this patient flip?")
    rows_t = []
    for name in methods_to_show:
        p = float(sel_row[name]) if pd.notna(sel_row[name]) else None
        if p is None:
            continue
        clf_key = {
            "Random Forest": "random_forest",
            "XGBoost": "xgboost",
            "Logistic Regression": "logistic_regression",
        }.get(name)
        # Optimal Youden threshold per classifier (cached per call)
        youden_t = 0.5
        if clf_key and cal_df is not None:
            sub = cal_df[(cal_df["score_set"] == score_mode) &
                          (cal_df["model_type"] == "slopes+intercepts") &
                          (cal_df["classifier"] == clf_key)]
            if not sub.empty:
                opt = optimal_threshold(sub["y_true"].values,
                                          sub["y_prob"].values,
                                          criterion="youden")
                youden_t = opt["threshold"]
        thresholds = (("0.30 (sens. priority)", 0.30),
                       ("0.50 (default)", 0.50),
                       (f"{youden_t:.2f} (Youden)", youden_t),
                       ("0.70 (spec. priority)", 0.70))
        for label, t in thresholds:
            cls = "Fast" if p >= t else "Slow"
            rows_t.append({
                "Method": name,
                "Threshold": label,
                "Patient P(Fast)": f"{p*100:.1f}%",
                "Class at this threshold": cls,
            })
    if rows_t:
        st.dataframe(pd.DataFrame(rows_t),
                      width="stretch", hide_index=True)
        st.caption(
            "**Youden** is the AUC-optimal cutoff per classifier (maximises "
            "sensitivity + specificity - 1). 0.30 favours catching Fast "
            "progressors (high sensitivity), 0.70 protects against "
            "overcalling Fast (high specificity)."
        )
    st.markdown("")

    # ---- (C) Noise robustness per classifier
    st.markdown("**(c) Robustness to measurement noise** -- 30 perturbations "
                  "with 10% feature-range Gaussian noise.")
    noise_rows = _noise_sensitivity_for_patient(patno, shap_ctx, score_mode)
    if noise_rows:
        st.dataframe(pd.DataFrame(noise_rows),
                      width="stretch", hide_index=True)
        st.caption(
            "**Flip probability** = fraction of perturbations that flip "
            "the patient's predicted class at threshold 0.5. **P(Fast) "
            "range** is the central 90% range of perturbed predictions. "
            "A flip probability below 10% indicates a robust prediction."
        )
    else:
        st.caption("Noise-sensitivity analysis not available for this "
                    "patient or model context.")
    st.markdown("")

    # ---- (D) Survival prediction: expected months to H&Y >= 3
    st.markdown("**(d) Expected time to Hoehn-Yahr stage 3** -- "
                  "from the Cox proportional hazards model (c-index 0.874).")
    surv_row = _survival_prediction_for_patient(patno, shap_ctx)
    if surv_row is not None:
        st.dataframe(pd.DataFrame([surv_row]),
                      width="stretch", hide_index=True)
        st.caption(
            "Median is the time at which 50% probability of remaining "
            "below HY 3 is reached. 25-75% range gives the inter-quartile "
            "spread of the predicted survival distribution. Predictions "
            "from a Cox model with the same slope+intercept features, "
            "fitted on the full PPMI cohort (n=408, 129 events). "
            "'Not reached' = median exceeds the observation horizon."
        )
    else:
        st.caption("Survival prediction not available for this patient.")
    st.markdown("")

    # ---- (E) Baseline model comparison
    st.markdown("**(e) Comparison with simple single-feature baselines** -- "
                  "what would a one-feature model predict?")
    base_rows = _baseline_comparison_for_patient(patno, shap_ctx)
    if base_rows:
        rows_out = list(base_rows)
        # Multi-model predictions for a direct comparison
        for m in methods_to_show:
            p = float(sel_row[m]) if pd.notna(sel_row[m]) else None
            if p is not None:
                rows_out.append({
                    "Method": m + " (full model)",
                    "P(Fast)": f"{p*100:.1f}%",
                    "Class at 0.5": "Fast" if p >= 0.5 else "Slow",
                    "Discriminative AUC on PPMI": "—",
                })
        st.dataframe(pd.DataFrame(rows_out),
                      width="stretch", hide_index=True)
        st.caption(
            "Single-feature LogReg baselines were trained on the full "
            "PPMI cohort (slope + intercept of one score only). Their "
            "in-sample AUC sets a lower bound for what any sophisticated "
            "model should beat. If the patient gets very different "
            "predictions from baselines vs. full models, the difference "
            "comes from the additional 15+ features the full model uses."
        )
    else:
        st.caption("Baseline-model predictions not available for this "
                    "patient.")


def _noise_sensitivity_for_patient(patno, shap_ctx, score_mode,
                                     n_perturbations=30, noise_sd_rel=0.10,
                                     seed=42):
    """Wrapper around src.robustness.noise_sensitivity with shap_ctx lookup."""
    from src.robustness import noise_sensitivity
    if "slope" not in shap_ctx:
        return []
    feats, models = shap_ctx["slope"]
    idx_str = [str(x) for x in feats.index]
    if patno not in idx_str:
        return []
    pos = idx_str.index(patno)
    return noise_sensitivity(feats, pos, models,
                                n_perturbations=n_perturbations,
                                noise_sd_rel=noise_sd_rel, seed=seed)


def _lr_breakdown_panel(lr_res, score_mode):
    """Shows the log10(LR) contribution of the LR method per score, sorted
    by |LR|. Makes explicit which scores dominate the LR prediction --
    and where the structural limitation of the LR method
    (two-tailed p-value + variance mismatch between the Fast and Slow
    distributions) kicks in."""
    per_score = lr_res.get("per_score", {})
    total = lr_res.get("total_log10_lr", 0.0)
    p_fast = lr_res.get("p_fast", 0.5)

    st.markdown("##### Likelihood Ratio: per-score breakdown")
    rows = []
    items = sorted(
        ((s, v) for s, v in per_score.items() if v is not None and not np.isnan(v)),
        key=lambda x: -abs(x[1])
    )
    for score, lr_val in items:
        direction = "Fast" if lr_val > 0 else "Slow"
        cap_marker = " (capped)" if abs(abs(lr_val) - 1.3) < 0.005 else ""
        rows.append({
            "Score": SCORE_LABELS.get(score, score),
            "log10(LR)": f"{lr_val:+.3f}{cap_marker}",
            "Pushes towards": direction,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown(
            f"**Total log10(LR) = {total:+.3f}** -> P(Fast) = "
            f"{p_fast*100:.1f}%"
        )
        st.caption(
            "Each score contributes log10(P(slope | fast) / P(slope | slow)) "
            "where the per-score likelihoods are the two-tailed p-values "
            "under fitted PPMI subtype distributions (the reference Likelihood Ratio method). "
            "Per-score values are capped at +/- 1.3 to prevent any single "
            "near-zero likelihood from dominating. The sum is converted to "
            "a probability via P(Fast) = 1 / (1 + 10^-total). **A score "
            "with log10(LR) near +1.3 indicates a slope that is many "
            "standard deviations from the Slow mean but within typical "
            "Fast range -- this is where the LR method has structural "
            "sensitivity, because the Fast distribution is typically "
            "wider than the Slow distribution. The ML methods (RF/XGB/"
            "LogReg) handle this jointly across all features and are "
            "more robust to single-score outliers.**"
        )
    else:
        st.caption("No per-score contributions available.")


def _survival_prediction_for_patient(patno, shap_ctx):
    """Wrapper around src.survival.predict_time_to_hy3 with shap_ctx lookup."""
    from src.survival import predict_time_to_hy3
    if "slope" not in shap_ctx:
        return None
    feats, _ = shap_ctx["slope"]
    idx_str = [str(x) for x in feats.index]
    if patno not in idx_str:
        return None
    pos = idx_str.index(patno)
    result = predict_time_to_hy3(feats.iloc[[pos]])
    if result is None:
        return None
    def fmt(v):
        return "not reached" if v is None else f"{v:.0f} mo"
    return {
        "Endpoint": "First visit with H&Y >= 3 (motor milestone)",
        "Median time (50%)": fmt(result["median"]),
        "25% (faster)": fmt(result["q25"]),
        "75% (slower)": fmt(result["q75"]),
    }


def _baseline_comparison_for_patient(patno, shap_ctx):
    """Wrapper around src.baselines.predict_baselines with shap_ctx lookup."""
    from src.baselines import predict_baselines, BASELINE_DEFINITIONS
    if "slope" not in shap_ctx:
        return []
    feats, _ = shap_ctx["slope"]
    idx_str = [str(x) for x in feats.index]
    if patno not in idx_str:
        return []
    pos = idx_str.index(patno)
    row = feats.iloc[[pos]]
    # Mean of all columns for NaN replacement
    all_feats_used = set()
    for _, _, _ in BASELINE_DEFINITIONS:
        pass
    # train_means: simply the feats column means (same distribution as the training set)
    return predict_baselines(row, feats.mean())


def _counterfactual_panel(feats, patient_idx, models, ml_methods, score_mode,
                            mtype):
    """Per-classifier single-feature counterfactual table."""
    import joblib
    import os
    from src.counterfactuals import single_feature_counterfactuals
    from src.constants import SCORE_LABELS

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")
    train_path = os.path.join(
        DATA_DIR, f"training_features_{score_mode}_{mtype}.joblib"
    )
    if not os.path.exists(train_path):
        st.caption("Training-feature reference not available "
                    "(run scripts/save_training_features.py).")
        return
    tr = joblib.load(train_path)
    X_train = tr["X"]
    cf_tabs = st.tabs(ml_methods)
    for tab, clf_name in zip(cf_tabs, ml_methods):
        with tab:
            query = feats.iloc[[patient_idx]]
            try:
                cf_df = single_feature_counterfactuals(
                    models[clf_name], query, X_train, n_top=5,
                    score_labels=SCORE_LABELS,
                )
            except Exception as e:
                st.caption(f"Could not compute counterfactuals: "
                           f"{type(e).__name__}: {e}")
                continue
            if cf_df is None or cf_df.empty:
                st.caption("No single-feature counterfactual found within "
                           "the 1.-99. percentile range of the training data. "
                           "Prediction is robust to single-feature changes.")
                continue
            display = cf_df.assign(
                **{"Patient": cf_df["original"].apply(lambda x: f"{x:+.3f}"),
                   "Target": cf_df["target_value"].apply(lambda x: f"{x:+.3f}"),
                   "Δ": cf_df["delta"].apply(lambda x: f"{x:+.3f}"),
                   "Relative change": cf_df["rel_delta_pct"].apply(
                       lambda x: f"{x:.1f}% of feature range")}
            )[["feature", "Patient", "Target", "Δ", "Relative change"]].rename(
                columns={"feature": "Feature"}
            )
            st.dataframe(display, width="stretch", hide_index=True)
