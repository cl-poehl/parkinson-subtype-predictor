"""Klinische Metriken fuer Prediction-Model Evaluation, publikationsreif:

- DeLong-Test (DeLong 1988) fuer paarweise AUC-Vergleiche
- Sensitivitaet, Spezifitaet, PPV, NPV bei Cutoffs mit Bootstrap-CIs
- Net Reclassification Improvement (Pencina 2008)
- Integrated Discrimination Improvement (Pencina 2008)
- Decision Curve Analysis Net Benefit (Vickers 2006)
- AUC mit Bootstrap-CIs (Patient-Level-Resampling)
- Calibration-Intercept und -Slope (Cox 1958, Steyerberg 2010)
- Hosmer-Lemeshow Goodness-of-Fit-Test
- Multiple-Comparison-Korrektur (Bonferroni-Holm, Benjamini-Hochberg)

Alle Funktionen arbeiten auf 1D-Arrays y_true (0/1) und y_prob (probability).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


# ---------- DeLong ----------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Schnellere DeLong-Variante (Sun & Xu 2014). Liefert AUCs + Kovarianz-Matrix."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])

    for r in range(k):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx = sx.reshape(1, 1)
        sy = sy.reshape(1, 1)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y_true, y_prob_a, y_prob_b):
    """Zwei-Klassifikator DeLong-Test. Liefert (auc_a, auc_b, p_value)."""
    y_true = np.asarray(y_true, dtype=int)
    order = (-y_true).argsort(kind="stable")
    y_true_sorted = y_true[order]
    label_1_count = int(y_true_sorted.sum())
    pred_sorted = np.vstack(
        (np.asarray(y_prob_a)[order], np.asarray(y_prob_b)[order])
    )
    aucs, delongcov = _fast_delong(pred_sorted, label_1_count)
    L = np.array([1, -1])
    se = float(np.sqrt(L @ delongcov @ L.T))
    if se == 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = float((aucs[0] - aucs[1]) / se)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(aucs[0]), float(aucs[1]), p


# ---------- Sens/Spec/PPV/NPV ----------
def _classification_metrics(y_true, y_prob, threshold):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return sens, spec, ppv, npv


def bootstrap_classification_metrics(y_true, y_prob, threshold, n_boot=1000, seed=42):
    """Sens/Spec/PPV/NPV plus 95% Bootstrap-CI durch Patient-Level-Resampling."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    point = _classification_metrics(y_true, y_prob, threshold)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = {k: [] for k in ("sens", "spec", "ppv", "npv")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s, sp, pp, np_ = _classification_metrics(y_true[idx], y_prob[idx], threshold)
        for k, v in zip(("sens", "spec", "ppv", "npv"), (s, sp, pp, np_)):
            if not np.isnan(v):
                boots[k].append(v)
    return {
        "sens": point[0], "spec": point[1], "ppv": point[2], "npv": point[3],
        "sens_ci": (float(np.quantile(boots["sens"], 0.025)) if boots["sens"] else np.nan,
                     float(np.quantile(boots["sens"], 0.975)) if boots["sens"] else np.nan),
        "spec_ci": (float(np.quantile(boots["spec"], 0.025)) if boots["spec"] else np.nan,
                     float(np.quantile(boots["spec"], 0.975)) if boots["spec"] else np.nan),
        "ppv_ci": (float(np.quantile(boots["ppv"], 0.025)) if boots["ppv"] else np.nan,
                    float(np.quantile(boots["ppv"], 0.975)) if boots["ppv"] else np.nan),
        "npv_ci": (float(np.quantile(boots["npv"], 0.025)) if boots["npv"] else np.nan,
                    float(np.quantile(boots["npv"], 0.975)) if boots["npv"] else np.nan),
    }


# ---------- NRI / IDI ----------
def nri_idi(y_true, y_prob_old, y_prob_new, threshold=0.5):
    """Pencina 2008. NRI = (Anstieg-Probability-bei-Events) - (Anstieg-bei-NonEvents).
    IDI = Differenz im mittleren Probability bei Events minus Differenz bei NonEvents."""
    y_true = np.asarray(y_true).astype(int)
    p_old = np.asarray(y_prob_old)
    p_new = np.asarray(y_prob_new)
    events = y_true == 1
    nonevents = y_true == 0

    # Kategoriale NRI bei Schwelle
    up_e = ((p_new[events] >= threshold) & (p_old[events] < threshold)).sum()
    down_e = ((p_new[events] < threshold) & (p_old[events] >= threshold)).sum()
    up_n = ((p_new[nonevents] >= threshold) & (p_old[nonevents] < threshold)).sum()
    down_n = ((p_new[nonevents] < threshold) & (p_old[nonevents] >= threshold)).sum()
    nri_event = (up_e - down_e) / max(events.sum(), 1)
    nri_nonevent = (down_n - up_n) / max(nonevents.sum(), 1)
    nri = float(nri_event + nri_nonevent)

    # IDI (kontinuierlich)
    diff_e = float(p_new[events].mean() - p_old[events].mean()) if events.sum() else 0
    diff_n = float(p_new[nonevents].mean() - p_old[nonevents].mean()) if nonevents.sum() else 0
    idi = float(diff_e - diff_n)
    return {
        "nri": nri, "nri_event": float(nri_event), "nri_nonevent": float(nri_nonevent),
        "idi": idi, "idi_event": diff_e, "idi_nonevent": -diff_n,
    }


# ---------- Decision Curve Analysis ----------
def net_benefit(y_true, y_prob, threshold):
    """Vickers 2006 Net Benefit = TP/N - FP/N * (pt/(1-pt))."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    n = len(y_true)
    if n == 0:
        return float("nan")
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    if threshold >= 1.0:
        return 0.0
    return tp / n - fp / n * (threshold / (1 - threshold))


# ---------- AUC mit Bootstrap-CI ----------
def bootstrap_auc(y_true, y_prob, n_boot=1000, seed=42):
    """ROC-AUC Punktschaetzer plus 95% Bootstrap-CI durch Patient-Level-Resampling.

    Returns dict mit Keys: auc, auc_lo, auc_hi, auc_mean, auc_se.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    auc_pt = float(roc_auc_score(y_true, y_prob))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, y_prob[idx]))
    if not boots:
        return {"auc": auc_pt, "auc_lo": np.nan, "auc_hi": np.nan,
                "auc_mean": np.nan, "auc_se": np.nan}
    boots = np.array(boots)
    return {
        "auc": auc_pt,
        "auc_mean": float(boots.mean()),
        "auc_se": float(boots.std(ddof=1)),
        "auc_lo": float(np.quantile(boots, 0.025)),
        "auc_hi": float(np.quantile(boots, 0.975)),
    }


# ---------- Calibration Intercept und Slope (Cox 1958, Steyerberg 2010) ----------
def calibration_intercept_slope(y_true, y_prob, eps=1e-8):
    """Cox-Calibration durch Logistic Regression von y_true auf log-odds(p).

    Perfect Calibration: intercept=0, slope=1.
    intercept > 0 --> Modell unterschaetzt systematisch (zu pessimistisch).
    intercept < 0 --> Modell ueberschaetzt.
    slope < 1 --> Predictions zu extrem ('overfit' am Rand).
    slope > 1 --> Predictions zu konservativ (mittig).

    Returns dict mit Keys: intercept, intercept_se, slope, slope_se,
    intercept_ci (95%), slope_ci (95%), p_intercept, p_slope.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    logit_p = np.log(y_prob / (1.0 - y_prob))

    # statsmodels GLM mit Logit-Link wenn verfuegbar (schneller, gibt SEs zurueck)
    try:
        import statsmodels.api as sm
        X = sm.add_constant(logit_p)
        model = sm.GLM(y_true, X, family=sm.families.Binomial()).fit(disp=0)
        intercept, slope = float(model.params[0]), float(model.params[1])
        se_int, se_slope = float(model.bse[0]), float(model.bse[1])
        # p-values: H0 intercept=0, slope=1
        z_int = intercept / se_int if se_int else np.nan
        z_slope = (slope - 1.0) / se_slope if se_slope else np.nan
        p_int = float(2 * (1 - stats.norm.cdf(abs(z_int)))) if np.isfinite(z_int) else np.nan
        p_slope = float(2 * (1 - stats.norm.cdf(abs(z_slope)))) if np.isfinite(z_slope) else np.nan
    except Exception:
        # Fallback: sklearn LogisticRegression ohne SEs, dann Bootstrap
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=200)
        lr.fit(logit_p.reshape(-1, 1), y_true)
        intercept, slope = float(lr.intercept_[0]), float(lr.coef_[0][0])
        se_int = se_slope = np.nan
        p_int = p_slope = np.nan

    return {
        "intercept": intercept, "intercept_se": se_int,
        "slope": slope, "slope_se": se_slope,
        "intercept_ci": (intercept - 1.96 * se_int, intercept + 1.96 * se_int)
                         if np.isfinite(se_int) else (np.nan, np.nan),
        "slope_ci": (slope - 1.96 * se_slope, slope + 1.96 * se_slope)
                     if np.isfinite(se_slope) else (np.nan, np.nan),
        "p_intercept": p_int, "p_slope": p_slope,
    }


# ---------- Hosmer-Lemeshow Goodness-of-Fit ----------
def hosmer_lemeshow(y_true, y_prob, g=10):
    """Hosmer-Lemeshow Chi-Quadrat-Test fuer Calibration.

    Sortiert nach predicted probability, bildet g (typisch 10) Gruppen mit
    gleicher Anzahl Patienten, vergleicht beobachtete vs erwartete Events
    pro Gruppe. p < 0.05 --> signifikante Miscalibration.

    Returns dict: chi2, dof (g-2), p_value, groups (DataFrame mit Details).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n < g * 2:
        return {"chi2": np.nan, "dof": g - 2, "p_value": np.nan,
                "groups": pd.DataFrame()}

    order = np.argsort(y_prob)
    yt_sorted = y_true[order]
    yp_sorted = y_prob[order]

    # gleichgrosse Gruppen mit ggf. einem Rest im letzten Bucket
    edges = np.linspace(0, n, g + 1).astype(int)
    chi2 = 0.0
    rows = []
    for i in range(g):
        sl = slice(edges[i], edges[i + 1])
        n_g = edges[i + 1] - edges[i]
        if n_g == 0:
            continue
        obs_events = float(yt_sorted[sl].sum())
        exp_events = float(yp_sorted[sl].sum())
        obs_non = n_g - obs_events
        exp_non = n_g - exp_events
        # Schutz vor Division-by-Zero
        e_min = 1e-8
        chi2 += (obs_events - exp_events) ** 2 / max(exp_events, e_min)
        chi2 += (obs_non - exp_non) ** 2 / max(exp_non, e_min)
        rows.append({
            "group": i + 1, "n": int(n_g),
            "observed_events": int(obs_events),
            "expected_events": exp_events,
            "observed_rate": obs_events / n_g,
            "mean_predicted": float(yp_sorted[sl].mean()),
        })
    dof = g - 2  # nach Hosmer-Lemeshow: g - 2
    p_value = float(1 - stats.chi2.cdf(chi2, dof))
    return {"chi2": float(chi2), "dof": dof, "p_value": p_value,
            "groups": pd.DataFrame(rows)}


# ---------- Multiple Comparison Correction ----------
def adjust_pvalues(pvalues, method="holm"):
    """FWER (Bonferroni-Holm) oder FDR (Benjamini-Hochberg) Korrektur.

    method: 'holm' (Bonferroni-Holm), 'bh' (Benjamini-Hochberg FDR),
            'bonferroni' (klassisch), oder 'by' (Benjamini-Yekutieli).
    NaNs werden uebersprungen und in der Ausgabe ebenfalls als NaN gefuehrt.
    """
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return p.copy()
    p_in = p[finite]
    m = len(p_in)
    order = np.argsort(p_in)
    p_sorted = p_in[order]
    adj_sorted = np.empty(m, dtype=float)

    if method == "bonferroni":
        adj_sorted = np.minimum(p_sorted * m, 1.0)
    elif method == "holm":
        for k in range(m):
            adj_sorted[k] = min((m - k) * p_sorted[k], 1.0)
        # monoton-nicht-fallend erzwingen
        for k in range(1, m):
            adj_sorted[k] = max(adj_sorted[k], adj_sorted[k - 1])
    elif method in ("bh", "fdr_bh"):
        for k in range(m):
            adj_sorted[k] = min(p_sorted[k] * m / (k + 1), 1.0)
        # rueckwaerts monoton-nicht-steigend erzwingen
        for k in range(m - 2, -1, -1):
            adj_sorted[k] = min(adj_sorted[k], adj_sorted[k + 1])
    elif method == "by":
        cm = float(np.sum(1.0 / (np.arange(m) + 1.0)))
        for k in range(m):
            adj_sorted[k] = min(p_sorted[k] * m * cm / (k + 1), 1.0)
        for k in range(m - 2, -1, -1):
            adj_sorted[k] = min(adj_sorted[k], adj_sorted[k + 1])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Rueckpermutation
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    out = np.full(p.shape, np.nan)
    out[finite] = adj
    return out


def equalized_odds(y_true, y_prob, group, threshold=0.5):
    """Hardt 2016 Equalized-Odds-Differenz zwischen Subgruppen.

    True-Positive-Rate (TPR) und False-Positive-Rate (FPR) werden pro Gruppe
    bei vorgegebenem Threshold berechnet. EOD = max(|TPR_A - TPR_B|,
    |FPR_A - FPR_B|) bei zwei Gruppen, bei mehr Gruppen die maximale
    paarweise Differenz.

    Returns dict: tpr_per_group, fpr_per_group, eod, n_per_group.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    group = np.asarray(group)
    y_pred = (y_prob >= threshold).astype(int)

    tpr_g = {}
    fpr_g = {}
    n_g = {}
    for g in np.unique(group):
        mask = group == g
        yt = y_true[mask]
        yp = y_pred[mask]
        pos = yt == 1
        neg = yt == 0
        tpr_g[str(g)] = float(yp[pos].mean()) if pos.sum() else np.nan
        fpr_g[str(g)] = float(yp[neg].mean()) if neg.sum() else np.nan
        n_g[str(g)] = int(mask.sum())

    # Maximale paarweise Differenz pro Klassen-Bedingung
    groups = list(tpr_g.keys())
    tpr_diff = 0.0
    fpr_diff = 0.0
    for i, ga in enumerate(groups):
        for gb in groups[i + 1:]:
            if np.isfinite(tpr_g[ga]) and np.isfinite(tpr_g[gb]):
                tpr_diff = max(tpr_diff, abs(tpr_g[ga] - tpr_g[gb]))
            if np.isfinite(fpr_g[ga]) and np.isfinite(fpr_g[gb]):
                fpr_diff = max(fpr_diff, abs(fpr_g[ga] - fpr_g[gb]))
    eod = max(tpr_diff, fpr_diff)

    return {"tpr_per_group": tpr_g, "fpr_per_group": fpr_g,
            "n_per_group": n_g, "tpr_diff_max": float(tpr_diff),
            "fpr_diff_max": float(fpr_diff), "eod": float(eod)}


def optimal_threshold(y_true, y_prob, criterion="youden", **kwargs):
    """Sucht den optimalen Decision-Threshold nach einem gewaehlten Kriterium.

    criterion:
        'youden': Maximiert Youden's J = Sensitivitaet + Spezifitaet - 1.
        'cost': Minimiert erwartete Kosten. Kosten via Keyword-Args
            fn_cost (default 5) und fp_cost (default 1).
        'net_benefit': Maximiert Vickers' Net Benefit bei pt=threshold.
        'f1': Maximiert F1-Score (harmonisches Mittel Sens/Praezision).

    Returns dict: threshold, sens, spec, ppv, npv, criterion_value, plus
    klassifikations-spezifische Felder.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    cands = np.unique(np.concatenate([y_prob, np.linspace(0.001, 0.999, 199)]))

    best = None
    best_val = -np.inf
    for t in cands:
        sens, spec, ppv, npv = _classification_metrics(y_true, y_prob, t)
        if np.isnan(sens) or np.isnan(spec):
            continue
        if criterion == "youden":
            val = sens + spec - 1
        elif criterion == "cost":
            fn_cost = float(kwargs.get("fn_cost", 5.0))
            fp_cost = float(kwargs.get("fp_cost", 1.0))
            fn_rate = 1 - sens
            fp_rate = 1 - spec
            prev = float(y_true.mean())
            val = -(fn_cost * fn_rate * prev + fp_cost * fp_rate * (1 - prev))
        elif criterion == "net_benefit":
            val = net_benefit(y_true, y_prob, t)
        elif criterion == "f1":
            if np.isnan(ppv) or (ppv + sens) == 0:
                continue
            val = 2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        if val > best_val:
            best_val = val
            best = (t, sens, spec, ppv, npv)
    if best is None:
        return {"threshold": np.nan, "sens": np.nan, "spec": np.nan,
                "ppv": np.nan, "npv": np.nan, "criterion_value": np.nan}
    t, sens, spec, ppv, npv = best
    return {"threshold": float(t), "sens": float(sens), "spec": float(spec),
            "ppv": float(ppv), "npv": float(npv),
            "criterion_value": float(best_val)}


def decision_curve(y_true, y_prob, thresholds=None):
    """DCA-Kurve: Net Benefit fuer jeden Threshold. Plus 'treat all' und 'treat none'."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    y_true = np.asarray(y_true).astype(int)
    prevalence = float(y_true.mean())

    rows = []
    for t in thresholds:
        nb_model = net_benefit(y_true, y_prob, t)
        # Treat all: assume everyone positive
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t)) if t < 1.0 else 0
        # Treat none: zero
        rows.append({
            "threshold": float(t),
            "Model": float(nb_model),
            "Treat all": float(nb_all),
            "Treat none": 0.0,
        })
    return pd.DataFrame(rows)
