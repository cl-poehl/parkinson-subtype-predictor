"""Klinische Metriken fuer Prediction-Model Evaluation, publikationsreif:

- DeLong-Test (DeLong 1988) fuer paarweise AUC-Vergleiche
- Sensitivitaet, Spezifitaet, PPV, NPV bei Cutoffs mit Bootstrap-CIs
- Net Reclassification Improvement (Pencina 2008)
- Integrated Discrimination Improvement (Pencina 2008)
- Decision Curve Analysis Net Benefit (Vickers 2006)

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
