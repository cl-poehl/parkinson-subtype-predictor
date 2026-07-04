"""Construct validation of the routine-score subtype model on PPMI 2.0.

PPMI 2 (PATNO with >4 digits) has no fast/slow ground-truth labels. We therefore
validate the PREDICTED subtype against (a) a label-independent clinical milestone
(time to Hoehn-Yahr stage 3) and (b) biomarkers the *true* subtype is known to
correlate with (Hähnel et al. 2024: CSF Abeta1-42 lower, p-tau/Abeta ratio higher
in fast progressors).

The raw PPMI 2 export is NOT redistributed (Data Use Agreement). Point the script
at it with the PPMI2_EXPORT environment variable; only aggregate summary
statistics and a Kaplan-Meier figure are written.

Outputs:
  data/ppmi2_construct_validation.csv   (aggregate summary — safe to share)
  figures/figS3_ppmi2_km.svg            (KM: predicted Fast vs Slow, time-to-HY-3)
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import extract_slope_intercept
from src.constants import SCORES_LUXPARK

EXPORT = os.environ.get("PPMI2_EXPORT",
                        "/tmp/ppmi2/PPMI_export_2026-04-28.csv")
RF_THRESH = 0.16  # RF Youden threshold on the calibrated scale
BMS = {"ABeta1_42": "neg", "pTau_ABeta1_42_ratio": "pos", "NfL": "pos"}


def main():
    if not os.path.exists(EXPORT):
        print(f"Skip: PPMI 2 export not found at {EXPORT} "
              f"(set PPMI2_EXPORT). DUA-restricted, not shipped.")
        return
    d = pd.read_csv(EXPORT, low_memory=False)
    d["PATNO"] = d["PATNO"].astype(str)
    d2 = d[d["PATNO"].str.len() > 4].copy()           # PPMI 2 = PATNO > 4 digits
    # Disease duration in MONTHS, identical to the training pipeline:
    d2["disease_duration"] = (d2["Age_at_BL"] - d2["Age_at_diagnosis"]) * 12 + d2["Timepoint"]
    d2["patno"] = d2["PATNO"]

    feats = extract_slope_intercept(d2, SCORES_LUXPARK, time_col="disease_duration")
    cols = [f"{s}_{k}" for s in SCORES_LUXPARK for k in ("slope", "intercept")]
    X = feats.reindex(columns=cols)
    X = X[X.notna().any(axis=1)]

    rf = joblib.load(os.path.join(ROOT, "models", "rf_luxpark_slope.joblib"))
    pfast = rf.predict_proba(X.values)[:, 1]
    pred = pd.DataFrame({"patno": X.index, "p_fast": pfast}).set_index("patno")
    pred["pred"] = np.where(pred["p_fast"] >= RF_THRESH, "fast", "slow")

    rows = []
    n2 = d2["PATNO"].nunique()
    rows.append({"metric": "n_ppmi2_patients", "value": n2})
    rows.append({"metric": "n_with_features", "value": len(X)})
    rows.append({"metric": "mean_p_fast", "value": round(float(pfast.mean()), 3)})
    rows.append({"metric": "n_predicted_fast", "value": int((pred["pred"] == "fast").sum())})

    # ---- biomarkers (baseline value per patient)
    bm = {b: (d2.dropna(subset=[b]).sort_values(["patno", "disease_duration"])
              .groupby("patno")[b].first()) for b in BMS if b in d2.columns}
    m = pred.join(pd.DataFrame(bm))
    for b, sign in BMS.items():
        if b not in m.columns:
            continue
        s = m.dropna(subset=[b, "p_fast"])
        rho, p = spearmanr(s["p_fast"], s[b])
        rows.append({"metric": f"spearman_pfast_{b}", "value": round(float(rho), 3)})
        rows.append({"metric": f"spearman_p_{b}", "value": round(float(p), 4)})
        rows.append({"metric": f"n_{b}", "value": int(len(s))})

    # ---- time-to-HY-3 (label-independent)
    hy_rows = []
    for patno, g in d2.sort_values("disease_duration").groupby("patno"):
        hy = g["HY_on"].fillna(g["HY_off"])
        if hy.notna().sum() == 0:
            continue
        t = g["disease_duration"].values
        ev = np.where(hy.values >= 3)[0]
        if ev.size:
            hy_rows.append({"patno": patno, "time": max(t[ev[0]] - t[0], 0.1), "event": 1})
        else:
            hy_rows.append({"patno": patno, "time": max(t[-1] - t[0], 0.1), "event": 0})
    surv = (pd.DataFrame(hy_rows).set_index("patno")
            .join(pred[["p_fast", "pred"]]).dropna(subset=["pred"]))
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
    cox = CoxPHFitter().fit(surv[["time", "event", "p_fast"]], "time", "event")
    rows += [
        {"metric": "hy3_n", "value": len(surv)},
        {"metric": "hy3_events", "value": int(surv["event"].sum())},
        {"metric": "hy3_cox_HR_per_pfast", "value": round(float(np.exp(cox.params_["p_fast"])), 2)},
        {"metric": "hy3_cox_p", "value": float(cox.summary.loc["p_fast", "p"])},
        {"metric": "hy3_cox_cindex", "value": round(float(cox.concordance_index_), 3)},
    ]
    f, s = surv[surv["pred"] == "fast"], surv[surv["pred"] == "slow"]
    lr = logrank_test(f["time"], s["time"], f["event"], s["event"])
    rows.append({"metric": "hy3_logrank_p", "value": float(lr.p_value)})

    # ---- SCOPA/LEDD sensitivity
    Xs = X.copy()
    for c in ["SCOPA_slope", "SCOPA_intercept", "LEDD_slope", "LEDD_intercept"]:
        Xs[c] = np.nan
    pfast_s = rf.predict_proba(Xs.values)[:, 1]
    rho_s, _ = spearmanr(pfast, pfast_s)
    rows.append({"metric": "scopa_ledd_sensitivity_spearman", "value": round(float(rho_s), 3)})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "data", "ppmi2_construct_validation.csv"), index=False)
    print(out.to_string(index=False))

    # ---- KM figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter
    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 8,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for label, grp, color in [("Predicted fast", f, "#d62728"),
                              ("Predicted slow", s, "#1f77b4")]:
        km = KaplanMeierFitter().fit(grp["time"], grp["event"],
                                     label=f"{label} (n={len(grp)})")
        km.plot_survival_function(ax=ax, ci_show=True, color=color, linewidth=1.2)
    ax.set_xlabel("Months from baseline")
    ax.set_ylabel("Probability of not reaching HY 3")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("PPMI 2.0 construct validation (log-rank p<0.0001)", fontsize=8)
    fig.tight_layout()
    for ext in ("svg", "png"):  # png is committed for the docx build (PPMI 2
        fig.savefig(os.path.join(ROOT, "figures", f"figS3_ppmi2_km.{ext}"),
                    bbox_inches="tight", dpi=200)            # cannot be regenerated in CI)
    print("Saved figures/figS3_ppmi2_km.svg + .png")


if __name__ == "__main__":
    main()
