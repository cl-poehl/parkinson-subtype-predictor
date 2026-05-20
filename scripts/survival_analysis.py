"""Survival Analysis (Item 15): Cox Proportional Hazards Modell.

Alternative Framing: Statt 'Fast vs Slow' (binaer aus prior Clustering)
direkter survival outcome 'Time to disease milestone'. Hier verwenden
wir 'Time to H&Y >= 3' (first reaching modified Hoehn & Yahr Stadium 3)
als motor-milestone.

Setup:
- Patienten mit erstem visit-time = 0 (Disease_duration relativ zu PD-Onset)
- Event = 1 wenn HY_on (oder HY_off als fallback) jemals >= 3
- Time-to-event = Monate vom ersten Visit bis zum ersten >= 3-Visit, oder
  Zensur bei letztem Visit ohne H&Y >= 3
- Covariates: Slope+Intercept-Features (gleicher Featureraum wie unsere
  Klassifikation)

Output:
- data/survival_analysis.csv: per Patient (time, event)
- docs/SURVIVAL_ANALYSIS.md: Cox-Regression-Ergebnisse, c-Index, KM-Daten
"""
import os
import sys

import numpy as np
import pandas as pd

PPMI_REPO = os.path.expanduser("~/Documents/SubtypePredictions")
sys.path.insert(0, PPMI_REPO)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_loading import load_data
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from src.constants import SCORES_LUXPARK
from src.features import extract_slope_intercept


HY_THRESHOLD = 3.0  # H&Y stage threshold for 'milestone'


def derive_time_to_event(df):
    """Aus Visit-Daten pro Patient (time, event) ableiten.

    event = 1 wenn HY_on jemals >= HY_THRESHOLD (Fallback HY_off).
    time = Monate von erstem Visit bis zum ersten visit mit HY >= threshold,
           sonst bis letztem Visit (zensiert).
    """
    df_pat = []
    for patno, grp in df.sort_values("disease_duration").groupby("patno"):
        # bestimme HY-Serie
        hy = grp["HY_on"].fillna(grp["HY_off"])
        if hy.notna().sum() == 0:
            # Keine HY-Daten -> ueberspringen
            continue
        # Erste verfuegbare HY-Messung als time=0 Referenz
        t = grp["disease_duration"].values
        t0 = t[0]
        hy_vals = hy.values
        # Erstes Event-Datum
        event_idx = np.where(hy_vals >= HY_THRESHOLD)[0]
        if event_idx.size > 0:
            ev = 1
            t_event = t[event_idx[0]] - t0
        else:
            ev = 0
            t_event = t[-1] - t0
        if t_event <= 0:
            # Patient war schon bei Baseline >= HY_THRESHOLD oder hat nur
            # einen Visit. Bei einzigem Visit mit HY>=3: t_event=0, ev=1.
            # Wir lassen das so stehen (immediate event) aber lifelines mag
            # t=0 nicht; daher leichten epsilon.
            t_event = max(t_event, 0.1)
        df_pat.append({"patno": patno, "time": t_event, "event": ev})
    return pd.DataFrame(df_pat)


def main():
    docs_dir = os.path.join(ROOT, "docs")
    out_dir = os.path.join(ROOT, "data")
    os.makedirs(docs_dir, exist_ok=True)

    data = load_data()
    df = data.rename(columns={"PATNO": "patno",
                                "Disease_duration": "disease_duration"})
    df = df.dropna(subset=["disease_duration"])

    survival = derive_time_to_event(df)
    print(f"Patients with H&Y data: {len(survival)}")
    print(f"Events (HY >= {HY_THRESHOLD}): {int(survival['event'].sum())} "
           f"({100 * survival['event'].mean():.1f}%)")
    print(f"Median follow-up (censored): "
           f"{survival[survival['event']==0]['time'].median():.1f} mo")
    print(f"Median time-to-event: "
           f"{survival[survival['event']==1]['time'].median():.1f} mo")

    # Features verheiraten
    feats = extract_slope_intercept(df, SCORES_LUXPARK)
    common = feats.index.intersection(set(survival["patno"]))
    feats = feats.loc[list(common)]
    survival = survival[survival["patno"].isin(common)].copy()
    survival = survival.set_index("patno").loc[list(common)].reset_index()

    # Imputation auf Feature-Set (Median-Imputation einfach hier)
    feats_imp = feats.fillna(feats.median())

    df_cox = pd.concat([survival.set_index("patno"),
                          feats_imp.loc[survival["patno"].values]], axis=1)
    df_cox.dropna(inplace=True)

    # Cox-Regression
    cph = CoxPHFitter(penalizer=0.05)
    try:
        cph.fit(df_cox.reset_index(drop=True),
                 duration_col="time", event_col="event",
                 fit_options={"step_size": 0.5})
    except Exception as e:
        print(f"Cox fit failed: {e}; trying without fit_options")
        cph.fit(df_cox.reset_index(drop=True),
                 duration_col="time", event_col="event")

    c_index = cph.concordance_index_
    print(f"\nCox c-index: {c_index:.3f}")
    print("Hazard ratios (top 10 by p-value):")
    summary = cph.summary.sort_values("p")
    print(summary[["exp(coef)", "exp(coef) lower 95%",
                    "exp(coef) upper 95%", "p"]].head(10))

    # Speichere Survival-Tabelle und Coefs
    surv_path = os.path.join(out_dir, "survival_analysis.csv")
    survival.to_csv(surv_path, index=False)
    coefs_path = os.path.join(out_dir, "cox_coefficients.csv")
    summary.to_csv(coefs_path)
    print(f"Saved {surv_path}, {coefs_path}")

    # KM nach Subtyp falls vorhanden
    subtype = df.groupby("patno")["Subtype"].first()
    survival["subtype"] = survival["patno"].map(subtype)
    km_results = {}
    kmf = KaplanMeierFitter()
    for st in (1, 2):
        sub = survival[survival["subtype"] == st]
        if len(sub) < 10:
            continue
        kmf.fit(sub["time"], event_observed=sub["event"],
                 label=f"Subtype {st}")
        km_results[st] = {
            "n": len(sub),
            "events": int(sub["event"].sum()),
            "median_survival": kmf.median_survival_time_,
        }

    md = ["# Survival Analysis: Time to Hoehn & Yahr >= 3", ""]
    md.append(f"Alternative outcome to the binary fast/slow framing: "
                f"time from first visit to the first visit with Hoehn & "
                f"Yahr (HY_on, fallback HY_off) >= {HY_THRESHOLD}. "
                f"Patients without an event are censored at their last "
                f"visit. Features are the same slope+intercept set used "
                "for the classification (median-imputed for missing).")
    md.append("")
    md.append(f"- Total patients with H&Y data: {len(survival)}")
    md.append(f"- Events: {int(survival['event'].sum())} "
                f"({100 * survival['event'].mean():.1f}%)")
    if km_results:
        md.append("")
        md.append("## Median survival per fast/slow subtype")
        md.append("")
        md.append("| Subtype | n | Events | Median time-to-event (months) |")
        md.append("|---------|---|--------|------------------------------|")
        for st, r in km_results.items():
            label = "Fast" if st == 1 else "Slow"
            ms = (f"{r['median_survival']:.1f}"
                   if not pd.isna(r["median_survival"]) else "not reached")
            md.append(f"| {label} (S{st}) | {r['n']} | {r['events']} | {ms} |")
    md.append("")
    md.append(f"## Cox Proportional Hazards Model")
    md.append("")
    md.append(f"- C-index: **{c_index:.3f}** (0.5 random, 1.0 perfect)")
    md.append(f"- Features: slope and intercept for each of "
                f"{len(SCORES_LUXPARK)} clinical scores")
    md.append("")
    md.append("**Hazard ratios** (top 10 by p-value):")
    md.append("")
    md.append("| Feature | HR | 95% CI | p-value |")
    md.append("|---------|-----|--------|---------|")
    for feat, r in summary.head(10).iterrows():
        hr = r["exp(coef)"]
        lo = r["exp(coef) lower 95%"]
        hi = r["exp(coef) upper 95%"]
        p = r["p"]
        md.append(f"| {feat} | {hr:.3f} | [{lo:.3f}, {hi:.3f}] | "
                    f"{p:.4f} |" if p >= 0.0001 else
                    f"| {feat} | {hr:.3f} | [{lo:.3f}, {hi:.3f}] | <0.0001 |")
    md.append("")
    md.append("HR > 1 means higher feature value increases hazard of "
                "reaching HY >= 3; HR < 1 means slower progression.")
    with open(os.path.join(docs_dir, "SURVIVAL_ANALYSIS.md"), "w") as f:
        f.write("\n".join(md))
    print(f"Saved {docs_dir}/SURVIVAL_ANALYSIS.md")


if __name__ == "__main__":
    main()
