"""Erzeugt Table 1: Baseline-Cohort-Characteristics nach Subtyp.

Output: docs/TABLE1_COHORT.md mit Markdown-Tabelle, direkt fuer
das Manuskript copy-paste-bar. Plus data/table1_cohort.csv mit den
Rohdaten.

Statistik:
- Kontinuierlich: Mann-Whitney U (non-parametrisch, robust gegen Skew)
- Kategorisch: Chi-Quadrat-Test
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
from scipy import stats


def fmt_median_iqr(values):
    v = pd.Series(values).dropna()
    if v.empty:
        return "—"
    return f"{v.median():.1f} ({v.quantile(0.25):.1f}-{v.quantile(0.75):.1f})"


def fmt_mean_sd(values):
    v = pd.Series(values).dropna()
    if v.empty:
        return "—"
    return f"{v.mean():.1f} +/- {v.std(ddof=1):.1f}"


def fmt_count_pct(n, total):
    return f"{int(n)} ({100 * n / max(total, 1):.0f}%)"


def mw_p(a, b):
    a = pd.Series(a).dropna().values
    b = pd.Series(b).dropna().values
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def chi2_p(table):
    try:
        return float(stats.chi2_contingency(table)[1])
    except Exception:
        return np.nan


def fmt_p(p):
    if pd.isna(p):
        return "—"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def main():
    data = load_data()
    data["Subtype"] = pd.to_numeric(data["Subtype"])

    # Baseline-Visit pro Patient = niedrigste Disease_duration
    base = (data.sort_values("Disease_duration")
                 .groupby("PATNO").first().reset_index())

    # Patienten-level Aggregate
    patient_first_age = data.groupby("PATNO")["Age_at_BL"].first()
    patient_first_onset = data.groupby("PATNO")["Age_at_onset"].first()
    patient_first_dx = data.groupby("PATNO")["Age_at_diagnosis"].first()
    patient_sex = data.groupby("PATNO")["SEX"].first()
    patient_subtype = data.groupby("PATNO")["Subtype"].first()
    follow_up = (data.groupby("PATNO")["Disease_duration"].max() -
                  data.groupby("PATNO")["Disease_duration"].min())
    n_visits = data.groupby("PATNO")["Timepoint"].nunique()

    # Auf 409 Patienten mit Subtyp-Label einschraenken
    mask = patient_subtype.notna()
    pat_ids = patient_subtype.index[mask]
    fast = patient_subtype[mask] == 1
    slow = patient_subtype[mask] == 2

    # Baseline-Scores aus dem early visit
    bl = base.set_index("PATNO")
    base_cols = ["UPDRS3_on", "UPDRS1", "UPDRS2", "MOCA", "HY_on",
                  "SCOPA"]
    for c in base_cols:
        if c not in bl.columns:
            bl[c] = np.nan

    pids_fast = pat_ids[fast.loc[pat_ids].values]
    pids_slow = pat_ids[slow.loc[pat_ids].values]
    n_fast = len(pids_fast)
    n_slow = len(pids_slow)
    n_total = n_fast + n_slow

    def sub(df_series, group_pids):
        return df_series.reindex(group_pids).dropna()

    rows = []
    rows.append({"Characteristic": "**n**",
                  f"Fast (n={n_fast})": str(n_fast),
                  f"Slow (n={n_slow})": str(n_slow),
                  f"Total (n={n_total})": str(n_total),
                  "p": "—"})

    # Sex
    n_male_fast = int((patient_sex.reindex(pids_fast) == 1).sum())
    n_male_slow = int((patient_sex.reindex(pids_slow) == 1).sum())
    p_sex = chi2_p(np.array([[n_male_fast, n_fast - n_male_fast],
                                [n_male_slow, n_slow - n_male_slow]]))
    rows.append({"Characteristic": "Male, n (%)",
                  f"Fast (n={n_fast})": fmt_count_pct(n_male_fast, n_fast),
                  f"Slow (n={n_slow})": fmt_count_pct(n_male_slow, n_slow),
                  f"Total (n={n_total})": fmt_count_pct(
                      n_male_fast + n_male_slow, n_total),
                  "p": fmt_p(p_sex)})

    # Age at baseline
    rows.append({
        "Characteristic": "Age at baseline (years), median (IQR)",
        f"Fast (n={n_fast})": fmt_median_iqr(sub(patient_first_age, pids_fast)),
        f"Slow (n={n_slow})": fmt_median_iqr(sub(patient_first_age, pids_slow)),
        f"Total (n={n_total})": fmt_median_iqr(
            patient_first_age.reindex(pat_ids).dropna()),
        "p": fmt_p(mw_p(sub(patient_first_age, pids_fast),
                          sub(patient_first_age, pids_slow))),
    })

    # Age at PD onset
    rows.append({
        "Characteristic": "Age at PD onset (years), median (IQR)",
        f"Fast (n={n_fast})": fmt_median_iqr(sub(patient_first_onset, pids_fast)),
        f"Slow (n={n_slow})": fmt_median_iqr(sub(patient_first_onset, pids_slow)),
        f"Total (n={n_total})": fmt_median_iqr(
            patient_first_onset.reindex(pat_ids).dropna()),
        "p": fmt_p(mw_p(sub(patient_first_onset, pids_fast),
                          sub(patient_first_onset, pids_slow))),
    })

    # Age at PD diagnosis
    rows.append({
        "Characteristic": "Age at PD diagnosis (years), median (IQR)",
        f"Fast (n={n_fast})": fmt_median_iqr(sub(patient_first_dx, pids_fast)),
        f"Slow (n={n_slow})": fmt_median_iqr(sub(patient_first_dx, pids_slow)),
        f"Total (n={n_total})": fmt_median_iqr(
            patient_first_dx.reindex(pat_ids).dropna()),
        "p": fmt_p(mw_p(sub(patient_first_dx, pids_fast),
                          sub(patient_first_dx, pids_slow))),
    })

    # Disease duration at baseline visit (months)
    bl_duration = base.set_index("PATNO")["Disease_duration"]
    rows.append({
        "Characteristic": "Disease duration at first visit (months), median (IQR)",
        f"Fast (n={n_fast})": fmt_median_iqr(sub(bl_duration, pids_fast)),
        f"Slow (n={n_slow})": fmt_median_iqr(sub(bl_duration, pids_slow)),
        f"Total (n={n_total})": fmt_median_iqr(
            bl_duration.reindex(pat_ids).dropna()),
        "p": fmt_p(mw_p(sub(bl_duration, pids_fast),
                          sub(bl_duration, pids_slow))),
    })

    # Follow-up
    rows.append({
        "Characteristic": "Total follow-up (months), median (IQR)",
        f"Fast (n={n_fast})": fmt_median_iqr(sub(follow_up, pids_fast)),
        f"Slow (n={n_slow})": fmt_median_iqr(sub(follow_up, pids_slow)),
        f"Total (n={n_total})": fmt_median_iqr(
            follow_up.reindex(pat_ids).dropna()),
        "p": fmt_p(mw_p(sub(follow_up, pids_fast),
                          sub(follow_up, pids_slow))),
    })

    # Number of visits
    rows.append({
        "Characteristic": "Number of visits, median (IQR)",
        f"Fast (n={n_fast})": fmt_median_iqr(sub(n_visits, pids_fast)),
        f"Slow (n={n_slow})": fmt_median_iqr(sub(n_visits, pids_slow)),
        f"Total (n={n_total})": fmt_median_iqr(
            n_visits.reindex(pat_ids).dropna()),
        "p": fmt_p(mw_p(sub(n_visits, pids_fast),
                          sub(n_visits, pids_slow))),
    })

    # Baseline clinical scores
    for col, label in (
        ("UPDRS1", "Baseline MDS-UPDRS I, median (IQR)"),
        ("UPDRS2", "Baseline MDS-UPDRS II, median (IQR)"),
        ("UPDRS3_on", "Baseline MDS-UPDRS III on, median (IQR)"),
        ("MOCA", "Baseline MoCA, median (IQR)"),
        ("HY_on", "Baseline Hoehn-Yahr on, median (IQR)"),
        ("SCOPA", "Baseline SCOPA-AUT, median (IQR)"),
    ):
        if col in bl.columns:
            rows.append({
                "Characteristic": label,
                f"Fast (n={n_fast})": fmt_median_iqr(sub(bl[col], pids_fast)),
                f"Slow (n={n_slow})": fmt_median_iqr(sub(bl[col], pids_slow)),
                f"Total (n={n_total})": fmt_median_iqr(
                    bl[col].reindex(pat_ids).dropna()),
                "p": fmt_p(mw_p(sub(bl[col], pids_fast),
                                  sub(bl[col], pids_slow))),
            })

    out = pd.DataFrame(rows)
    csv_path = os.path.join(ROOT, "data", "table1_cohort.csv")
    out.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Markdown-Tabelle
    md_lines = ["# Table 1: Cohort Characteristics by Subtype", ""]
    md_lines.append("Baseline demographic and clinical characteristics of "
                     f"the n={n_total} PPMI patients with subtype labels "
                     "(fast vs slow progressors). Continuous variables: "
                     "Mann-Whitney U test. Categorical variables: "
                     "Chi-square test. P-values not corrected for multiple "
                     "comparisons.")
    md_lines.append("")
    md_lines.append("| Characteristic | " + " | ".join(out.columns[1:]) + " |")
    md_lines.append("|" + "---|" * (len(out.columns)))
    for _, r in out.iterrows():
        md_lines.append("| " + " | ".join(str(r[c]) for c in out.columns) + " |")
    md_lines.append("")
    md_lines.append("**Note.** P-values < 0.05 indicate baseline differences "
                     "between fast and slow progressors that may inform model "
                     "interpretation but are not adjusted for the multiple "
                     "comparisons in this table.")
    md_lines.append("")
    md_lines.append("Data source: PPMI extract `PPMI_PD_2024-03-13.csv`, "
                     "merged with progression subtype labels from a prior "
                     "latent-time clustering analysis "
                     "(`ParkinsonPredict_PPMI_progression_subtypes.csv`).")
    md_path = os.path.join(ROOT, "docs", "TABLE1_COHORT.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved {md_path}")

    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
