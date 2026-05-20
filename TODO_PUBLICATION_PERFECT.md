# TODO: Items to make this publication-perfect

Stand 2026-05-20. Liste der noch ausstehenden Punkte, sortiert nach
Wichtigkeit fuer eine Top-Tier-Publikation (BMJ / JAMA / Nature Medicine
/ Lancet AI). Ergaenzt die bereits erledigten Items aus
`CHANGELOG_PUBLICATION_GRADE.md`.

## Kritisch (sollten unbedingt rein)

### 1. Externe Validierung auf LuxPARK

**Status:** Wartet auf Toms Luxemburger Kollegen / Datenzugang. Skript
ist im Hauptprojekt vorbereitet.

**Was:** PPMI-trainierte Modelle auf 560 LuxPARK-Patienten evaluieren,
AUC, Calibration-Drift, externe Conformal-Coverage berichten.

**Warum:** Single wichtigster Punkt. Reviewer fordern externe Validierung
bei klinischer ML-Publikation praktisch immer. Ohne dies gehen wir
maximal in Tier-2-Journals.

### 2. Bootstrap-CIs auf Headline-AUCs

**Status:** Aktuell stehen 'RF AUC 0.95' als Punktschaetzer in den
Metric-Cards des About-Tabs.

**Was:** Soll werden 'RF AUC 0.95 [0.92, 0.97]' ueberall wo AUC genannt
wird. Bootstrap auf den CV-Predictions, 1000 Resamples.

**Aufwand:** ~30 min.

### 3. Calibration Intercept und Slope (Cox 1958)

**Status:** Aktuell nur Brier und ECE.

**Was:** Logistic Regression von wahrer Outcome auf log-odds der
vorhergesagten Wahrscheinlichkeit. Perfect calibration: intercept=0,
slope=1. Praeziser als ECE. Steyerberg 2010.

**Aufwand:** ~1h.

### 4. Hosmer-Lemeshow Goodness-of-Fit-Test

**Status:** Aktuell kein formaler Calibration-Test.

**Was:** Chi-Quadrat-Test erwartete vs beobachtete Counts in 10 Deciles
der Wahrscheinlichkeit. p < 0.05 = signifikante Miscalibration.

**Aufwand:** ~30 min.

### 5. Vergleich gegen einfache Baselines

**Status:** Wir haben kein Vergleich gegen 'trivial' und 'single-feature'
Modelle.

**Was:** Constant 'Slow' (82% Accuracy auf PPMI), UPDRS3-slope-only
LogReg, MoCA-only LogReg. Reviewer fragen das immer: "Was bringt das
komplexe Modell ueber eine simple Regel?"

**Aufwand:** ~1h.

### 6. Multiple-Comparison-Correction

**Status:** Aktuell keine FWER-Korrektur trotz Dutzender p-Werte.

**Was:** Bonferroni-Holm fuer DeLong-Tabelle (6 Vergleiche), Benjamini-
Hochberg FDR fuer NRI/IDI-Matrix und Subgroup-Tests. Anwenden auf alle
p-Werte im About-Tab.

**Aufwand:** ~30 min.

### 7. Sample Size + Post-hoc Power Analysis

**Status:** Wir nutzen n=409 weil PPMI das hat, ohne explizite Power-
Berechnung.

**Was:** "Mit n=409 koennen wir AUC-Unterschiede ≥ X mit 80% Power
detektieren". Methodik aus Pepe 2003 / Obuchowski 2004. Ins Methods-
Kapitel der Publikation.

**Aufwand:** ~2h fuer Methodik + Dokumentation.

## Wichtig (erwartete Reviewer-Fragen)

### 8. Hyperparameter-Tuning oder explizite Begruendung

**Status:** Aktuell praktische Defaults ohne Tuning.

**Was:** Entweder Nested-CV mit Optuna ueber den Hyperparameter-Grid
(5-10h Compute), oder explizite methodische Begruendung warum kein
Tuning (Overfitting-Risiko bei n=409 zu hoch, Defaults sind state-of-
the-art).

### 9. Vergleich mit publizierter PD-Progressions-Literatur

**Status:** Wir berichten unsere AUCs isoliert.

**Was:** Etablierte Modelle nachimplementieren und auf PPMI evaluieren:
Latourelle 2017 (Lancet Neurol, Genetik+Klinik), Velseboer 2013 (Disease
milestones), Macleod & Counsell 2016. Direkter AUC-Vergleich.

### 10. PPMI-Daten-Version + Dependency-Pinning

**Status:** PPMI-CSV von 2024-03-13 (laut Dateiname), aber nicht
explizit dokumentiert. requirements.txt hat lose Versionen.

**Was:** PPMI-Version in MODEL_CARD + Methods-Kapitel. requirements.txt
mit `==`-Pins. Optional Dockerfile fuer komplette Reproduzierbarkeit.

### 11. Variable Importance Stabilitaet

**Status:** SHAP nur auf einem Trainings-Snapshot (wenn auch ueber alle
Calibration-Folds gemittelt).

**Was:** SHAP-Rankings ueber Bootstrap-Resamples des Trainings. Sind
die Top-Features stabil? Metrik: Spearman-Korrelation der Rankings ueber
100 Resamples.

### 12. Partial Dependence Plots (PDP) und ICE

**Status:** Wir haben SHAP, aber keine global-aggregierten Feature-Effekte.

**Was:** sklearn.inspection.partial_dependence pro Klassifikator fuer
die Top-Features. PDP fuer globale Effekte, ICE-Linien fuer Heterogenitaet
pro Patient.

## Nice-to-have (gut, nicht zwingend)

### 13. Temporale Validierung

**Was:** PPMI rekrutiert seit 2010. Split nach Einschluss-Jahr:
Training vor 2018, Test nach 2018. Pruefen ob Performance ueber Zeit
stabil ist. Wichtig wenn PPMI sich aenderte.

### 14. Stress-Test / Adversariale Robustheit

**Was:** Sensitivitaet gegen Messfehler. Addieren von zufaelligem
Rauschen (z.B. UPDRS3 ± 2 Punkte) zu jedem Input, schauen wie sich die
Vorhersagen aendern. Anteil der Patienten, die die Klasse flippen.

### 15. Time-to-Event-Perspektive (Survival)

**Was:** Cox-Regression auf 'Time to Disease Milestone' (H&Y 3,
Falling-Onset, Dementia) als alternative Framing. Methodisch eleganter
als binaere Klassifikation aus abgeleiteten fast/slow-Labels.

### 16. Decision-Threshold-Begruendung

**Was:** Klinisch sinnvoller Threshold-Methodik: Youden-Index-Maximierung,
Net-Benefit-Maximierung aus DCA, oder klinische Cost-Funktion (z.B.
False-Negative-Cost 5x False-Positive-Cost). Default empfehlen statt
willkuerlich 0.5.

### 17. Class-Conditional Fairness

**Was:** Subgruppen-Fairness nicht nur per AUC, sondern auch
class-conditional: gibt es Bias INNERHALB Fast oder INNERHALB Slow,
ueber Demografien hinweg? Equalized-Odds-Difference (Hardt 2016) als
Fairness-Metrik.

## Aus dem vorherigen Set noch offen

### O. True Bootstrap auf Trainings-Resamples

**Status:** Noch nicht gestartet (~20h Compute).

**Was:** N=200 vollstaendige Trainings-Resamples des Pipelines (nicht
nur CV-Folds), pro Resample alle Modelle trainieren und Predictions
sammeln. Wissenschaftlich saubere AUC-CI-Schaetzung.

## Compute-Status (Stand 2026-05-20 nachmittag)

- **D laeuft im Hintergrund** (`b5z8f7z9v`): Re-Run aller Simulationen
  mit kNN-Imputation, gerade bei `run_score_combinations.py`, ~5h zu
  gehen. Outputs ueberschreiben die Median-basierten Files (Backups
  in `_median.csv` gesichert).

## Empfohlene Reihenfolge

1. Bootstrap-CIs auf Headline-AUCs (2)
2. Multiple-Comparison-Correction (6)
3. PPMI-Version + Dependency-Pinning (10)
4. Calibration Intercept/Slope + Hosmer-Lemeshow (3, 4)
5. Vergleich gegen einfache Baselines (5)
6. Sample Size Power (7)
7. Variable Importance Stabilitaet (11)
8. PDP/ICE (12)
9. Decision-Threshold-Begruendung (16)
10. Class-Conditional Fairness (17)
11. Hyperparameter-Begruendung (8)
12. Literatur-Vergleich (9)
13. Stress-Test (14)
14. Temporale Validierung (13)
15. Survival-Perspektive (15)
16. True Bootstrap (O)
17. Externe Validierung auf LuxPARK (1) — sobald Daten da
