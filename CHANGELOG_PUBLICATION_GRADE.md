# Changelog: Publication-Grade Upgrade (May 17-20, 2026)

Zusammenfassung der Architektur-Aenderungen, die die Webapp von einem
funktionalen Prototyp auf wissenschaftlich publikationsreif gehoben haben.

## Pipeline-Aenderungen (Backend)

### Imputation
- **Default-Imputer auf `KNNImputer(n_neighbors=5)` umgestellt** in
  `src/inference.py`-Pipeline und in `evaluate_cv` aus dem Hauptprojekt.
  Begruendung: PPMI hat ein 4.5:1 slow:fast Klassenverhaeltnis; eine globale
  Median-Imputation schiebt fast-Patienten systematisch in die Slow-Region.
  kNN nutzt die k=5 aehnlichsten PPMI-Patienten je Feature, was den Klassen-
  Bias vermeidet.
- Erweiterte Imputer in `ml_models.IMPUTERS`: median, mean, knn, mice,
  median+indicator, knn+indicator. Letztere fuegen pro Feature ein binaeres
  Missing-Flag hinzu, damit das Modell unterscheiden kann zwischen
  gemessenen und imputierten Werten.
- `run_imputation_comparison_extended.py` testet alle Varianten plus
  Native-NaN-Handling fuer XGBoost (bypasst Imputer komplett).

### Conformal Prediction
- Neu: `SplitConformalClassifier` (MAPIE 1.4, LAC-Score, `cv="prefit"`)
  auf 20%-Holdout pro Modell. Output: 90%-Coverage-Garantie ueber
  Prediction Sets {Fast}, {Slow}, oder {Fast, Slow}.
- 12 Conformal-Wrapper Joblibs in `models/`, je `*_conformal.joblib`.
- Code: `src/conformal.py:fit_conformal()`, `predict_sets()`.

### Training-Workflow
- 80% Training (mit isotonic CalibratedClassifierCV cv=5)
- 20% Holdout fuer Conformal-Kalibrierung
- Random Seed 42 ueberall

## Publication-Standards (frontend + analysis)

### Decision Curve Analysis (Vickers 2006)
- `src/clinical_metrics.py:decision_curve()`, `net_benefit()`
- Im About-Tab: Net-Benefit-Kurven pro Klassifikator plus "Treat all" /
  "Treat none" Baselines, ueber Schwellenwerte 1-99%.

### DeLong-Test (DeLong 1988, Sun & Xu 2014)
- `src/clinical_metrics.py:delong_test()` (schnelle Variante mit
  vorberechneten Midranks)
- Paarweise AUC-Vergleiche zwischen Klassifikatoren mit p-Werten.

### Sensitivitaet / Spezifitaet / PPV / NPV
- `src/clinical_metrics.py:bootstrap_classification_metrics()`
- 1000 Bootstrap-Resamples auf Patientenebene, 95%-CIs an Cutoffs
  0.3-0.7 (slider im UI).

### NRI / IDI (Pencina 2008)
- `src/clinical_metrics.py:nri_idi()`
- Paarweise Matrix im About-Tab.

### Conformal Prediction Sets im UI
- Detail-Panel pro Klassifikator zeigt jetzt **90% Set: { Fast }** /
  **{ Slow }** / **{ Fast, Slow }** prominent, farblich kodiert.
- P(Fast), Klasse, Confidence + 95% CI bleiben als Subtext.

### Calibration-Diagnostik
- Reliability Diagrams, Brier-Score, ECE pro Klassifikator pro Score-Set.
- Daten aus `run_calibration.py` -> `ml_calibration_predictions.csv`.

### Subgroup Fairness
- Bootstrap-basierter Two-Sample-Test fuer AUC-Differenzen zwischen
  Demografie-Subgruppen.

### Counterfactual Explanations
- `src/counterfactuals.py:single_feature_counterfactuals()` per
  Binaersuche pro Feature: kleinste Aenderung pro Feature, die die
  Klasse flippt.
- `dice_counterfactuals()` mit DiCE genetic-Methode als optionale
  Multi-Feature-Variante.
- UI: Sub-Tabs pro Klassifikator unter "What would change this prediction?"

## Dokumentation

### TRIPOD+AI Checkliste (Collins et al. BMJ 2024)
- `TRIPOD_AI_CHECKLIST.md` im Repo
- Alle 22 Items mit Verweisen auf Code- oder Dokumentations-Locations.

### Model Card (Mitchell et al. 2019 + FDA-2023)
- `MODEL_CARD.md` im Repo
- Intended Use, Training Data, Evaluation Data, Metrics, Ethical
  Considerations, Caveats, Reproducibility.

## Bestaende und Datenfiles

`data/`:
- `ml_calibration_predictions.csv` — CV-Predictions fuer alle 12 Konfigurationen
- `ml_missingness_bootstrap_luxpark.csv` / `_full.csv` — Score-Set-spezifische
  Missingness-Bootstrap-CIs
- `training_features_<mode>_<type>.joblib` — kNN-imputierte Trainings-
  Features fuer DiCE und Counterfactual-Referenz
- `ml_missingness_simulation_bootstrap.csv` — 5-Kern-Scores Bootstrap (Legacy)

`models/`:
- 12 Modelle: `<rf|xgb|logreg>_<luxpark|full>_<slope|baseline>.joblib`
- 12 Conformal-Wrapper: gleiche Namen + `_conformal.joblib`
- `lr_reference_<luxpark|full>.joblib` — Toms-LR-Slope-Verteilungen plus
  OLS-Verteilungen fuer Perzentile

`SubtypePredictions/intermediate_data/_median.csv`-Backups (16 Files) sichern
die Median-basierten Pre-kNN-Ergebnisse fuer Imputer-Vergleich.

## Was noch ansteht

### Compute-Tasks
- [ ] **D fortlaufend**: kNN-Re-Run aller Simulationen
  (`run_stratified.py`, `run_score_combinations.py`,
  `run_random_score_combinations.py` etc.). Aktuell bei
  `run_score_combinations.py`, Background-ID `b5z8f7z9v`.
  Nach Abschluss: alle `_median.csv`-Backups gegenuebergestellt zu den
  neuen kNN-Versionen.
- [ ] **R extended imputation comparison**: `run_imputation_comparison_extended.py`
  laufen lassen sobald der grosse Re-Run durch ist (~1.5h).
- [ ] **O True Bootstrap (Pencina-Style)**: N=200 vollstaendige Trainings-
  Resamples. ~20h Compute. Bietet wissenschaftlich saubere Unsicherheits-
  Schaetzung der AUC und Calibration jenseits der CV-Folds. Noch nicht
  gestartet.

### Wenn LuxPARK-Daten da sind
- [ ] Externes Validierungs-Skript (steht im CHANGELOG_2026-04-29.md vor)
- [ ] Pre-Registered SAP fuer LuxPARK-Validierung bei OSF (Item Q wurde
  fuer PPMI uebersprungen)
- [ ] Updates aller Performance-Metriken auf der externen Kohorte
- [ ] Calibration-Drift-Analyse zwischen PPMI und LuxPARK

### Nice-to-have (nicht zwingend fuer Publikation)
- [ ] LIME als alternative SHAP-Validierung
- [ ] Calibration-Drift ueber Patient-Subkohorten
- [ ] Hyperparameter-Tuning (steht seit Anfang offen)
- [ ] Deployment auf Streamlit Community Cloud / HF Spaces
- [ ] Use-Container-Width Deprecation in Streamlit-API fixen

## Architektur-Gesamtansicht

```
parkinson-subtype-predictor/
├── app.py                          # Hauptseite mit Top-Tab-Navigation
├── views/
│   ├── single_patient.py           # Form-basierte Einzelvorhersage
│   ├── batch.py                    # CSV-Upload
│   ├── demo.py                     # Synthetische Demo-Patienten
│   ├── about.py                    # Methodologie + alle Analysen
│   └── _utils.py                   # Shared run_predictions + render_results
├── src/
│   ├── constants.py                # Score sets, paths
│   ├── features.py                 # Slope+Intercept extraction
│   ├── inference.py                # Model loading, predict_with_folds
│   ├── conformal.py                # SplitConformalClassifier wrapper
│   ├── shap_utils.py               # SHAP ueber alle Calibration-Folds
│   ├── lr_method.py                # Toms LR mit OLS-Slope-Distributions
│   ├── reliability.py              # Expected AUC + CI Lookup
│   ├── clinical_metrics.py         # DCA, DeLong, Sens/Spec, NRI/IDI
│   └── counterfactuals.py          # Single-feature + DiCE Counterfactuals
├── scripts/
│   ├── train_models.py             # Trainings + Conformal Wrapper
│   ├── train_lr_method.py          # LR-Referenz-Verteilungen
│   └── save_training_features.py   # Trainings-Daten fuer DiCE
├── models/                         # 12 Klassifikator + 12 Conformal joblibs
├── data/                           # Demo, Simulations- und Performance-Daten
├── MODEL_CARD.md
├── TRIPOD_AI_CHECKLIST.md
└── CHANGELOG_2026-04-29.md         # Vorgaenger-Changelog
```
