# Email-Entwurf an Tom

**Betreff:** Update Parkinson-Subtyp-Predictor — Publikations-Vorbereitung
und zwei Fragen

---

Lieber Tom,

seit unserem Meeting am 29.04. ist einiges passiert. Hier ein Update,
plus zwei konkrete Fragen wo ich auf dich angewiesen bin.

## Was seit dem Meeting erledigt ist

**Deine sieben Punkte aus dem 29.04. Meeting** sind alle abgearbeitet und
im `CHANGELOG_2026-04-29.md` dokumentiert (LR-Doppelplot-Bug, LR-slopes-
only Vergleichslinien, Stratified-Plot mit Bootstrap-CIs, 2D Missingness-
x-FollowUp, Greedy Forward Selection bis n=10, zufaellige Score-
Kombinationen mit gepaartem Vergleich, alternative Imputationsverfahren).

Anschliessend habe ich im Mai die Webapp `parkinson-subtype-predictor`
auf einen publikationsreifen Stand gehoben. Im wesentlichen 17 zusaetzliche
Methodik-Items, alle dokumentiert in
`parkinson-subtype-predictor/TODO_PUBLICATION_PERFECT.md` und
`CHANGELOG_PUBLICATION_GRADE.md`. Stichpunkte:

- **Imputation:** kNN(k=5) als Default. Sensitivity-Analyse ueber 8
  Imputer (median, mean, kNN, iterative-BR, missForest, kNN+indicator,
  median+indicator, XGBoost native NaN) -- alle AUC-Unterschiede
  innerhalb +/-0.013, ueberlappende Bootstrap-CIs, statistisch nicht
  signifikant. Methodische Wahl jetzt mit zwei load-bearing Argumenten
  begruendet (Klassen-Bias-Robustheit, EPV-Erhalt).
- **Kalibrierung:** CalibratedClassifierCV (isotonic, cv=5) auf 80%
  Training. Reliability Diagrams, Brier, ECE, Cox-Calibration-Intercept/
  Slope (Cox 1958), Hosmer-Lemeshow-Test pro Klassifikator.
- **Konformale Vorhersage:** MAPIE 1.4 SplitConformalClassifier mit
  LAC-Score, 90% Coverage-Garantie. Empirische Coverage 0.89-0.93 auf
  PPMI bestaetigt.
- **CV-Strategie:** StratifiedGroupKFold (Patient-gruppiert, klassen-
  stratifiziert) -- wichtig bei 74 Fast / 10 Folds.
- **Klinische Utility:** Decision Curve Analysis (Vickers 2006),
  Sens/Spec/PPV/NPV mit Bootstrap-CIs an drei Cutoffs, NRI/IDI-
  Reklassifikations-Matrix, Empfohlene Decision-Thresholds (Youden,
  Net-Benefit-Max, 5x Cost-Weighted).
- **Statistische Vergleiche:** Paarweise DeLong-Tests (Sun-Xu 2014
  Variante) mit Bonferroni-Holm-FWER und Benjamini-Hochberg-FDR.
- **Unsicherheit:** Patient-Level Bootstrap (1000 Resamples) und
  Pencina-Style Full-Pipeline Bootstrap (100 Resamples).
- **Robustheit:** Stress-Test mit Gauss-Rauschen 5-30%, SHAP-Stabilitaet
  ueber 50 Bootstrap-Resamples (Spearman 0.86), Hyperparameter-Tuning
  per Optuna Nested-CV (kein materieller Gewinn vs Defaults).
- **Alternative Outcomes:** Cox PH auf Time-to-Hoehn-Yahr-3 mit c-index
  0.874 -- als label-definition-unabhaengiger Sanity-Check.
- **Fairness:** AUC pro Subgruppe (Alter, Geschlecht) plus Class-
  Conditional Equalized-Odds-Differenz (Hardt 2016) -- keine
  Disparitaeten ueber 0.1 detektiert.

**Headline-Performance (10-fold StratifiedGroupKFold, kNN-Imputation):**

- Random Forest: AUC 0.944 [0.902, 0.974]
- XGBoost: AUC 0.949 [0.911, 0.978]
- L1-Logistic Regression: AUC 0.905 [0.858, 0.947]
- Likelihood-Ratio Referenz: AUC 0.895 [0.850, 0.935]

Nach Bonferroni-Holm-Korrektur ist keine paarweise AUC-Differenz
signifikant, was mit der Power-Analyse uebereinstimmt (MDE ~0.06 bei
n=409).

## Manuskript-Vorbereitung

Unter `parkinson-subtype-predictor/docs/` liegt jetzt ein
~10-seitiger `PAPER_DRAFT.md` mit Abstract, Introduction, Methods (15
Sub-Sektionen), Results und Discussion. Plus Begleit-Doku:

- `MODEL_CARD.md` (Mitchell 2019, FDA-2023)
- `TRIPOD_AI_CHECKLIST.md` (Collins BMJ 2024, alle 22+9 Items)
- `PROBAST_ASSESSMENT.md` (Wolff Ann Intern Med 2019, Overall Low risk)
- `TABLE1_COHORT.md` (Baseline-Demographics mit Mann-Whitney/Chi-Quadrat
  pro Subtyp -- Fast deutlich aelter, hoeheres baseline UPDRS, sehr
  hoeheres SCOPA-AUT)
- `LITERATURE_COMPARISON.md` (Positionierung gegen 7 publizierte PD-
  Progression-Modelle: Latourelle 2017, Wang 2025, Dai 2025, Faouzi 2022
  etc., alle mit DOIs)
- `POWER_ANALYSIS.md`, `EMPIRICAL_COVERAGE.md`, `TEMPORAL_VALIDATION.md`,
  `SURVIVAL_ANALYSIS.md`, `STRESS_TEST.md`, `SHAP_STABILITY.md`,
  `HYPERPARAMETER_TUNING.md`, `TRUE_BOOTSTRAP.md`
- Sechs publikations-fertige SVG-Figures unter `figures/` (Liberation
  Serif 8pt, vector format)

Webapp ist unter http://localhost:8501 lokal lauffaehig, im About-Tab
sind jetzt 18 Analyse-Sektionen mit ausfuehrlichen Captions und einem
"How to read this page"-Glossar. Live unter
github.com/cl-poehl/parkinson-subtype-predictor (privates Repo).

## Zwei Fragen an dich

**1. LuxPARK-Daten.**

Wir sind komplett bereit fuer die externe Validierung. Das Skript
`scripts/external_validation.py` liest eine LuxPARK-CSV im PPMI-
Format und liefert sofort AUC mit Bootstrap-CI, Cox-Calibration-Drift,
Hosmer-Lemeshow, empirische Conformal-Coverage, plus einen Markdown-
Report. Aufruf:

```bash
python scripts/external_validation.py \
    --csv /pfad/zu/luxpark_visits.csv \
    --subtype-col Subtype \
    --score-set luxpark \
    --out external_validation_luxpark
```

Erwartete Spalten: `patno`, `disease_duration` (Monate seit Diagnose),
plus die 17 Standard-Scores (UPDRS3_off/on, UPDRS1-4, MoCA, SCOPA, RBDScr,
VFT_phon_f, JLO, HY_off/on, AXSC_off/on, PIGD_off/on, LEDD) und eine
`Subtype`-Spalte (1=fast, 2=slow). Subtype-Labels brauchen wir nur,
falls die LuxPARK-Kollegen sie schon vergeben haben -- ansonsten waere
das nur Discrimination ohne externes Outcome.

Aktueller Stand bei den Luxemburger Kollegen? Brauchen sie noch etwas
von unserer Seite, um die Daten freizugeben (Data Use Agreement,
ethik-Vote, etc)?

**2. LTJMM-Methodik-Zitat.**

Im Methods-Kapitel § 2.2 steht aktuell `[CITATION NEEDED FROM AUTHORS]`
fuer die Definition der `Subtype`-Labels. Wir haben die zwei Dateien
`ParkinsonPredict_PPMI_progression_subtypes.csv` und
`ParkinsonPredict_PPMI_ltjmm_latent_time.csv`, aber nicht dokumentiert
welche Scores das LTJMM benutzt hat und nach welchem Kriterium die
Cluster-Grenze gezogen wurde.

Kannst du mir die zugehoerige Publikation oder zumindest die Methodik
schicken? Konkret brauche ich:

- Welche klinischen Scores gingen als Input ins LTJMM?
- Was war das Cluster-Boundary-Kriterium (BIC, AIC, fixed cutoff am
  median annual change, etc.)?
- Gibt es eine bereits publizierte Referenz die wir zitieren koennen?

Das ist wichtig fuer das Methods-Kapitel und fuer eine Reviewer-Frage
zur potenziellen Feature-Label-Abhaengigkeit (wenn die LTJMM-Inputs
mit unseren Features ueberlappen, ueberbewertet die interne AUC die
echte prognostische Leistung).

## Sonst

- Code, Modelle, Daten und Doku alle reproduzierbar, mit Random Seed 42
  und gepinnten Dependencies in `requirements.txt`
- 13 sanity tests fuer `src/clinical_metrics.py`, alle gruen
- Webapp deploybar (Streamlit Community Cloud / HuggingFace Spaces) sobald
  wir mit der externen Validierung ein offizielles "submission target"
  haben

Falls du mit dem Manuskript-Skelett anfangen moechtest oder einzelne
Teile siehst die du anders haben willst -- gerne kurz melden, dann
adressieren wir das vor der LuxPARK-Validierung.

Viele Gruesse,
Carl
