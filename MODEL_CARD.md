# Model Card: Parkinson Subtype Predictor

Following Mitchell et al. (2019) "Model Cards for Model Reporting" and FDA
guidance (2023) on transparency for AI-enabled medical devices.

## Model Details

- **Developers:** Carl Poehl
- **Model versions:** Multiple; deployed models defined by `models/*.joblib`
  in this repository's main branch
- **Model type:** Calibrated ensemble for binary classification
- **Architecture:**
  - Feature extraction: OLS slope and intercept per clinical score (per
    patient longitudinal trajectory)
  - Imputation: **kNN with k=5** on training data. Primary empirical
    finding from a nine-method sensitivity analysis (median, mean, kNN,
    MICE, missForest, kNN + indicator, median + indicator, SoftImpute,
    XGBoost native NaN): AUC differences ≤ 0.013 with overlapping
    bootstrap 95% CIs -- **the choice of imputer is statistically
    insensitive in our cohort**. Among patient-aware methods (which
    avoid the class-imbalance bias affecting global Median/Mean at
    PPMI's 4.5:1 slow:fast ratio), kNN was selected for operational
    reasons (single hyperparameter; deterministic with default seed;
    avoids the doubled feature count of indicator variants at our
    n=409, EPV 2.2). MICE or missForest would be equally defensible
    alternatives.
  - Scaling: StandardScaler
  - Base classifier: one of Random Forest (500 trees), XGBoost (500 trees,
    max_depth=4, lr=0.05), Logistic Regression (L1, saga)
  - Calibration: `CalibratedClassifierCV(method="isotonic", cv=5)`
  - Uncertainty: `SplitConformalClassifier` (MAPIE 1.4, LAC,
    confidence_level=0.9) on 20% held-out PPMI
- **Comparison method:** Likelihood Ratio approach (per-subtype slope
  distributions, log10 LR sum, sigmoid posterior conversion)
- **Training framework:** scikit-learn 1.x, XGBoost 3.x

## Intended Use

- **Primary use case:** Research and demonstration. Educate clinicians and
  researchers about how multivariable models can predict PD progression
  subtype.
- **Out of scope:** Direct clinical decision-making, prognosis
  communication to patients, treatment decisions, regulatory submission.
- **Users:** Movement disorder researchers, clinical machine learning
  practitioners.

## Training Data

- **Source:** PPMI (Parkinson's Progression Markers Initiative,
  https://www.ppmi-info.org)
- **Data extract:** `PPMI_PD_2024-03-13.csv` (data freeze
  2024-03-13). Subtype labels from
  `ParkinsonPredict_PPMI_progression_subtypes.csv` (latent-time progression
  clustering, prior project).
- **Subtypes:** Fast vs slow progressors from a prior subtype-clustering
  project (binary labels, n=409 patients with available labels)
- **Class balance:** 74 fast (18%), 335 slow (82%); class_weight="balanced"
  used in all classifiers
- **Predictors:** 17 routine clinical scores (Standard mode, matches
  LuxPARK external cohort) or 25 PPMI-specific scores (Extended mode)
- **Visit structure:** Up to 10 years of longitudinal follow-up at irregular
  intervals
- **Missingness:** Variable by score; cognitive battery (HVLT, SDM, LNS,
  VFT semantic) sparser than UPDRS

## Evaluation Data

- **Internal validation:** 10-fold cross-validation grouped by patient
  (no patient appears in both train and test) on full PPMI
- **External validation:** In preparation on the LuxPARK cohort
  (Luxembourg, ~560 patients, 17 shared scores)

## Metrics

Headline metrics (10-fold CV on PPMI, kNN imputation):

- Random Forest AUC ≈ 0.94
- XGBoost AUC ≈ 0.94
- Logistic Regression AUC ≈ 0.88
- Reference Likelihood Ratio method AUC ≈ 0.91

Additional reporting in the About tab includes:

- Calibration: Brier score, Expected Calibration Error, reliability diagrams
- Clinical utility: Decision Curve Analysis, sensitivity/specificity/PPV/NPV
  with bootstrap CIs at three decision thresholds
- Statistical comparison: DeLong test for paired AUC differences,
  NRI/IDI for reclassification improvement
- Robustness: imputation method comparison, sensitivity to follow-up
  duration, missingness levels

## Ethical Considerations

- **Class imbalance:** PPMI's 4.5:1 slow:fast ratio can systematically bias
  predictions towards slow. Mitigated via class-weighted training and
  kNN imputation (chosen over median to avoid the same bias). Documented
  in About tab.
- **Group fairness:** Subgroup analyses (age, sex) reported. No systematic
  performance gap detected in PPMI; external validation planned to confirm.
- **Misuse potential:** Predictions could in principle be used in
  treatment-allocation decisions. The tool is not validated for this and
  the disclaimer is shown prominently on every page.

## Caveats and Limitations

1. **PPMI is not representative of the global PD population.** PPMI
   patients are recruited at academic centers, skew younger, and have
   higher follow-up adherence than typical clinic populations.
2. **Subtype labels are derived from a previous clustering project**, not
   from a ground-truth biological measurement. Reclassification of labels
   under newer clustering schemes could change the apparent performance.
3. **The "Extended (25 scores)" mode is overfit to PPMI** in the sense
   that the included cognitive battery (HVLT, SDM, LNS, VFT semantic) is
   not routinely measured in most clinical practice. The "Standard (17
   scores)" mode is more transferable.
4. **No prospective evaluation.** All performance numbers are retrospective
   internal validation on the same cohort used for training.
5. **The Likelihood Ratio comparison method uses a two-tailed p-value
   approach as the per-score likelihood**, which is not a strict
   statistical likelihood. We retain it for methodological consistency
   with prior published analyses but note the limitation.

## Quantitative Analyses

See the About tab in the deployed web app for live, computed metrics
including all bootstrap CIs and pre-computed simulations.

Supplementary methodological analyses under `docs/`:

- `POWER_ANALYSIS.md` -- Hanley-McNeil 1982 / Obuchowski 1998 power
  calculations; with n=409 we detect AUC differences >= 0.06 at 80%
  power.
- `TEMPORAL_VALIDATION.md` -- enrollment-year split inside PPMI 1.0;
  Random Forest AUC stable at 0.97-0.98 across splits 2012, 2013.
- `SURVIVAL_ANALYSIS.md` -- Cox PH on time-to-HY-3 milestone with the
  same slope+intercept feature set, c-index 0.87.
- `LITERATURE_COMPARISON.md` -- positioned against 7 published PD
  progression-prediction studies (Latourelle 2017, Wang 2025, Dai 2025
  et al.) with explicit AUC and external validation comparisons.
- `HYPERPARAMETER_TUNING.md`, `SHAP_STABILITY.md`, `STRESS_TEST.md`,
  `TRUE_BOOTSTRAP.md` -- additional robustness analyses (compute
  pending).

## Reproducibility

- Code: github.com/cl-poehl/parkinson-subtype-predictor
- Random seeds: fixed at 42 across all training/CV/bootstrap procedures
- Model artifacts: 12 calibrated classifiers + 12 conformal wrappers in
  `models/`, total ~30 MB
- Python: 3.14
- Dependencies: exact versions pinned in `requirements.txt` (streamlit
  1.57.0, scikit-learn 1.8.0, XGBoost 3.2.0, MAPIE 1.4.0, statsmodels
  0.14.6, SHAP 0.51.0)
- PPMI extract: `PPMI_PD_2024-03-13.csv` (publicly available on
  ppmi-info.org under data use agreement)
