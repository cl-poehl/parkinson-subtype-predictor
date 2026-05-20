# Model Card: Parkinson Subtype Predictor

Following Mitchell et al. (2019) "Model Cards for Model Reporting" and FDA
guidance (2023) on transparency for AI-enabled medical devices.

## Model Details

- **Developers:** Carl Poehl (development), Tom (project lead)
- **Model versions:** Multiple; deployed models defined by `models/*.joblib`
  in this repository's main branch
- **Model type:** Calibrated ensemble for binary classification
- **Architecture:**
  - Feature extraction: OLS slope and intercept per clinical score (per
    patient longitudinal trajectory)
  - Imputation: kNN with k=5 on training data
  - Scaling: StandardScaler
  - Base classifier: one of Random Forest (500 trees), XGBoost (500 trees,
    max_depth=4, lr=0.05), Logistic Regression (L1, saga)
  - Calibration: `CalibratedClassifierCV(method="isotonic", cv=5)`
  - Uncertainty: `SplitConformalClassifier` (MAPIE 1.4, LAC,
    confidence_level=0.9) on 20% held-out PPMI
- **Comparison method:** Tom's Likelihood Ratio approach (per-subtype slope
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
- Likelihood Ratio (Tom's method) AUC ≈ 0.91

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
5. **The Likelihood Ratio comparison method uses Tom's two-tailed p-value
   approach as the per-score likelihood**, which is not a strict
   statistical likelihood. We retain it for methodological consistency
   with Tom's published analyses but note the limitation.

## Quantitative Analyses

See the About tab in the deployed web app for live, computed metrics
including all bootstrap CIs and pre-computed simulations.

## Reproducibility

- Code: github.com/cl-poehl/parkinson-subtype-predictor
- Random seeds: fixed at 42 across all training/CV/bootstrap procedures
- Model artifacts: 12 calibrated classifiers + 12 conformal wrappers in
  `models/`, total ~30 MB
- Dependencies: pinned via `requirements.txt`
