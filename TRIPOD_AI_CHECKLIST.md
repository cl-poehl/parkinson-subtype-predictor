# TRIPOD+AI Checklist

Following the TRIPOD+AI guideline (Collins GS, Moons KGM, Dhiman P, et al.
*BMJ* 2024;385:e078378), this checklist documents the development and
reporting of the Parkinson Subtype Predictor.

## Title and Abstract

| Item | Description | Location |
|---|---|---|
| 1a | Identify the study as developing or evaluating a multivariable prediction model and specify the target population | Paper title; About tab intro |
| 1b | Provide a structured summary with study design, predictors, outcome, methods, key results | Paper abstract; README.md |

## Introduction

| Item | Description | Location |
|---|---|---|
| 2 | Explain the medical context and rationale | About tab "Methodology" |
| 3a | Specify the objectives, including whether developing or evaluating | Paper introduction |
| 3b | State the target users and intended use | MODEL_CARD.md: Intended Use |

## Methods

### Source of Data
| Item | Description | Location |
|---|---|---|
| 4a | Describe data source (PPMI cohort) | About tab "Code and data" |
| 4b | Specify the key dates of recruitment, follow-up, outcome ascertainment | MODEL_CARD.md: Training Data |

### Participants
| Item | Description | Location |
|---|---|---|
| 5a | Specify eligibility criteria (PD diagnosis, available longitudinal scores) | MODEL_CARD.md |
| 5b | Describe settings (multi-center PPMI) | MODEL_CARD.md |
| 5c | Describe sample size and how derived (n=409 patients with subtype labels) | MODEL_CARD.md |

### Outcome
| Item | Description | Location |
|---|---|---|
| 6a | Define the outcome (fast vs slow progression subtype from prior project) | About tab Methodology |
| 6b | Outcome ascertainment method and timing | About tab Methodology |

### Predictors
| Item | Description | Location |
|---|---|---|
| 7a | Define all predictors with their measurement scale | constants.py SCORE_LABELS, SCORE_RANGES |
| 7b | Specify timing of predictor measurement (multi-visit longitudinal) | About tab Methodology |
| 7c | Specify any blinding of outcome (n/a, retrospective labels) | n/a |

### Sample size
| Item | Description | Location |
|---|---|---|
| 8 | Minimum detectable effect analysis (Hanley-McNeil 1982 / Obuchowski 1998, framed per Hoenig & Heisey 2001 rather than post-hoc power). With n=409 we can detect AUC differences >= 0.06 at 80% power. | docs/POWER_ANALYSIS.md |

### Missing data
| Item | Description | Location |
|---|---|---|
| 9 | Describe how missing data were handled (kNN k=5 imputation; sensitivity analysis comparing median/mean/kNN/MICE) | About tab Methodology + Imputation Comparison section |

### Statistical analysis - Development
| Item | Description | Location |
|---|---|---|
| 10a | Specify type of model and predictor inclusion strategy (RF/XGB/LogReg/LR, all features used, no manual selection) | About tab Methodology |
| 10b | Specify model type and assumptions | About tab Methodology |
| 10c | Hyperparameter selection: pragmatic defaults for deployed models, plus nested-CV Optuna tuning (5 outer folds, 50 trials per fold) reported separately to show no material gain | scripts/train_models.py + docs/HYPERPARAMETER_TUNING.md |
| 10d | Specify approach to internal validation (10-fold patient-grouped CV) | About tab Methodology |
| 10e | Describe procedure for model uncertainty (Calibrated probabilities + Conformal prediction sets) | About tab Methodology |

### Statistical analysis - AI specifics
| Item | Description | Location |
|---|---|---|
| 11a | State whether model includes AI/ML methods | Yes - About tab Methodology |
| 11b | Describe the AI/ML pipeline (kNN imputation, scaling, isotonic calibration, conformal wrapper) | About tab Methodology + scripts/train_models.py |
| 11c | Specify software, libraries, versions (scikit-learn, XGBoost, MAPIE 1.4, SHAP, DiCE) | requirements.txt |

### Risk of bias
| Item | Description | Location |
|---|---|---|
| 12 | Class imbalance: 4.5:1 slow:fast in PPMI; addressed via class_weight="balanced", kNN imputation (avoids majority-class bias), and subgroup analyses | About tab + MODEL_CARD.md |

## Results

### Participants
| Item | Description | Location |
|---|---|---|
| 13a | Describe the flow of participants | MODEL_CARD.md |
| 13b | Tabulate participant characteristics (n=409, ratio fast:slow, demographics) | MODEL_CARD.md |

### Model development
| Item | Description | Location |
|---|---|---|
| 14 | Specify the number of participants and outcome events in the development set | MODEL_CARD.md |
| 15 | Report model performance: AUC with bootstrap CI (1000 patient-level resamples), Brier score, ECE, Cox calibration intercept/slope (Cox 1958), Hosmer-Lemeshow test, DCA, sens/spec/PPV/NPV with CIs | About tab "Headline accuracy", "Probability calibration diagnostics", "Clinical utility metrics" |
| 16 | Report DeLong test for AUC differences with Bonferroni-Holm and Benjamini-Hochberg FDR corrections to control family-wise error and FDR | About tab "DeLong test for AUC differences" |
| 17 | Report NRI and IDI for model comparisons | About tab "Reclassification metrics" |
| 17b | Compare against simple baselines (Constant Slow, UPDRS3-only LogReg, MoCA-only LogReg) | About tab "Comparison with trivial baselines" |
| 17c | Compare against published PD progression models (Latourelle 2017, Wang 2025, Dai 2025, etc.) | docs/LITERATURE_COMPARISON.md |
| 17d | Fairness assessment with Equalized-Odds-Difference (Hardt 2016) for age and sex subgroups | About tab "Class-conditional fairness" |
| 17e | Class-conditional TPR/FPR analysis | About tab "Class-conditional fairness" |
| 17f | Temporal robustness: within-PPMI-1.0 enrollment-year split | docs/TEMPORAL_VALIDATION.md |
| 17g | Robustness to measurement noise (stress test): Gaussian noise on raw scores at 5/10/20/30% range SD | docs/STRESS_TEST.md (compute pending) |
| 17h | SHAP feature importance bootstrap stability | docs/SHAP_STABILITY.md (compute pending) |
| 17i | Alternative outcome framing: Cox PH on time-to-HY-3 milestone, c-index 0.87 | docs/SURVIVAL_ANALYSIS.md |

### Model specification
| Item | Description | Location |
|---|---|---|
| 18 | Provide the full model coefficients or sufficient information to reproduce | Code on GitHub + joblib model files in models/ |

## Discussion

| Item | Description | Location |
|---|---|---|
| 19a | State limitations: PPMI cohort imbalance; subtype labels from prior project; no external validation yet (planned: LuxPARK); single-cohort training | MODEL_CARD.md "Limitations"; About tab |
| 19b | Generalizability: 17 standard scores are routine in most PD clinics; 25-score extended mode less generalizable | About tab Score Sets |
| 19c | Implications for practice: research tool, not clinically validated | About tab Disclaimer |

## Other information

| Item | Description | Location |
|---|---|---|
| 20 | Funding source | (Add when applicable in paper) |
| 21 | Pre-registration: pre-registration prepared for LuxPARK external validation phase only | (Add OSF link when registered) |
| 22 | Data and code availability | About tab "Code and data" - GitHub link |
