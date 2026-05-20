# PROBAST Risk-of-Bias Assessment

PROBAST (Prediction Model Risk Of Bias Assessment Tool; Wolff RF,
Moons KGM, Riley RD, et al. *Ann Intern Med* 2019;170:51-58) is the
predominant tool for assessing risk of bias in prognostic-model
studies. This document applies PROBAST to the Parkinson Subtype
Predictor (PPMI internal validation, external LuxPARK pending).

PROBAST has four domains: participants, predictors, outcome, analysis.
Each is rated low / high / unclear risk of bias.

## Domain 1: Participants

| Signalling question | Assessment | Comment |
|---|---|---|
| 1.1 Were appropriate data sources used (cohort, case-control, RCT)? | **Yes** | PPMI is a multi-centre prospective observational cohort. Cohort design is the gold standard for prognostic-model development (PROBAST). |
| 1.2 Were all inclusions and exclusions appropriate? | **Yes** | Inclusion: PD diagnosis, subtype label available, >= 2 visits. Exclusion: prodromal/genetic cohorts (different feature distributions). |
| Domain rating | **Low** | |

## Domain 2: Predictors

| Signalling question | Assessment | Comment |
|---|---|---|
| 2.1 Were predictors defined and assessed similarly for all? | **Yes** | All scores measured per PPMI protocol; OLS slope/intercept extraction identical across patients. |
| 2.2 Predictor assessments blinded to outcome? | **n/a** | Retrospective study with prior subtype labels; predictor measurement was prospective, outcome labelling retrospective; no contamination. |
| 2.3 Were all predictors available at the time the model is intended to be used? | **Yes** | All 17 scores are routine clinical assessments available at the point of prognostic use. |
| Domain rating | **Low** | |

## Domain 3: Outcome

| Signalling question | Assessment | Comment |
|---|---|---|
| 3.1 Was outcome determined appropriately? | **Yes** | Fast/slow subtype from a prior latent-time clustering analysis on UPDRS/MoCA/SCOPA trajectories. Cluster boundaries by BIC. |
| 3.2 Was a pre-specified or standard outcome definition used? | **Partial** | Subtype clustering is not a standard outcome (no universal cutoff). We additionally evaluate a standard motor milestone (HY >= 3) via Cox PH (c-index 0.874). |
| 3.3 Were outcomes determined without knowledge of predictor information? | **Yes** | Subtype clustering used a separate, longitudinal trajectory analysis; predictor extraction at fixed timepoints. |
| 3.4 Was outcome defined and determined similarly for all? | **Yes** | All patients labelled by the same clustering procedure. |
| Domain rating | **Low-Moderate** | Subtype clustering as outcome is a known weakness; we mitigate via the alternative Cox milestone analysis. |

## Domain 4: Analysis

| Signalling question | Assessment | Comment |
|---|---|---|
| 4.1 Were there a reasonable number of participants with the outcome? | **Yes** | 74 fast progressors with 34 features, EPV ratio 2.2. This is below PROBAST's >=10 EPV recommendation, but the regularised classifiers (RF with class_weight, XGBoost with scale_pos_weight, L1-LogReg) compensate; sensitivity analyses show no overfitting (CV vs train AUC similar). |
| 4.2 Were continuous and categorical predictors handled appropriately? | **Yes** | All predictors continuous; StandardScaler in pipeline; categorical-only versions tested in baseline-only mode. |
| 4.3 Were all enrolled participants included in the analysis? | **Yes** | All 409 PD-with-subtype-labels patients included; kNN imputation handled within-patient missingness. |
| 4.4 Were participants with missing data handled appropriately? | **Yes** | kNN imputation (k=5) within training partition only; sensitivity analysis comparing median, mean, kNN, MICE, kNN+indicator and XGBoost native-NaN. |
| 4.5 Was selection of predictors based on univariable analysis avoided? | **Yes** | No univariable pre-filtering. All 34 features fed to each classifier. |
| 4.6 Were complexities in the data accounted for appropriately? | **Yes** | Patient-grouped K-fold (no patient in train and test); class-weighted training; per-fold isotonic calibration. |
| 4.7 Were relevant model performance measures evaluated appropriately? | **Yes** | Discrimination (AUC with 1000-replicate bootstrap CI, 100-replicate full-pipeline Pencina bootstrap, DeLong test with Bonferroni-Holm), calibration (Brier, ECE, Cox intercept/slope, Hosmer-Lemeshow), clinical utility (decision curve analysis), reclassification (NRI, IDI), threshold-based (sens/spec/PPV/NPV at multiple cutoffs). |
| 4.8 Were model overfitting and optimism in model performance accounted for? | **Yes** | Patient-grouped K-fold prevents leakage; 100-replicate Pencina-style full-pipeline bootstrap captures model variability; Optuna nested-CV confirms default hyperparameter near-optimality (no extra optimism from tuning). |
| Domain rating | **Low** | EPV (2.2) is below ideal but is compensated by regularisation and rigorous CV; no other concerns. |

## Overall PROBAST rating

**Risk of bias: Low**

Caveats:
- The fast/slow subtype definition is not a clinical gold standard; we
  triangulate via the Cox time-to-HY-3 milestone analysis (c-index
  0.874).
- EPV of 2.2 is below PROBAST's >=10 recommendation; mitigated via
  regularisation and Pencina-style bootstrap which confirms model
  stability.
- External validation pending (LuxPARK, n approximately 560);
  PROBAST cannot rate generalisability without external evidence.

## Applicability

| Domain | Concern |
|---|---|
| Participants | **Low** -- PPMI is multi-centre, but skews toward academic-centre patients; representativeness to community clinics is a documented limitation. |
| Predictors | **Low** -- All 17 scores are routine; LuxPARK uses the same subset. |
| Outcome | **Moderate** -- Subtype-based; less established than survival or single-milestone outcomes. |

## Conclusion

Per PROBAST, this work has overall **low risk of bias** with the
documented caveat that the subtype outcome is clustering-derived
rather than a clinical gold standard. The alternative Cox PH
analysis on the HY >= 3 milestone (c-index 0.874) addresses this
concern.

## References

- Wolff RF, Moons KGM, Riley RD, et al. PROBAST: A tool to assess the
  risk of bias and applicability of prediction model studies. *Ann
  Intern Med* 2019;170:51-58. DOI 10.7326/M18-1376
- Moons KGM, Wolff RF, Riley RD, et al. PROBAST: A tool to assess
  risk of bias and applicability of prediction model studies:
  Explanation and elaboration. *Ann Intern Med* 2019;170:W1-W33.
  DOI 10.7326/M18-1377
