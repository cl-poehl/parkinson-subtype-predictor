# Supplementary Material Structure

Mapping of `docs/` files to manuscript supplement sections. Numbering
follows the convention S1-Sn for supplement tables/figures, M1-Mn for
methods supplement.

## Main manuscript

| Section | Source |
|---|---|
| Introduction | `docs/PAPER_DRAFT.md` § 1 |
| Methods (condensed) | `docs/PAPER_DRAFT.md` § 2 |
| Results | `docs/PAPER_DRAFT.md` § 3 |
| Discussion | `docs/PAPER_DRAFT.md` § 4 |
| Headline AUCs with CIs (Table 1) | Computed from `data/ml_calibration_predictions.csv` |
| Decision curve analysis (Figure 1) | Live in About-tab, also in `figures/dca.svg` |
| Reliability diagram (Figure 2) | `figures/reliability.svg` |
| SHAP top features (Figure 3) | `figures/shap_summary.svg` |

## Supplementary Methods (M1-M14)

| Section | Source |
|---|---|
| M1. Cohort selection | `MODEL_CARD.md` |
| M2. Feature engineering (slope + intercept) | `src/features.py` + `PAPER_DRAFT.md` § 2.3 |
| M3. Missing-value handling (kNN) | `PAPER_DRAFT.md` § 2.4 |
| M4. Classifier specifications | `scripts/train_models.py` + `PAPER_DRAFT.md` § 2.5 |
| M5. Calibration (isotonic) | `PAPER_DRAFT.md` § 2.6 |
| M6. Conformal prediction (MAPIE LAC) | `src/conformal.py` + `PAPER_DRAFT.md` § 2.7 |
| M7. Cross-validation strategy | `PAPER_DRAFT.md` § 2.8 |
| M8. Statistical comparison (DeLong, Holm, BH) | `src/clinical_metrics.py` + `PAPER_DRAFT.md` § 2.9 |
| M9. Decision curve analysis | `PAPER_DRAFT.md` § 2.10 |
| M10. Sample size and power | `docs/POWER_ANALYSIS.md` |
| M11. Hyperparameter tuning | `docs/HYPERPARAMETER_TUNING.md` |
| M12. Time-to-event analysis (Cox PH) | `docs/SURVIVAL_ANALYSIS.md` |
| M13. Temporal robustness | `docs/TEMPORAL_VALIDATION.md` |
| M14. Stress test / SHAP stability | `docs/STRESS_TEST.md` + `docs/SHAP_STABILITY.md` |

## Supplementary Results / Tables (S1-S10)

| Section | Source |
|---|---|
| S1. Cohort characteristics by subtype | Derived from `data/raw/` |
| S2. Per-classifier full performance metrics | About-tab `_clinical_metrics_panel` |
| S3. Hosmer-Lemeshow + Cox calibration table | About-tab `_calibration_panel` |
| S4. Sens/Spec/PPV/NPV at three thresholds | About-tab `_clinical_metrics_panel` |
| S5. NRI/IDI matrix | About-tab `_clinical_metrics_panel` |
| S6. Subgroup fairness (age, sex) | About-tab `_subgroup_fairness_panel` + `_class_conditional_fairness_panel` |
| S7. Recommended decision thresholds | About-tab `_decision_threshold_panel` |
| S8. SHAP top-20 features with stability | `data/shap_stability.csv` |
| S9. Cox PH coefficients (Hazard Ratios) | `data/cox_coefficients.csv` |
| S10. Stress test flip rates per noise level | `data/stress_test.csv` |

## Supplementary Figures (SF1-SF8)

| Section | Source |
|---|---|
| SF1. Score combinations boxplot (k-features sensitivity) | About-tab `_score_combinations_chart` |
| SF2. Missingness sensitivity | About-tab `_missingness_chart` |
| SF3. Follow-up duration sensitivity | About-tab `_followup_chart` |
| SF4. Per-score isolated AUC | About-tab `_per_score_chart` |
| SF5. PDP+ICE for top features | About-tab `_pdp_panel` |
| SF6. Reliability diagrams (full) | About-tab `_calibration_panel` |
| SF7. Kaplan-Meier by subtype | Derived from `data/survival_analysis.csv` |
| SF8. SHAP stability summary | `data/shap_stability.csv` |

## Auxiliary documents

| File | Purpose |
|---|---|
| `MODEL_CARD.md` | Per Mitchell 2019 + FDA-2023 |
| `TRIPOD_AI_CHECKLIST.md` | Per Collins et al. BMJ 2024 (22+9 items) |
| `docs/PROBAST_ASSESSMENT.md` | Per Wolff et al. Ann Intern Med 2019 |
| `docs/LITERATURE_COMPARISON.md` | Comparator table with DOIs |
| `CHANGELOG_PUBLICATION_GRADE.md` | Reviewer-facing change history |
| `requirements.txt` | Exact dependency pins |

## Code and data availability statement (suggested wording)

"All code, trained model artefacts, and per-patient predictions are
publicly available at https://github.com/cl-poehl/parkinson-subtype-predictor
under an MIT licence. PPMI raw data are available via the
Parkinson's Progression Markers Initiative (https://ppmi-info.org)
under a data use agreement; the specific extract used was
`PPMI_PD_2024-03-13.csv`. Software dependencies are pinned in
`requirements.txt` (Python 3.14, scikit-learn 1.8.0, XGBoost 3.2.0,
MAPIE 1.4.0, statsmodels 0.14.6, lifelines 0.30.3, SHAP 0.51.0,
Streamlit 1.57.0)."
