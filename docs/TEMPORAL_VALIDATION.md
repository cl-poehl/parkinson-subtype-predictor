# Temporal Validation within PPMI 1.0

**Note:** All 409 patients with subtype labels were enrolled in PPMI 1.0 (2010-2017). The subtype clustering publication used only PPMI 1.0; PPMI 2.0 patients have no subtype labels yet. Therefore we split *within* PPMI 1.0 by enrollment year to test robustness to recruitment drift (different sites activating over time, evolving inclusion criteria). True PPMI 1.0 vs PPMI 2.0 validation will require new subtype labels for the PPMI 2.0 cohort.

## Results

| Split year | Classifier | Train AUC [95% CI] | Late-test AUC [95% CI] | n_train / n_test |
|------------|------------|--------------------|----------------------|------------------|
| 2012 | random_forest | 0.998 [0.993, 1.000] | 0.975 [0.951, 0.992] | 141 / 267 |
| 2012 | xgboost | 1.000 [1.000, 1.000] | 0.941 [0.905, 0.971] | 141 / 267 |
| 2012 | logistic_regression | 0.979 [0.953, 0.996] | 0.887 [0.825, 0.941] | 141 / 267 |
| 2013 | random_forest | 0.998 [0.995, 1.000] | 0.971 [0.920, 1.000] | 331 / 77 |
| 2013 | xgboost | 1.000 [1.000, 1.000] | 0.976 [0.937, 0.999] | 331 / 77 |
| 2013 | logistic_regression | 0.969 [0.947, 0.990] | 0.952 [0.894, 0.991] | 331 / 77 |

Performance on the late-train test set approximates out-of-time generalization within the PPMI 1.0 era. A drop of more than 0.05-0.10 AUC across split years indicates temporal distribution shift in features or labels.