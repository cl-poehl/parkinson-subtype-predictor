# True Bootstrap AUC Confidence Intervals (Item O)

Bootstrap resamples: 100, GroupKFold folds: 10. Each resample is a full retraining of the pipeline on a patient-level bootstrap sample of n=409, evaluated by patient-grouped CV.

| Classifier | Mean AUC | SD | 95% CI (percentile) |
|------------|----------|-----|---------------------|
| logistic_regression | 0.896 | 0.035 | [0.827, 0.948] |
| random_forest | 0.945 | 0.024 | [0.882, 0.975] |
| xgboost | 0.944 | 0.023 | [0.889, 0.974] |

This is the publication-grade uncertainty estimate. The intervals are typically wider than CV-bootstrap intervals because the full training-pipeline noise (model fit variation, imputer behaviour at different patient distributions) is captured.