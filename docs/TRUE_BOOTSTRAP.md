# True Bootstrap AUC Confidence Intervals (Item O)

Bootstrap resamples: 100, GroupKFold folds: 10. Each resample is a full retraining of the pipeline on a patient-level bootstrap sample of n=409, evaluated by patient-grouped CV.

| Classifier | Mean AUC | SD | 95% CI (percentile) |
|------------|----------|-----|---------------------|
| logistic_regression | 0.949 | 0.018 | [0.917, 0.975] |
| random_forest | 0.983 | 0.008 | [0.965, 0.997] |
| xgboost | 0.984 | 0.009 | [0.966, 0.997] |

This is the publication-grade uncertainty estimate. The intervals are typically wider than CV-bootstrap intervals because the full training-pipeline noise (model fit variation, imputer behaviour at different patient distributions) is captured.