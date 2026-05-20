# SHAP Feature Importance Stability

Random Forest re-trained on 50 bootstrap resamples (patient-level) of the training set. Feature importance computed as mean |SHAP value| per feature over the full PPMI cohort.

## Summary

- Mean Spearman correlation of |SHAP| rankings vs. full-data reference: **0.871** (SD 0.040)
- Mean Top-5 feature overlap with reference: **4.0/5** (SD 0.6)

Spearman correlation > 0.7 indicates stable feature ranking. Top-5 overlap of 4-5 indicates that the most important features are robustly identified across different random samples of the training data.

## Per-feature mean |SHAP| (Bootstrap)

| Feature | Mean |SHAP| | SD | Mean rank |
|---------|-------------|----|-----------|
| UPDRS2_slope | 0.0975 | 0.0104 | 1.2 |
| UPDRS1_slope | 0.0838 | 0.0135 | 1.8 |
| PIGD_off_slope | 0.0381 | 0.0102 | 4.1 |
| PIGD_on_slope | 0.0340 | 0.0079 | 4.9 |
| UPDRS3_off_slope | 0.0319 | 0.0107 | 5.7 |
| MOCA_slope | 0.0273 | 0.0083 | 6.5 |
| AXSC_off_slope | 0.0268 | 0.0077 | 6.7 |
| LEDD_slope | 0.0220 | 0.0068 | 8.4 |
| SCOPA_slope | 0.0159 | 0.0070 | 10.6 |
| AXSC_on_slope | 0.0157 | 0.0048 | 10.3 |
| UPDRS3_on_slope | 0.0149 | 0.0053 | 10.7 |
| HY_off_slope | 0.0148 | 0.0059 | 10.9 |
| HY_on_slope | 0.0083 | 0.0043 | 16.1 |
| SCOPA_intercept | 0.0078 | 0.0043 | 16.6 |
| JLO_slope | 0.0075 | 0.0034 | 16.7 |
| PIGD_on_intercept | 0.0074 | 0.0032 | 16.1 |
| UPDRS1_intercept | 0.0064 | 0.0029 | 18.2 |
| VFT_phon_f_slope | 0.0054 | 0.0028 | 20.8 |
| VFT_phon_f_intercept | 0.0047 | 0.0019 | 22.0 |
| MOCA_intercept | 0.0044 | 0.0023 | 22.8 |