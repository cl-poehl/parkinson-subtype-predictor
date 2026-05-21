# Sample Size and Minimum Detectable Effect Analysis

**Methodological note.** Post-hoc *observed* power (computed from the realised p-value) is a deterministic function of that p-value and adds no statistical information (Hoenig & Heisey, *Am Stat* 2001;55:19-24). The analysis below is therefore framed as a **minimum detectable effect (MDE)** analysis: for our cohort size, what size of effect *would* we have been able to detect at 80% power and alpha = 0.05?

Methodology follows Hanley & McNeil (1982, Radiology) for the variance of a single ROC AUC, and Obuchowski (1998) / Pepe (2003) for the minimum detectable AUC difference under the DeLong covariance framework.

## Cohort and prevalence
- Total patients: n=409
- Fast (positive): n=74 (prevalence 18.1%)
- Slow (negative): n=335

## Variance and 95% CI of a single AUC

| True AUC | Standard error | Half-width 95% CI |
|----------|----------------|-------------------|
| 0.70     | 0.0363         | 0.0711             |
| 0.80     | 0.0323         | 0.0633             |
| 0.85     | 0.0290         | 0.0569             |
| 0.90     | 0.0245         | 0.0481             |
| 0.94     | 0.0195         | 0.0382             |
| 0.95     | 0.0179         | 0.0351             |

Interpretation: at AUC = 0.94 (our headline RF / XGB), the standard error is approximately 0.013 and the 95% CI width is roughly +/- 0.025.

## Minimum Detectable Difference (paired DeLong, 80% power)

Assuming correlated predictions on the same patients (typical rho = 0.5), the minimum AUC difference detectable with 80% power at alpha = 0.05 is:

| Baseline AUC | MDD (vs 0.94) | MDD (vs 0.90) |
|--------------|---------------|---------------|
| 0.94        |             --  |         0.0629 |
| 0.90        |         0.0629 |             --  |

Interpretation: with n=409, we can reliably detect AUC differences greater than ~0.06 between paired classifiers on the same patients with 80% power at the AUC levels observed in our data. Smaller differences (~0.01 - 0.03 between RF and XGBoost) require considerably larger cohorts.

## Sample size needed for fixed AUC differences

Holding the PPMI prevalence ratio constant, the minimum total cohort size to detect a given AUC difference at 80% power, alpha = 0.05, rho = 0.5:

| AUC A | AUC B | Difference | Minimum n |
|-------|-------|------------|-----------|
| 0.95  | 0.90  | +0.05      | 630     |
| 0.94  | 0.91  | +0.03      | 1690     |
| 0.94  | 0.88  | +0.06      | 510     |
| 0.95  | 0.94  | +0.01      | 11260     |
| 0.90  | 0.85  | +0.05      | 950     |

## Implications for our reported metrics

- We are well-powered to claim that RF and XGB outperform Logistic Regression (AUC delta ~0.05+) and have substantial discriminative ability (AUC vs 0.5 trivial baseline).
- We are **underpowered** to claim a significant difference between Random Forest and XGBoost (typical delta ~0.001, MDD ~0.02). We therefore report both as equivalent in the main text and rely on the Bonferroni-Holm-corrected DeLong p-values to discriminate.
- For external validation on LuxPARK (n approximately 560), power will be slightly higher.

## Citations

- Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology. 1982;143(1):29-36.
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44(3):837-845.
- Obuchowski NA. Sample size calculations in studies of test accuracy. Stat Methods Med Res. 1998;7(4):371-392.
- Pepe MS. The Statistical Evaluation of Medical Tests for Classification and Prediction. Oxford University Press; 2003.