# Stress Test: Robustness to Measurement Noise

Random Forest predictions are evaluated after injecting Gaussian noise into the input scores (per-visit). Noise SD is given as fraction of each score's full range; e.g., 10% noise on MoCA (range 0-30) corresponds to SD = 3 points.

Realisations per noise level: 50.

## Flip rate and probability drift

| Noise SD (rel) | Flip rate (mean +/- SD) | Mean |dP(Fast)| (mean +/- SD) |
|----------------|--------------------------|--------------------------------|
| 0%           | 0.000 +/- nan   | 0.000 +/- nan     |
| 5%           | 0.054 +/- 0.007   | 0.078 +/- 0.002     |
| 10%           | 0.082 +/- 0.009   | 0.121 +/- 0.004     |
| 20%           | 0.131 +/- 0.012   | 0.180 +/- 0.005     |
| 30%           | 0.171 +/- 0.017   | 0.220 +/- 0.005     |

Flip rate = fraction of patients whose predicted class (Fast vs Slow at threshold 0.5) changes due to noise. Mean |dP| = average absolute change in predicted P(Fast) across patients.

## Interpretation

- At zero noise (deterministic baseline) the flip rate is zero by construction.
- A flip rate < 5% at the typical inter-rater noise level (~5-10% of range for clinical scores) indicates the model is robust to plausible measurement variability.
- Flip rates above 10% would suggest patients near the decision boundary -- the Conformal prediction set {Fast, Slow} should be invoked rather than a sharp classification.