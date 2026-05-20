# Survival Analysis: Time to Hoehn & Yahr >= 3

Alternative outcome to the binary fast/slow framing: time from first visit to the first visit with Hoehn & Yahr (HY_on, fallback HY_off) >= 3.0. Patients without an event are censored at their last visit. Features are the same slope+intercept set used for the classification (median-imputed for missing).

- Total patients with H&Y data: 408
- Events: 129 (31.6%)

## Median survival per fast/slow subtype

| Subtype | n | Events | Median time-to-event (months) |
|---------|---|--------|------------------------------|
| Fast (S1) | 74 | 47 | 55.0 |
| Slow (S2) | 334 | 82 | 157.0 |

## Cox Proportional Hazards Model

- C-index: **0.874** (0.5 random, 1.0 perfect)
- Features: slope and intercept for each of 17 clinical scores

**Hazard ratios** (top 10 by p-value):

| Feature | HR | 95% CI | p-value |
|---------|-----|--------|---------|
| HY_on_slope | 20124604752.541 | [23860.319, 16973776613208726.000] | 0.0007 |
| HY_off_slope | 343006371.544 | [191.093, 615686487902420.250] | 0.0075 |
| PIGD_off_slope | 6008610763.528 | [383.055, 94251241611935728.000] | 0.0077 |
| PIGD_on_slope | 6702316655.573 | [27.839, 1613590326444557056.000] | 0.0216 |
| SCOPA_intercept | 1.024 | [1.003, 1.045] | 0.0277 |
| AXSC_off_slope | 11.822 | [1.260, 110.954] | 0.0306 |
| AXSC_on_intercept | 1.070 | [0.993, 1.154] | 0.0755 |
| PIGD_off_intercept | 2.108 | [0.874, 5.085] | 0.0968 |
| LEDD_intercept | 0.999 | [0.998, 1.000] | 0.1388 |
| UPDRS4_intercept | 1.065 | [0.976, 1.161] | 0.1559 |

HR > 1 means higher feature value increases hazard of reaching HY >= 3; HR < 1 means slower progression.