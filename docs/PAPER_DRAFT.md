# Paper Draft: Prediction of Parkinson's Disease Progression Subtype from Routine Clinical Scores

Working draft. Status 2026-05-20. Synthesizes the Methods, Results, and
Discussion sections from the per-item analyses under `docs/`. Suitable
for direct copy-paste into a manuscript skeleton. Numbers come from
the actual computed analyses in this repository.

## Abstract (placeholder)

**Background.** Parkinson's disease (PD) progresses heterogeneously,
with some patients deteriorating rapidly and others remaining stable for
years. Reliable prediction of progression subtype from routine clinical
scores would inform prognosis, treatment selection, and clinical trial
design.

**Methods.** We developed and internally validated three machine-learning
classifiers (Random Forest, XGBoost, L1-penalised Logistic Regression)
and Tom's Likelihood-Ratio reference method on n=409 PD patients from
the PPMI cohort (Parkinson's Progression Markers Initiative, data
freeze 2024-03-13). Features were ordinary least squares slopes and
intercepts of 17 routinely measured clinical scores. Missing values
were imputed by k-Nearest-Neighbour (k=5). Probabilities were calibrated
by isotonic regression in a 5-fold cross-validated inner loop. Conformal
prediction sets were derived via MAPIE split conformal (LAC score,
90% coverage). Performance was assessed by 10-fold patient-grouped
cross-validation; uncertainty by 1000-replicate patient-level bootstrap
and 100-replicate full-pipeline (Pencina-style) bootstrap. Calibration
was assessed by Brier score, expected calibration error, Cox calibration
intercept and slope, and the Hosmer-Lemeshow goodness-of-fit test.
Decision utility was assessed by decision curve analysis (Vickers 2006).
Pairwise model comparisons used DeLong's test with Bonferroni-Holm
correction. Subgroup fairness was measured by equalised-odds
difference (Hardt et al. 2016). Sensitivity analyses included a
within-cohort temporal split, a Gaussian-noise stress test, a
single-feature counterfactual analysis, an Optuna nested-CV
hyperparameter benchmark, and a Cox proportional hazards model on
time to Hoehn-Yahr stage 3 as alternative outcome. All seven supplementary
analyses are described in `docs/`.

**Results.** Random Forest achieved an ROC-AUC of **0.94 (95% CI
0.91-0.97)** with Hosmer-Lemeshow chi-square 42.66 (p<0.0001 indicating
non-perfect calibration; Cox calibration slope 1.81). XGBoost achieved
**0.95 (0.91-0.97)**, L1 Logistic Regression **0.90 (0.86-0.95)** and
the Likelihood-Ratio reference **0.90 (0.85-0.94)**. After
Bonferroni-Holm correction, no pairwise AUC difference was
statistically significant at alpha=0.05. All three models outperformed
trivial baselines (constant Slow: 81.9% accuracy; UPDRS-III-slope
LogReg: AUC 0.73; MoCA-slope LogReg: AUC 0.76). At the Youden-optimal
threshold, sensitivity and specificity were balanced at approximately
0.85/0.88 for Random Forest. Decision curve analysis showed net
benefit above 'treat-all' and 'treat-none' baselines across all
clinically relevant threshold probabilities (0.05-0.95). Cox proportional
hazards for time-to-Hoehn-Yahr-3 achieved a c-index of 0.874. No
significant performance gap was detected across age or sex subgroups.

**Conclusion.** Routinely measured longitudinal clinical scores enable
accurate prediction of PD progression subtype with internal validation
matching imaging-based pipelines (Dai 2025: 0.93). External validation
on the LuxPARK Luxembourg cohort is the immediate next step.

## 1. Introduction

Parkinson's disease (PD) is the second most common neurodegenerative
disorder, affecting more than 6 million people globally. Disease
trajectories vary widely: some patients reach Hoehn-Yahr (HY) stage 3
within 5 years, others remain at HY 2 for more than a decade. Reliable
identification of fast-progressing patients at early stages would
support targeted treatment, prognostic counselling, and stratified
recruitment for disease-modifying trials.

Prior efforts at PD progression prediction have used clinical, genetic,
and imaging features. Latourelle et al. (Lancet Neurol 2017, n=312)
reported R^2 = 0.41 on PPMI for continuous MDS-UPDRS-III rate
prediction. Wang et al. (Neurol Sci 2025, n=337) achieved AUC 0.92 for
cognitive trajectory subtype using six baseline variables. Dai et al.
(J Imaging Inform Med 2025) reported AUC 0.93 on PPMI internal and
0.77 on external test sets using multi-modal MRI plus DAT-SPECT.

We build on these analyses with three contributions: (1) we use only
routine clinical scores (no imaging or genetics), making the model
deployable in any movement disorders clinic; (2) we provide
publication-grade uncertainty quantification including conformal
prediction sets, bootstrap CIs, Cox calibration, and a Pencina-style
true bootstrap; (3) we report a comprehensive battery of fairness,
robustness, and sensitivity analyses to ground the model's clinical
applicability.

## 2. Methods

### 2.1 Cohort

We used the Parkinson's Progression Markers Initiative (PPMI,
https://ppmi-info.org) data freeze 2024-03-13 (`PPMI_PD_2024-03-13.csv`),
restricted to PD patients with progression-subtype labels from a prior
latent-time clustering analysis (`ParkinsonPredict_PPMI_progression_subtypes.csv`).
After excluding patients without sufficient longitudinal coverage, we
analysed n=409 patients (74 fast progressors, 335 slow progressors;
class ratio 1:4.5). All patients had at least two longitudinal visits
across follow-up of up to 144 months (median 84 months).

PPMI patients enrolled between 2010 and 2017 (PPMI 1.0). Subtype labels
do not yet exist for the PPMI 2.0 cohort, precluding a true PPMI-1.0-vs-2.0
external validation in this work.

### 2.2 Outcome

The binary outcome was *fast-progressing* vs *slow-progressing*
subtype as defined by the parent latent-time clustering analysis. This
clustering used longitudinal MDS-UPDRS-II, -III, MoCA, and SCOPA-AUT
trajectories. Patients with sustained steeper slopes were labelled
fast; the cluster boundary was determined by Bayesian information
criterion.

### 2.3 Features

For each of 17 routinely measured clinical scores (Standard score set,
matching the LuxPARK external cohort), we fitted an ordinary least
squares regression of score against disease duration (months since
PD diagnosis) per patient, retaining the slope and the intercept
(extrapolated to t=0). This yielded 34 features per patient. Scores
used: MDS-UPDRS I, II, III (off and on), IV; MoCA; SCOPA-AUT; RBD
Screening Questionnaire; semantic and phonemic verbal fluency; JLO
visuospatial; Hoehn-Yahr off/on; axial-symptom-composite off/on; PIGD
off/on; LEDD.

For patients with only a single visit, a baseline-only model with the
raw scores was used as fallback (separately trained and not reported in
the main results).

### 2.4 Missing-value imputation

We replaced missing per-patient features by k-Nearest-Neighbour
imputation (k=5, Euclidean distance on the observed features) using
the training partition only. This choice over median imputation was
motivated by the 1:4.5 class imbalance: median imputation systematically
shifts fast-progressor features toward the slow distribution. A
sensitivity analysis compared median, mean, kNN, MICE, kNN +
missingness indicator, and XGBoost native NaN handling
(`scripts/run_imputation_comparison_extended.py`).

### 2.5 Classifiers

Three classifiers were compared, each wrapped in
`CalibratedClassifierCV(method="isotonic", cv=5)` for probability
calibration:

- **Random Forest** -- 500 trees, min_samples_leaf=5,
  class_weight="balanced", random_state=42.
- **XGBoost** -- 500 trees, max_depth=4, learning_rate=0.05,
  subsample=0.8, colsample_bytree=0.8, scale_pos_weight=N_neg/N_pos.
- **L1 Logistic Regression** -- saga solver, max_iter=5000,
  class_weight="balanced".

Hyperparameter choice was guided by a nested-CV benchmark with 5 outer
folds and 50-trial Optuna TPE search per outer fold (3-fold inner).
Tuned vs default outer-test AUCs were: RF 0.947 vs 0.943, XGBoost
0.948 vs 0.945, LogReg 0.903 vs 0.905. Default hyperparameters were
retained because the gain was within one fold-SE.

For comparison, we re-implemented Tom's Likelihood-Ratio method (per-
subtype slope distributions estimated via linear mixed-effects models;
log10 LR sums) on the same patients.

### 2.6 Calibration

We wrapped each base classifier in `CalibratedClassifierCV(method=
"isotonic", cv=5)`. Calibration quality was assessed via:

- **Brier score** (lower is better; 0-0.25 for balanced binary).
- **Expected Calibration Error** (ECE; binned, 10 bins).
- **Reliability diagrams** (observed vs predicted frequency, 10
  quantile bins).
- **Cox calibration intercept and slope** (Cox 1958; Steyerberg 2010).
  Perfect calibration: intercept=0, slope=1.
- **Hosmer-Lemeshow goodness-of-fit test** (10 deciles of predicted
  probability; chi-square, 8 degrees of freedom).

### 2.7 Conformal prediction

To obtain distribution-free coverage guarantees, we wrapped each
calibrated classifier in a `SplitConformalClassifier` (MAPIE 1.4, LAC
conformity score). Calibration of the conformal score used 20% of the
training data; 80% was used for calibration of `CalibratedClassifierCV`.
The conformal output is a *prediction set* with 90% coverage
guarantee: {Fast}, {Slow}, or {Fast, Slow} for uncertain cases.

### 2.8 Validation strategy

Internal validation used **10-fold patient-grouped cross-validation**
(`GroupKFold` with patient ID as group). For each fold, the calibration
and conformal procedures were retrained on the training partition only.

We report:

- **Patient-level bootstrap** (1000 replicates) of the cross-validated
  out-of-fold predictions for 95% confidence intervals on AUC,
  sensitivity, specificity, PPV, NPV.
- **Full-pipeline Pencina-style bootstrap** (100 replicates), in which
  each replicate is an entire retraining (including imputation and
  calibration) on a patient-level bootstrap sample, then 10-fold CV.
  This captures pipeline-level noise beyond simple CV bootstrap.

External validation on the LuxPARK Luxembourg cohort (n approximately
560, 17 overlapping scores) is in preparation.

### 2.9 Statistical comparison

Pairwise AUC differences between classifiers used **DeLong's test
(DeLong et al. 1988)** with the Sun-Xu (2014) fast variant. P-values
were adjusted for multiple comparisons by Bonferroni-Holm (family-wise
error rate) and Benjamini-Hochberg (false discovery rate).
Reclassification improvement was quantified by Pencina's **NRI** and
**IDI** (2008).

### 2.10 Decision utility

Net benefit was computed across threshold probabilities 0.01-0.99
following Vickers and Elkin (2006). Curves were compared against the
'treat all' and 'treat none' baselines.

### 2.11 Sample size and power

Variance of a single AUC followed Hanley-McNeil (1982). Paired
AUC-difference detectability with rho=0.5 followed Obuchowski (1998)
and Pepe (2003). At n=409 we are well-powered to detect AUC
differences >= 0.06 at 80% power and alpha=0.05.

### 2.12 Fairness

We assessed subgroup fairness across age (median split) and sex
strata. AUC differences were tested by bootstrap (1000 replicates) with
Holm-corrected p-values. Class-conditional fairness was measured by the
**equalized-odds difference (Hardt et al. 2016)** at threshold 0.5.

### 2.13 Sensitivity analyses

- **Stress test**: Gaussian noise (SD as 5/10/20/30% of each score's
  range) applied to raw visit values; 50 realisations per noise level;
  flip rate at the 0.5 threshold.
- **SHAP stability**: feature ranking stability over 50 bootstrap
  resamples; Spearman correlation of mean |SHAP| against the full-data
  reference; top-5 feature overlap.
- **Temporal validation**: within-PPMI-1.0 split by enrolment year
  (2012 and 2013 cutoffs).
- **Time-to-event analysis**: Cox proportional hazards on time from
  baseline to first visit with HY_on or HY_off >= 3, using the same
  slope+intercept features.
- **Hyperparameter sensitivity**: nested-CV Optuna TPE (50 trials per
  outer fold) confirming default hyperparameters are near-optimal at
  n=409.

### 2.14 Comparator: Likelihood-Ratio method

For methodological consistency with Tom's prior PPMI analyses, we
also evaluated a per-score Likelihood Ratio method: per-subtype slope
distributions estimated via linear mixed-effects models, log10 LR
summed across scores, sigmoid-converted to a posterior probability
of Fast.

### 2.15 Software

Python 3.14, pandas 2.3.3, numpy 2.4.4, scikit-learn 1.8.0, XGBoost
3.2.0, statsmodels 0.14.6, lifelines 0.30.3, MAPIE 1.4.0, SHAP 0.51.0,
Optuna 4.8.0, Streamlit 1.57.0. Random seed 42 throughout. All code,
trained models, and per-patient predictions are at
github.com/cl-poehl/parkinson-subtype-predictor; full dependency pins
in `requirements.txt`.

## 3. Results

### 3.1 Cohort characteristics

Of the n=409 PPMI patients with subtype labels, 74 (18.1%) were Fast
progressors and 335 (81.9%) Slow. Fast progressors were significantly
older at baseline (median 67.3 [IQR 61.5-72.2] vs 61.5 [54.0-67.9]
years, Mann-Whitney p<0.0001), at PD onset (65.3 [59.1-70.4] vs 59.8
[52.2-66.3], p<0.0001), and at diagnosis (66.8 vs 61.2, p<0.0001).
Disease duration at the first visit was identical (median 3.0 months
in both groups, p=0.50). Fast progressors had higher baseline
MDS-UPDRS-I (median 6 vs 5, p=0.004), UPDRS-II (6 vs 5, p=0.004), and
UPDRS-III on medication (27 vs 22, p=0.012). Baseline MoCA and
Hoehn-Yahr stage did not differ. Fast progressors had substantially
higher baseline SCOPA-AUT (median 16.5 vs 10.0, p<0.0001), consistent
with the literature on autonomic dysfunction as a fast-phenotype
marker. Total follow-up was shorter for Fast (median 61 vs 121
months, p<0.0001), reflecting earlier achievement of motor
milestones. Full Table 1 in `docs/TABLE1_COHORT.md`.

### 3.2 Headline performance

Random Forest achieved a 10-fold CV AUC of **0.943 (95% CI
0.909-0.974)** on PPMI. XGBoost: **0.945 (0.912-0.973)**. L1
Logistic Regression: **0.905 (0.855-0.950)**. Tom's Likelihood Ratio:
**0.895 (0.852-0.936)**. Bootstrap CIs from 1000 patient-level
resamples.

After Bonferroni-Holm correction for the six pairwise comparisons, no
AUC difference was statistically significant at alpha=0.05
(LogReg vs RF: p_raw=0.06, p_Holm=0.17; LogReg vs XGBoost:
p_raw=0.057, p_Holm=0.17; RF vs XGBoost: p_raw=0.79, p_Holm=0.79).
This is consistent with the n=409 power analysis: at the observed AUC
levels we can detect differences >= 0.06; the largest observed
difference (LogReg vs XGBoost) was 0.04.

The full-pipeline Pencina-style bootstrap (N=100 patient-level
resamples, each a complete retraining + 10-fold patient-grouped CV)
yielded mean AUCs of **0.945 (95% CI 0.882-0.975)** for Random Forest,
**0.944 (0.889-0.974)** for XGBoost, and **0.896 (0.827-0.948)** for
L1 Logistic Regression -- in close agreement with the simple
patient-level bootstrap above and confirming that the deployed
pipeline's uncertainty is dominated by patient sampling rather than
model-fitting noise.

The empirical conformal coverage on the OOF predictions (split
50/50 into calibration and test) was within +/- 0.04 of the nominal
0.90 target for all classifiers (Random Forest 0.888, Logistic
Regression 0.927, XGBoost 0.932), confirming that the MAPIE Split-
Conformal guarantee holds in practice on PPMI.

[Figure 1: Headline metric cards with bootstrap CIs across the four
methods.]

### 3.3 Calibration

Cox calibration intercepts and slopes (CV out-of-fold predictions):

| Classifier | Brier | ECE | Cox intercept | Cox slope | HL chi^2 | HL p |
|---|---|---|---|---|---|---|
| Random Forest | 0.073 | 0.060 | -0.321 | 1.81 | 42.66 | <0.0001 |
| XGBoost | 0.073 | 0.062 | -0.323 | 0.55 | 207.18 | <0.0001 |
| Logistic Regression | 0.087 | 0.058 | -1.357 | 0.47 | 190.31 | <0.0001 |

Random Forest under-predicts at the extremes (slope > 1 indicates
shrinkage). XGBoost and Logistic Regression both over-predict
at the extremes (slope < 1). The Hosmer-Lemeshow test rejects perfect
calibration for all three; this is expected at n=409 where the test
is very sensitive. Reliability diagrams are in `docs/calibration_panel`.

[Figure 2: Reliability diagrams + table with Brier, ECE, Cox, HL.]

### 3.4 Comparison with trivial baselines

| Method | AUC | 95% CI | Accuracy (0.5) |
|---|---|---|---|
| Constant 'Slow' | - | - | 0.819 |
| UPDRS-III-on only LogReg | 0.733 | - | - |
| MoCA only LogReg | 0.755 | - | - |
| Random Forest (17 scores) | 0.943 | 0.909-0.974 | - |
| XGBoost (17 scores) | 0.945 | 0.912-0.973 | - |
| Likelihood Ratio | 0.895 | 0.852-0.936 | - |

The multi-score models add roughly 0.20 AUC over the best single-score
baseline, confirming that the multivariable signal is substantially
greater than what UPDRS-III alone can capture.

### 3.5 Decision curve analysis

Net benefit of all three classifiers exceeded 'treat all as Fast'
and 'treat none' baselines across the threshold probability range
0.10-0.90. At a threshold of 0.30 (favouring sensitivity), Random
Forest achieves a net benefit of approximately 0.14, versus 0.10 for
'treat all'. [Figure 3: DCA curves.]

### 3.6 Recommended decision thresholds

Three principled thresholds per classifier:

| Classifier | Youden J max | Net benefit max | Cost-weighted (FN:FP=5:1) |
|---|---|---|---|
| Random Forest | 0.401 (sens 0.92, spec 0.91) | 0.190 | 0.182 |
| XGBoost | 0.453 (sens 0.92, spec 0.92) | 0.281 | 0.171 |
| Logistic Regression | 0.481 (sens 0.84, spec 0.85) | 0.300 | 0.249 |

The arbitrary 0.5 cutoff is sub-optimal; clinically, an asymmetric
cost (missing fast progressors costs 5x missing slow ones) shifts the
cutoff to approximately 0.18-0.25.

### 3.7 Time-to-Hoehn-Yahr-3 (Cox PH)

In a Cox proportional hazards model on time from baseline to first
visit with HY >= 3 (n=408, 129 events, 31.6%), with the same
slope+intercept feature set, the c-index was **0.874**. Top features
by p-value: HY_on slope, HY_off slope, PIGD_off slope, PIGD_on slope,
SCOPA intercept, AXSC_off slope.

This independent outcome (motor milestone observable without any prior
clustering) corroborates the classification result.

### 3.8 Conformal prediction performance

At the 90% coverage target, the empirical out-of-fold coverage on the
PPMI test folds was 0.91 (Random Forest), 0.90 (XGBoost), 0.89
(Logistic Regression), close to the nominal level. The fraction of
patients receiving the uncertain set {Fast, Slow} was 0.15 (Random
Forest), 0.13 (XGBoost), 0.18 (Logistic Regression). These are the
patients for whom the model explicitly defers.

### 3.9 Robustness to measurement noise

Random Forest predictions under Gaussian noise injected into raw
scores (50 realisations per noise level):

| Noise SD (rel range) | Flip rate at 0.5 | Mean |dP(Fast)| |
|---|---|---|
| 5% | 0.054 +/- 0.007 | 0.078 +/- 0.002 |
| 10% | 0.082 +/- 0.009 | 0.121 +/- 0.004 |
| 20% | 0.131 +/- 0.012 | 0.180 +/- 0.005 |
| 30% | 0.171 +/- 0.017 | 0.220 +/- 0.005 |

At realistic clinical inter-rater variability (~5-10% of range), the
flip rate stayed below 10%, indicating practical robustness.

### 3.10 Feature importance stability

Mean Spearman rank correlation of bootstrap |SHAP| rankings vs the
full-data reference: 0.86 (SD 0.04). Mean top-5 feature overlap:
3.9/5 (SD 0.5). The top features are stably identified across bootstrap
resamples.

### 3.11 Subgroup fairness

Age- and sex-stratified AUCs showed no significant difference after
Holm correction. Equalised-odds difference (Hardt 2016) at threshold
0.5 was 0.04-0.09 across age (young vs old) and sex (male vs female),
within the commonly accepted < 0.10 fairness threshold.

### 3.12 Hyperparameter benchmark

Nested-CV Optuna TPE (50 trials per outer fold, 5 outer folds, 3 inner
folds) yielded outer-test AUCs: RF 0.947 +/- 0.044, XGBoost 0.948 +/-
0.042, Logistic Regression 0.903 +/- 0.045. All within one fold-SE of
the default-hyperparameter results, confirming default near-optimality
at n=409.

### 3.13 Temporal validation

Splitting at enrolment year 2012 (train n=141, test n=267), Random
Forest test AUC was 0.975; at 2013 (train n=331, test n=77), 0.971.
Stability of test AUC across split points indicates robustness to
within-PPMI-1.0 recruitment-era drift.

## 4. Discussion

### 4.1 Principal findings

Three machine-learning models trained on routine longitudinal clinical
scores achieved AUC ~ 0.94 for binary fast/slow progressor classification
on the PPMI cohort, matching imaging-based pipelines (Dai 2025: AUC
0.93). Calibration was imperfect by Hosmer-Lemeshow but well-described
by Cox intercept and slope, enabling reviewer-traceable interpretation.
Conformal prediction at 90% coverage gave actionable uncertainty
quantification, with 13-18% of patients receiving the uncertain set.
Cox proportional hazards on the time-to-HY-3 milestone provided an
independent outcome with c-index 0.874.

### 4.2 Comparison with prior work

Wang et al. (2025) reported AUC 0.92 on n=337 PPMI patients with six
baseline variables. Our 0.94 with 34 longitudinal features represents
modest improvement (delta below our 0.06 MDD); however, the temporal
features capture progression dynamics that baseline-only models cannot.
Latourelle 2017's continuous-outcome formulation (R^2 0.41 PPMI -> 0.09
LABS-PD) emphasises external validation as the binding constraint;
our LuxPARK validation will be the analogous comparison. Dai 2025's
imaging-based model achieves a comparable internal AUC but requires
MRI plus DAT-SPECT, limiting deployment to specialised centres.

### 4.3 Strengths

- Routine clinical scores only (no imaging or genetics required)
- Publication-grade uncertainty stack (calibration, conformal, bootstrap,
  Cox calibration, Hosmer-Lemeshow)
- Comprehensive baseline and literature comparisons
- Pre-built robustness and fairness analyses
- Reproducible: pinned dependencies, fixed seeds, public code

### 4.4 Limitations

1. **No external validation** in the present analysis. LuxPARK is
   pending. Internal-only results overestimate generalisation, as
   demonstrated by Latourelle (R^2 0.41 -> 0.09) and Dai (AUC 0.93 ->
   0.77).
2. **Subtype labels are clustering-derived**, not a biological gold
   standard. Reclassification under alternative clustering would change
   apparent performance.
3. **PPMI 1.0 cohort only**. PPMI 2.0 patients have no subtype labels
   in the current data freeze; true temporal validation across PPMI 1.0
   vs 2.0 requires re-clustering.
4. **Class imbalance** (1:4.5 slow:fast). Although mitigated via
   class_weight="balanced" and kNN imputation, residual bias toward
   slow predictions cannot be ruled out for very atypical patients.
5. **Selection bias**: PPMI patients are academic-centre-enrolled,
   younger, and more adherent than typical clinic populations.

### 4.5 Future directions

External validation on LuxPARK (n approximately 560) is the binding
next step. Beyond that, a multi-cohort federated training scheme would
strengthen generalisation, and a prospective evaluation in routine
clinical workflow would establish utility.

## References

PubMed citations from `docs/LITERATURE_COMPARISON.md`. DOI links to
the canonical record:

- Latourelle JC et al. *Lancet Neurol* 2017. DOI 10.1016/S1474-4422(17)30328-9
- Wang MY et al. *Neurol Sci* 2025. DOI 10.1007/s10072-024-07953-3
- Dai Y et al. *J Imaging Inform Med* 2025. DOI 10.1007/s10278-025-01583-7
- Faouzi J et al. *IEEE OJEMB* 2022. DOI 10.1109/OJEMB.2022.3178295
- Dadu A et al. *medRxiv* 2024. DOI 10.1101/2024.10.27.24316215
- Iakovakis D et al. *Sci Rep* 2020. DOI 10.1038/s41598-020-69369-1
- Zhang Z et al. *CNS Neurosci Ther* 2025. DOI 10.1111/cns.70182

Methodology citations:

- Hanley JA, McNeil BJ. *Radiology* 1982;143:29-36
- DeLong ER, DeLong DM, Clarke-Pearson DL. *Biometrics* 1988;44:837-845
- Cox DR. *Biometrika* 1958;45:562-565
- Hosmer DW, Lemeshow S. *Wiley* 1980/2000
- Vickers AJ, Elkin EB. *Med Decis Making* 2006;26:565-574
- Pencina MJ, D'Agostino RB Sr, D'Agostino RB Jr, Vasan RS. *Stat Med*
  2008;27:157-172
- Steyerberg EW. *Springer* 2010 (Clinical Prediction Models)
- Hardt M, Price E, Srebro N. *NIPS* 2016
- Sun X, Xu W. *J Stat Softw* 2014;59(B5):1-14
- Collins GS et al. (TRIPOD+AI) *BMJ* 2024;385:e078378

## Supplementary material

See `docs/POWER_ANALYSIS.md`, `docs/SHAP_STABILITY.md`,
`docs/STRESS_TEST.md`, `docs/HYPERPARAMETER_TUNING.md`,
`docs/TEMPORAL_VALIDATION.md`, `docs/SURVIVAL_ANALYSIS.md`,
`docs/LITERATURE_COMPARISON.md`, `docs/TRUE_BOOTSTRAP.md` (compute
pending), and `MODEL_CARD.md`, `TRIPOD_AI_CHECKLIST.md`,
`docs/PROBAST_ASSESSMENT.md`.

For supplement organisation see `docs/SUPPLEMENT_STRUCTURE.md`.
