# Comparison with Published Parkinson's Disease Progression Models

Stand 2026-05-20. Stellt unsere Headline-Ergebnisse in den Kontext
publizierter ML-Modelle zur Vorhersage der Parkinson-Krankheitsprogression
auf PPMI und vergleichbaren Kohorten. Diese Datei dient als
methodische Begruendung im Methods-Kapitel und als direkter
Reviewer-Antwort fuer die Frage "Wie schneidet ihr im Vergleich zu
bestehenden Ansaetzen ab?".

Alle Referenzen sind aus PubMed bezogen und enthalten DOI-Links auf den
Originalartikel.

## Direct comparisons: PPMI-based progression models

### Wang et al. 2025 -- Cognitive trajectory subtypes

[DOI 10.1007/s10072-024-07953-3](https://doi.org/10.1007/s10072-024-07953-3)

- **Cohort:** n=337 early PPMI patients, 6-year follow-up
- **Outcome:** Two cognitive trajectories via Latent Class Mixed Models
  (LCMM): stable (81.9%) vs deteriorating (18.1%)
- **Predictors:** Six baseline clinical variables (age, LNS, SDM, JLO,
  HVLT-R, RBD)
- **Model:** Nomogram (multivariable logistic regression)
- **Reported AUC:** 0.92 (internal CV, no external validation)
- **Comment:** Closely matches our setup -- same database (PPMI), same
  ~82:18 class imbalance, same outcome paradigm (binary progression
  subtype from latent-time clustering). Our headline AUC of 0.94 (Random
  Forest, slopes+intercepts on 17 routine scores) is within 1 SE of this
  benchmark, suggesting we extract similar discriminative information
  from a similar feature space.

### Latourelle et al. 2017 -- Motor progression Bayesian ensemble

[DOI 10.1016/S1474-4422(17)30328-9](https://doi.org/10.1016/S1474-4422(17)30328-9)

- **Cohort:** n=312 PPMI training, n=317 LABS-PD validation
- **Outcome:** Continuous annual rate of change in MDS-UPDRS II + III
- **Predictors:** Baseline clinical, molecular, genetic
- **Model:** Bayesian multivariate predictive inference (Reverse
  Engineering / Forward Simulation, REFS)
- **Reported performance:** R² = 41% (PPMI 5-fold CV), R² = 9% (LABS-PD
  external)
- **Comment:** Predicts a continuous outcome, not directly comparable to
  our binary AUC. The large drop from internal to external (41% to 9%)
  is the canonical example of why external validation matters and is the
  reason we treat our LuxPARK plans as central.

### Latourelle 2017 simulated trials

The same paper showed that incorporating predicted motor progression
rates reduces required trial sizes by up to 20%, demonstrating clinical
utility of progression prediction beyond the AUC metric.

### Faouzi et al. 2022 -- Impulse Control Disorders

[DOI 10.1109/OJEMB.2022.3178295](https://doi.org/10.1109/OJEMB.2022.3178295)

- **Cohort:** n=380 PPMI training, n=388 DIGPD external
- **Outcome:** Next-visit ICD occurrence (different from our outcome)
- **Model:** Recurrent neural network on longitudinal clinical features
- **Reported AUC:** 0.85 [0.80-0.90] internal, 0.802 [0.78-0.83] external
- **Comment:** Includes proper external validation -- excellent template
  for our LuxPARK comparison. The recurrent neural network only marginally
  outperformed a trivial "same as last visit" baseline -- shows the
  importance of comparing against simple baselines, which we have done
  via constant-slow / single-feature LogReg.

## Imaging-based models on PPMI

### Dai et al. 2025 -- Multimodal MRI + DAT-SPECT

[DOI 10.1007/s10278-025-01583-7](https://doi.org/10.1007/s10278-025-01583-7)

- **Cohort:** PPMI internal + external test set, fast vs slow stratification
- **Predictors:** Conventional MRI + DAT-SPECT radiomics + clinical
- **Reported AUC:** 0.93 [0.80-1.00] internal, 0.77 [0.53-0.93] external
- **Comment:** Same binary outcome as ours, but uses imaging features
  (clinically much more expensive). Internal AUC matches ours (0.94),
  external drops further -- demonstrates the typical performance loss
  to expect on LuxPARK. Our LuxPARK validation should aim for AUC > 0.80.

### Dadu et al. 2024 -- MRI imaging score (preprint)

[DOI 10.1101/2024.10.27.24316215](https://doi.org/10.1101/2024.10.27.24316215)

- **Cohort:** PPMI 684 PD images / 319 participants, validated on UK
  Biobank n=42,835
- **Outcome:** PD vs control classification + survival
- **Reported AUC:** 0.63 [0.57-0.71] PD detection on T1w MRI features;
  time-dependent AUC 0.89 [0.85-0.94] for 5-year PD diagnosis on Biobank
- **Comment:** Different task (diagnosis vs progression-subtype), but
  good demonstration of biobank-scale external validation.

## Smartphone-based models

### Iakovakis et al. 2020 -- Touchscreen typing

[DOI 10.1038/s41598-020-69369-1](https://doi.org/10.1038/s41598-020-69369-1)

- **Cohort:** n=39 PD patients + healthy controls, plus 253 remote
  participants for replication
- **Outcome:** Fine-motor impairment / UPDRS-III item correlation
- **Predictors:** Smartphone keystroke dynamics, deep learning
- **Reported AUC:** 0.89 [0.80-0.96] training, 0.97 [0.93-1.00] de novo PD vs
  controls, 0.79 [0.66-0.91] remote cohort
- **Comment:** Different sensor modality, but cited as evidence that
  longitudinal data sources beyond clinical scores can predict PD
  progression.

## Cognition-specific models

### Zhang et al. 2025 -- Depression subtypes in PD (MRI radiomics)

[DOI 10.1111/cns.70182](https://doi.org/10.1111/cns.70182)

- **Cohort:** n=272 PPMI patients, n=45 NACC external
- **Outcome:** Depression subtype in PD
- **Reported AUC:** 0.731 (traditional), 0.853 training / 0.81 testing
  (high-risk subtype model)
- **Comment:** Non-motor outcome, but useful for showing how subtype
  models perform.

## Summary table -- direct comparators

| Study                  | Cohort                 | Outcome             | Predictors                    | AUC internal       | AUC external       |
|------------------------|------------------------|---------------------|-------------------------------|--------------------|--------------------|
| **Ours**               | PPMI n=409             | Fast vs slow        | 17 clinical scores (slopes+intercepts) | 0.94 [0.91, 0.97]  | (LuxPARK pending)  |
| Wang 2025              | PPMI n=337             | Cognitive subtype   | 6 baseline clinical           | 0.92               | --                 |
| Dai 2025               | PPMI                   | Fast vs slow motor  | MRI + DAT-SPECT + clinical    | 0.93 [0.80, 1.00]  | 0.77 [0.53, 0.93]  |
| Latourelle 2017        | PPMI n=312             | Motor rate (cont.)  | Clinical + genetic + CSF      | R²=0.41            | R²=0.09 (LABS-PD)  |
| Faouzi 2022            | PPMI n=380             | ICD (different)     | Longitudinal clinical RNN     | 0.85 [0.80, 0.90]  | 0.802 [0.78, 0.83] |

## Positioning of our work

1. **We achieve state-of-the-art internal AUC** (0.94) on a routine clinical
   feature set, matching imaging-based pipelines while using only
   inexpensive bedside scores.
2. **We provide systematically reported uncertainty** -- bootstrap CIs,
   Conformal prediction sets, calibration intercept and slope --
   substantially beyond the typical single-AUC reporting in the cited
   work.
3. **External validation is the next critical step.** All cited PPMI
   models show meaningful AUC drops on external cohorts (Dai 2025:
   0.93 -> 0.77; Latourelle 2017: R² 0.41 -> 0.09). Our LuxPARK validation
   is therefore the most important pending analysis for the paper.
4. **We compare against simple baselines** (Constant Slow, UPDRS3-only,
   MoCA-only LogReg) -- a step that Faouzi 2022 highlighted as essential
   but is missing from most cited papers.
5. **We pre-specify a multi-comparison correction strategy** (Bonferroni-
   Holm on DeLong, Benjamini-Hochberg FDR on NRI), avoiding cherry-
   picking concerns.

## Citations note

Article references retrieved via PubMed. All DOI links are intended as
inline citations and direct readers to the canonical record. Where
PMC IDs are available, both DOI and PMC are listed in the Model Card.
