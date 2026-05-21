# Parkinson Subtype Predictor

Publication-grade web app for predicting Parkinson's disease progression
subtype (fast-progressing vs slow-progressing) from longitudinal
clinical score trajectories. Trained on the PPMI (Parkinson's
Progression Markers Initiative) cohort.

Three machine-learning classifiers (Random Forest, XGBoost,
L1-Logistic Regression) plus the reference Likelihood-Ratio method. Wrapped
in CalibratedClassifierCV (isotonic) and MAPIE Split-Conformal (90%
coverage). External validation on the LuxPARK Luxembourg cohort is in
preparation.

## Features

**Web app (`streamlit run app.py`)**

- **Single Patient** -- form-based entry, per-classifier predictions
  with conformal sets, expected AUC, confidence CI.
- **Batch** -- CSV upload for multiple patients.
- **Demo** -- six synthetic patients to try without your own data.
- **About** -- methodology + 19 publication-grade analyses including
  bootstrap CIs, calibration diagnostics, decision curve analysis,
  DeLong tests with FWER correction, subgroup fairness, PDP/ICE,
  stress test, hyperparameter robustness, Cox survival analysis,
  literature comparison.

**Per-patient diagnostics** (Single Patient / Batch / Demo) include
five scientific context panels: calibration anchor, threshold table,
noise robustness, time-to-Hoehn-Yahr-3 prediction (Cox c-index 0.874),
single-feature-baseline comparison.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.14 recommended.

## Repository structure

```
parkinson-subtype-predictor/
|-- app.py                            # Main Streamlit entry
|-- views/                            # Tab implementations
|   |-- single_patient.py
|   |-- batch.py
|   |-- demo.py
|   |-- about.py
|   `-- _utils.py                     # Shared prediction + render helpers
|-- src/                              # Library code (importable)
|   |-- clinical_metrics.py           # DeLong, NRI/IDI, DCA, Cox cal, HL, FWER
|   |-- conformal.py                  # MAPIE Split-Conformal wrapper
|   |-- counterfactuals.py            # DiCE + single-feature counterfactuals
|   |-- features.py                   # Slope + intercept extraction
|   |-- inference.py                  # Model loading + prediction
|   |-- lr_method.py                  # Reference Likelihood Ratio method
|   |-- reliability.py                # Expected AUC lookup
|   |-- shap_utils.py                 # SHAP across calibration folds
|   |-- survival.py                   # Cox time-to-HY-3 inference
|   |-- baselines.py                  # UPDRS3-only + MoCA-only LogReg
|   `-- robustness.py                 # Per-patient noise sensitivity
|-- scripts/                          # Training + analysis scripts
|   |-- train_models.py               # Main 3-classifier training
|   |-- train_full_models.py          # Cox + baselines (no CV)
|   |-- train_baselines.py            # Baselines via CV
|   |-- compute_pdp.py                # Partial Dependence + ICE
|   |-- compute_empirical_coverage.py # Validate conformal guarantee
|   |-- power_analysis.py             # Hanley-McNeil sample size
|   |-- survival_analysis.py          # Cox PH analysis
|   |-- temporal_validation.py        # Within-PPMI-1.0 split
|   |-- stress_test.py                # Noise robustness
|   |-- shap_stability.py             # SHAP bootstrap stability
|   |-- hyperparameter_tuning.py      # Optuna nested CV
|   |-- true_bootstrap.py             # Pencina-style bootstrap
|   |-- external_validation.py        # LuxPARK validation hook
|   |-- generate_publication_figures.py
|   |-- generate_table1.py            # Cohort characteristics
|   `-- save_lr_predictions.py
|-- models/                           # Trained joblibs (~30 MB total)
|-- data/                             # Pre-computed metrics + figures source
|-- docs/                             # Paper-ready documentation
|-- figures/                          # Publication-grade SVGs
`-- tests/                            # Pytest sanity tests
```

## Documentation

Paper-ready material in `docs/`:

- `PAPER_DRAFT.md` -- full Manuscript draft with Abstract, Methods, Results, Discussion, References
- `TABLE1_COHORT.md` -- baseline characteristics by subtype with stat tests
- `MODEL_CARD.md` (root) -- Mitchell 2019 + FDA-2023 model card
- `TRIPOD_AI_CHECKLIST.md` (root) -- Collins BMJ 2024 checklist
- `PROBAST_ASSESSMENT.md` -- Wolff Ann Intern Med 2019 risk-of-bias
- `SUPPLEMENT_STRUCTURE.md` -- mapping for online supplement
- `POWER_ANALYSIS.md` -- post-hoc power (Hanley-McNeil 1982)
- `LITERATURE_COMPARISON.md` -- 7 PubMed comparator studies with DOIs
- `SURVIVAL_ANALYSIS.md` -- Cox proportional hazards time-to-HY-3
- `TEMPORAL_VALIDATION.md` -- enrollment-year split within PPMI 1.0
- `EMPIRICAL_COVERAGE.md` -- validation of conformal 90% guarantee
- `STRESS_TEST.md` -- measurement noise robustness
- `SHAP_STABILITY.md` -- feature importance stability over bootstraps
- `HYPERPARAMETER_TUNING.md` -- Optuna nested CV
- `TRUE_BOOTSTRAP.md` -- Pencina-style full-pipeline bootstrap

Publication-grade SVG figures in `figures/`:

- `fig1_dca.svg` -- decision curve analysis
- `fig2_reliability.svg` -- reliability diagrams
- `fig3_shap.svg` -- top SHAP features with bootstrap SD
- `fig4_roc.svg` -- ROC curves with bootstrap-CI labels
- `figS7_km.svg` -- Kaplan-Meier by subtype
- `figS10_stress.svg` -- stress-test flip rates

All SVGs use Liberation Serif 8pt with `svg.fonttype="none"`, editable
in Illustrator / Inkscape.

## Reproducibility

- All training is deterministic with random_state=42
- Dependencies pinned exactly in `requirements.txt`
- PPMI extract `PPMI_PD_2024-03-13.csv` (see MODEL_CARD.md)
- Trained model artefacts in `models/` (12 calibrated classifiers + 12
  conformal wrappers + Cox PH + 2 baseline LogRegs)

## Tests

```bash
python -m pytest tests/ -v
```

Currently 13 sanity tests cover the statistical functions in
`src/clinical_metrics.py` (bootstrap AUC, Cox calibration, Hosmer-Lemeshow,
DeLong, FWER correction, equalized odds, optimal threshold, DCA, NRI/IDI).

## Headline performance

Internal 10-fold patient-grouped CV on PPMI (n=409):

| Method | AUC | 95% CI (bootstrap) |
|---|---|---|
| Random Forest | 0.943 | 0.909-0.974 |
| XGBoost | 0.945 | 0.912-0.973 |
| Logistic Regression | 0.905 | 0.855-0.950 |
| Likelihood Ratio | 0.895 | 0.852-0.936 |

Empirical conformal coverage at 90% target: 0.89-0.93 across all
classifiers (within +/- 0.04, MAPIE guarantee holds). See
`docs/EMPIRICAL_COVERAGE.md`.

## Citation

If you use this code or the publication-grade documentation, please
cite the paper (in preparation). For the libraries:

- MAPIE: Mendil M, Mossina L, Reichelt A et al. SoftwareX 2024
- SHAP: Lundberg SM, Lee SI. NeurIPS 2017
- TRIPOD+AI: Collins GS et al. BMJ 2024;385:e078378
- PROBAST: Wolff RF et al. Ann Intern Med 2019;170:51-58
