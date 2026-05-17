# Parkinson Subtype Predictor

Web app for predicting Parkinson's disease progression subtype (fast-progressing vs. slow-progressing) from clinical score trajectories.

Trained on the PPMI (Parkinson's Progression Markers Initiative) cohort with three classifiers (Random Forest, XGBoost, Logistic Regression). External validation on the LuxPARK cohort is in preparation.

## Features

- **Single patient**: form-based score entry with prediction probabilities per model and reliability estimate.
- **Batch**: CSV upload for multiple patients, results as download.
- **Demo**: six synthetic patients to try the app without your own data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Models

Six calibrated classifiers, three for multi-visit data using slope and intercept features, three for single-visit data using absolute scores. Trained on either 17 scores (LuxPARK-compatible, "Standard" mode) or 25 scores ("Extended" mode).

Training script: `scripts/train_models.py`.

## Status

Work in progress.
