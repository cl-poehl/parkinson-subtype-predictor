# Parkinson Subtype Predictor

Web-App zur Vorhersage des Parkinson-Progressionssubtyps (fast-progressing vs. slow-progressing) auf Basis klinischer Scores.

Trainiert auf der PPMI-Kohorte mit drei Klassifikatoren (Random Forest, XGBoost, Logistic Regression) plus Toms Likelihood-Ratio-Methode als Vergleich. Externe Validierung auf der LuxPARK-Kohorte geplant.

## Funktionen

- **Einzelpatient**: Maske fuer manuelle Eingabe der Scores eines Patienten, mit Wahrscheinlichkeitsbalken pro Modell und SHAP-Waterfall fuer die Erklaerbarkeit.
- **Batch**: CSV-Upload mit mehreren Patienten, Ausgabe der Predictions als Download-CSV.

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Modelle

Drei kalibrierte Klassifikatoren auf Basis von Slope- und Intercept-Features pro Score. Fuer Patienten ohne Verlaufsdaten zusaetzlich ein Single-Visit-Modell auf Basis absoluter Scores.

Trainings-Skript unter `scripts/train_models.py`.

## Status

Work in Progress.
