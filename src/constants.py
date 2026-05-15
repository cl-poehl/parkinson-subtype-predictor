"""Score-Definitionen und Anzeige-Labels.
Stimmt mit dem PPMI-Hauptprojekt ueberein."""

SCORE_LABELS = {
    "UPDRS3_off": "MDS-UPDRS III (Off)",
    "UPDRS3_on": "MDS-UPDRS III (On)",
    "UPDRS1": "MDS-UPDRS I",
    "UPDRS2": "MDS-UPDRS II",
    "UPDRS4": "MDS-UPDRS IV",
    "MOCA": "MoCA",
    "SCOPA": "SCOPA-AUT",
    "RBDScr": "RBD-SQ",
    "VFT_phon_f": "Verbal Fluency (phonemic)",
    "JLO": "Judgment of Line Orientation",
    "HY_off": "Hoehn and Yahr (Off)",
    "HY_on": "Hoehn and Yahr (On)",
    "AXSC_off": "Axial Score (Off)",
    "AXSC_on": "Axial Score (On)",
    "PIGD_off": "PIGD Score (Off)",
    "PIGD_on": "PIGD Score (On)",
    "LEDD": "Levodopa Equivalent Daily Dose",
}

# Range pro Score fuer die Eingabemaske (min, max, default)
SCORE_RANGES = {
    "UPDRS3_off": (0, 132, 25),
    "UPDRS3_on": (0, 132, 18),
    "UPDRS1": (0, 52, 8),
    "UPDRS2": (0, 52, 8),
    "UPDRS4": (0, 24, 0),
    "MOCA": (0, 30, 27),
    "SCOPA": (0, 69, 8),
    "RBDScr": (0, 13, 4),
    "VFT_phon_f": (0, 50, 12),
    "JLO": (0, 30, 25),
    "HY_off": (0, 5, 2),
    "HY_on": (0, 5, 2),
    "AXSC_off": (0, 20, 3),
    "AXSC_on": (0, 20, 2),
    "PIGD_off": (0, 20, 3),
    "PIGD_on": (0, 20, 2),
    "LEDD": (0, 2000, 400),
}

SUBTYPE_LABELS = {1: "Fast Progression", 2: "Slow Progression"}
SUBTYPE_FAST = 1
SUBTYPE_SLOW = 2

MODEL_FILES = {
    "Random Forest": "models/rf_slope.joblib",
    "XGBoost": "models/xgb_slope.joblib",
    "Logistic Regression": "models/logreg_slope.joblib",
}
MODEL_FILES_BASELINE = {
    "Random Forest": "models/rf_baseline.joblib",
    "XGBoost": "models/xgb_baseline.joblib",
    "Logistic Regression": "models/logreg_baseline.joblib",
}
