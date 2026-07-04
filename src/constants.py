"""Score-Definitionen und Anzeige-Labels.

Wir unterstuetzen zwei Score-Sets:
- SCORES_LUXPARK (17): Schnittmenge PPMI/LuxPARK, ideal fuer externe Validierung
  und fuer Kliniken, die die PPMI-spezifische Batterie nicht erheben.
- SCORE_LABELS (25): voller PPMI-Score-Umfang, gibt etwas hoehere AUCs.

Welches Set die Webapp benutzt, waehlt der Nutzer ueber den Sidebar-Toggle.
"""

# Voller PPMI-Score-Set (25)
SCORE_LABELS = {
    "UPDRS3_off": "MDS-UPDRS III (Off)",
    "UPDRS3_on": "MDS-UPDRS III (On)",
    "UPDRS1": "MDS-UPDRS I",
    "UPDRS2": "MDS-UPDRS II",
    "UPDRS4": "MDS-UPDRS IV",
    "MOCA": "MoCA",
    "SCOPA": "SCOPA-AUT",
    "RBDScr": "RBD-SQ",
    "LNS": "Letter Number Sequencing",
    "VFT_phon_f": "Verbal Fluency (phonemic)",
    "VFT_sem_sum": "Verbal Fluency (semantic)",
    "HVLT_DR": "HVLT Delayed Recall",
    "HVLT_IR": "HVLT Immediate Recall",
    "JLO": "Judgment of Line Orientation",
    "SDM": "Symbol Digit Modalities Test",
    "SEADL": "Schwab and England ADL",
    "HY_off": "Hoehn and Yahr (Off)",
    "HY_on": "Hoehn and Yahr (On)",
    "AXSC_off": "Axial Score (Off)",
    "AXSC_on": "Axial Score (On)",
    "PIGD_off": "PIGD Score (Off)",
    "PIGD_on": "PIGD Score (On)",
    "ESS": "Epworth Sleepiness Scale",
    "GDS": "Geriatric Depression Scale",
    "LEDD": "Levodopa Equivalent Daily Dose",
}

# LuxPARK-kompatibles Subset (17)
SCORES_LUXPARK = [
    "UPDRS3_off", "UPDRS3_on", "UPDRS1", "UPDRS2", "UPDRS4",
    "MOCA", "SCOPA", "RBDScr", "VFT_phon_f", "JLO",
    "HY_off", "HY_on", "AXSC_off", "AXSC_on", "PIGD_off", "PIGD_on",
    "LEDD",
]

# Werte-Ranges (min, max, default) fuer die Eingabemaske
SCORE_RANGES = {
    "UPDRS3_off": (0, 132, 25),
    "UPDRS3_on": (0, 132, 18),
    "UPDRS1": (0, 52, 8),
    "UPDRS2": (0, 52, 8),
    "UPDRS4": (0, 24, 0),
    "MOCA": (0, 30, 27),
    "SCOPA": (0, 69, 8),
    "RBDScr": (0, 13, 4),
    "LNS": (0, 21, 9),
    "VFT_phon_f": (0, 50, 12),
    "VFT_sem_sum": (0, 60, 18),
    "HVLT_DR": (0, 12, 8),
    "HVLT_IR": (0, 36, 24),
    "JLO": (0, 30, 25),
    "SDM": (0, 110, 40),
    "SEADL": (0, 100, 90),
    "HY_off": (0, 5, 2),
    "HY_on": (0, 5, 2),
    "AXSC_off": (0, 20, 3),
    "AXSC_on": (0, 20, 2),
    "PIGD_off": (0, 20, 3),
    "PIGD_on": (0, 20, 2),
    "ESS": (0, 24, 6),
    "GDS": (0, 15, 3),
    "LEDD": (0, 2000, 400),
}

# Score-Gruppierung fuer die UI (volles 25-Set, 17er-Modus filtert die nicht
# enthaltenen Scores raus)
SCORE_GROUPS = {
    "Motor symptoms": [
        "UPDRS3_off", "UPDRS3_on", "UPDRS2", "UPDRS4",
        "HY_off", "HY_on", "AXSC_off", "AXSC_on", "PIGD_off", "PIGD_on",
    ],
    "Cognition": ["MOCA", "VFT_phon_f", "VFT_sem_sum", "JLO",
                  "HVLT_DR", "HVLT_IR", "LNS", "SDM"],
    "Non-motor symptoms": ["UPDRS1", "SCOPA", "RBDScr", "ESS", "GDS"],
    "Activities of daily living": ["SEADL"],
    "Medication": ["LEDD"],
}

SUBTYPE_LABELS = {1: "Fast Progression", 2: "Slow Progression"}
SUBTYPE_FAST = 1
SUBTYPE_SLOW = 2

# Modell-Pfade pro Score-Set und Modelltyp
MODEL_FILES_LUXPARK = {
    "Random Forest": "models/rf_luxpark_slope.joblib",
    "XGBoost": "models/xgb_luxpark_slope.joblib",
    "Logistic Regression": "models/logreg_luxpark_slope.joblib",
}
MODEL_FILES_LUXPARK_BASELINE = {
    "Random Forest": "models/rf_luxpark_baseline.joblib",
    "XGBoost": "models/xgb_luxpark_baseline.joblib",
    "Logistic Regression": "models/logreg_luxpark_baseline.joblib",
}
MODEL_FILES_FULL = {
    "Random Forest": "models/rf_full_slope.joblib",
    "XGBoost": "models/xgb_full_slope.joblib",
    "Logistic Regression": "models/logreg_full_slope.joblib",
}
MODEL_FILES_FULL_BASELINE = {
    "Random Forest": "models/rf_full_baseline.joblib",
    "XGBoost": "models/xgb_full_baseline.joblib",
    "Logistic Regression": "models/logreg_full_baseline.joblib",
}


def get_score_set(mode):
    """Liste der Score-Codes fuer den gewaehlten Modus."""
    if mode == "luxpark":
        return list(SCORES_LUXPARK)
    return list(SCORE_LABELS.keys())


def get_model_paths(mode, n_visits, imputer="knn"):
    """Gibt die richtigen Modell-Pfade fuer Modus + Visits + Imputer zurueck.

    Default-Imputer ist 'knn' (die deployten Modelle). Fuer alternative
    Imputer (z.B. 'median', 'mice') wird der Suffix in den Dateinamen
    eingefuegt: rf_luxpark_slope.joblib -> rf_luxpark_slope_median.joblib.
    Falls die Datei nicht existiert, faellt es auf knn zurueck.
    """
    if mode == "luxpark":
        base = MODEL_FILES_LUXPARK if n_visits >= 2 else MODEL_FILES_LUXPARK_BASELINE
    else:
        base = MODEL_FILES_FULL if n_visits >= 2 else MODEL_FILES_FULL_BASELINE
    if imputer == "knn":
        return dict(base)
    import os
    out = {}
    for k, v in base.items():
        alt_path = v.replace(".joblib", f"_{imputer}.joblib")
        full = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), alt_path)
        out[k] = alt_path if os.path.exists(full) else v
    return out


def get_conformal_paths(mode, n_visits, imputer="knn"):
    """Pfade zu den SplitConformalClassifier-Joblibs (parallel zu den Modellen)."""
    base = get_model_paths(mode, n_visits, imputer=imputer)
    return {k: v.replace(".joblib", "_conformal.joblib") for k, v in base.items()}


# --- Missing-core fallback routing (paper 3.4.2) -------------------------------
# Each of the three MDS-UPDRS core parts (I, II, III); III spans both med states.
CORE_PART = {"UPDRS1": "I", "UPDRS2": "II", "UPDRS3_off": "III", "UPDRS3_on": "III"}

# Scores dropped for each core-presence fallback (must match train_fallback_models).
CORE_DROP = {
    "coreI":    ["UPDRS2", "UPDRS3_off", "UPDRS3_on"],
    "coreII":   ["UPDRS1", "UPDRS3_off", "UPDRS3_on"],
    "coreIII":  ["UPDRS1", "UPDRS2"],
    "coreNone": ["UPDRS1", "UPDRS2", "UPDRS3_off", "UPDRS3_on"],
}


def core_presence_route(measured_scores):
    """Which fallback (if any) a patient should use, given the score codes they
    actually have measured. Returns None to use the default full model when at
    most one MDS-UPDRS core part is missing (imputation is adequate, 3.4.2), or
    'coreI'/'coreII'/'coreIII'/'coreNone' when two or more core parts are missing
    -- keyed on the (at most one) core part that remains."""
    present = {CORE_PART[s] for s in measured_scores if s in CORE_PART}
    missing = {"I", "II", "III"} - present
    if len(missing) < 2:
        return None
    if present == {"I"}:
        return "coreI"
    if present == {"II"}:
        return "coreII"
    if present == {"III"}:
        return "coreIII"
    return "coreNone"


def fallback_scores(active_scores, pattern):
    """The native score set a given fallback was trained on."""
    drop = set(CORE_DROP[pattern])
    return [s for s in active_scores if s not in drop]


def get_fallback_model_paths(mode, n_visits, pattern, imputer="knn"):
    """Model paths for a core-presence fallback, parallel to get_model_paths."""
    base = get_model_paths(mode, n_visits, imputer=imputer)
    return {k: v.replace(".joblib", f"_{pattern}.joblib") for k, v in base.items()}


def get_fallback_conformal_paths(mode, n_visits, pattern, imputer="knn"):
    base = get_fallback_model_paths(mode, n_visits, pattern, imputer=imputer)
    return {k: v.replace(".joblib", "_conformal.joblib") for k, v in base.items()}


def get_vennabers_paths(mode, n_visits, imputer="knn"):
    """Paths to the Venn-Abers calibrators (parallel to the models)."""
    base = get_model_paths(mode, n_visits, imputer=imputer)
    return {k: v.replace(".joblib", "_vennabers.joblib") for k, v in base.items()}


def get_fallback_vennabers_paths(mode, n_visits, pattern, imputer="knn"):
    base = get_fallback_model_paths(mode, n_visits, pattern, imputer=imputer)
    return {k: v.replace(".joblib", "_vennabers.joblib") for k, v in base.items()}


# Backward-Compat (alte Imports)
MODEL_FILES = MODEL_FILES_LUXPARK
MODEL_FILES_BASELINE = MODEL_FILES_LUXPARK_BASELINE
