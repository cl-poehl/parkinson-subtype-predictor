"""Constants used across the Parkinson's progression analysis project."""

import pandas as pd

SUBTYPE_LABELS = {
    1: "Fast-progressing",
    2: "Slow-progressing",
    pd.NA: "No Subtype",
}

SUBTYPE_FAST = 1
SUBTYPE_SLOW = 2

SUBTYPE_COLORS = {1: "tab:blue", 2: "tab:orange", pd.NA: "tab:gray"}

# not used as assessed only in PPMI phase 2: BNT, TMTA, TMTB, NQCOG; not used as not useful: NFL, QUIP, STA,
# Striatum, Striatum_AI, TD
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
    "VFT_phon_f": "Verbal Fluency Test (phonemic)",
    "VFT_sem_sum": "Verbal Fluency Test (semantic)",
    "HVLT_DR": "Hopkins Verbal Learning Test - Delayed Recall",
    "HVLT_IR": "Hopkins Verbal Learning Test - Immediate Recall",
    "JLO": "Judgment of Line Orientation",
    "SDM": "Symbol Digit Modalities Test",
    "SEADL": "Schwab and England Activities of Daily Living",
    "HY_off": "Hoehn and Yahr Stage (Off)",
    "HY_on": "Hoehn and Yahr Stage (On)",
    "AXSC_off": "Axial Score (Off)",
    "AXSC_on": "Axial Score (On)",
    "PIGD_off": "PIGD Score (Off)",
    "PIGD_on": "PIGD Score (On)",
    "ESS": "Epworth Sleepiness Scale",
    "GDS": "Geriatric Depression Scale",
    "LEDD": "Levodopa Equivalent Daily Dose"
}


COLOR_COGNITION = "#FF6347"
COLOR_MOTOR = "#008080"
COLOR_AXIAL = "#00C0C0"
COLOR_OTHER = "#6A5ACD"

SCORE_COLORS = {
    "UPDRS3_off": COLOR_MOTOR,
    "UPDRS3_on": COLOR_MOTOR,
    "UPDRS1": COLOR_OTHER,
    "UPDRS2": COLOR_MOTOR,
    "UPDRS4": COLOR_MOTOR,
    "MOCA": COLOR_COGNITION,
    "SCOPA": COLOR_OTHER,
    "RBDScr": COLOR_OTHER,
    "LNS": COLOR_COGNITION,
    "VFT_phon_f": COLOR_COGNITION,
    "VFT_sem_sum": COLOR_COGNITION,
    "HVLT_DR": COLOR_COGNITION,
    "HVLT_IR": COLOR_COGNITION,
    "JLO": COLOR_COGNITION,
    "SDM": COLOR_COGNITION,
    "SEADL": COLOR_OTHER,
    "HY_off": COLOR_MOTOR,
    "HY_on": COLOR_MOTOR,
    "AXSC_off": COLOR_AXIAL,
    "AXSC_on": COLOR_AXIAL,
    "PIGD_off": COLOR_AXIAL,
    "PIGD_on": COLOR_AXIAL,
    "ESS": COLOR_OTHER,
    "GDS": COLOR_OTHER,
    "LEDD": COLOR_OTHER,
    "log10_lr_total": "#000000"
}

MODEL_LABELS = {
    "absolute_first": "First Score",
    "absolute_all": "All Scores",
    "slopes": "Slopes",
    "slopes+absolute_first": "Slopes + First Score",
    "slopes+absolute_all": "Slopes + All Scores",
}

MODEL_COLORS = {
    "absolute_first": "tab:blue",
    "absolute_all": "tab:orange",
    "slopes": "tab:green",
    "slopes+absolute_first": "tab:red",
    "slopes+absolute_all": "tab:purple",
}

import os as _os
_basepath = _os.path.dirname(_os.path.abspath(__file__))
_file_pd_data = "data/PPMI_PD_2024-03-13.csv"
_file_subtypes = "data/ParkinsonPredict_PPMI_progression_subtypes.csv"

_intermediate_data_dir = "intermediate_data"
