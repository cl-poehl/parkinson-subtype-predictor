#!/usr/bin/env bash
# Wartet bis D (SubtypePredictions kNN-Re-Run) durch ist, kopiert die
# aktualisierten CSVs ins Webapp-data/ Verzeichnis, regeneriert
# Publikations-Figures, committet und pusht.
#
# Trigger: kein run_*.py-Prozess mehr aktiv im SubtypePredictions-Repo.
set -u

SUBTYPES_REPO="$HOME/Documents/SubtypePredictions"
WEBAPP_REPO="$HOME/Documents/parkinson-subtype-predictor"
INTER="$SUBTYPES_REPO/intermediate_data"
DATA="$WEBAPP_REPO/data"
LOG="$WEBAPP_REPO/logs/sync_intermediate_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$WEBAPP_REPO/logs"

echo "=== sync_intermediate_data start: $(date) ===" | tee -a "$LOG"

# Warte bis kein run_*.py-Script mehr laeuft
while pgrep -f "python run_.*\.py" > /dev/null 2>&1; do
    sleep 60
done
# Plus zusaetzlich auf den D-Orchestrator-Shell warten
while pgrep -f "Start KNN re-run" > /dev/null 2>&1; do
    sleep 60
done
echo "D-Re-Run finished at $(date)" | tee -a "$LOG"

# Pruefe, ob die noetigen Files existieren
FILES=(
    "ml_score_combinations.csv"
    "ml_random_score_combinations.csv"
    "lr_random_score_combinations.csv"
    "ml_per_score_roc_auc.csv"
    "ml_follow_up_simulation.csv"
    "ml_missingness_simulation.csv"
    "ml_missingness_followup_simulation.csv"
    "ml_missingness_followup_simulation_luxpark.csv"
    "ml_missingness_followup_simulation_full.csv"
    "ml_missingness_simulation_bootstrap.csv"
    "ml_missingness_bootstrap_luxpark.csv"
    "ml_missingness_bootstrap_full.csv"
)

copied=0
skipped=0
for f in "${FILES[@]}"; do
    if [ -f "$INTER/$f" ]; then
        cp -p "$INTER/$f" "$DATA/$f"
        echo "  copied: $f" | tee -a "$LOG"
        ((copied++))
    else
        echo "  skipped (not in intermediate_data): $f" | tee -a "$LOG"
        ((skipped++))
    fi
done
echo "Copied $copied, skipped $skipped" | tee -a "$LOG"

# Regeneriere die SVG-Figures mit den neuen Daten
echo "Regenerating publication figures..." | tee -a "$LOG"
cd "$WEBAPP_REPO"
/opt/homebrew/Caskroom/miniforge/base/envs/urodoc/bin/python \
    scripts/generate_publication_figures.py >> "$LOG" 2>&1
echo "  figures regenerated" | tee -a "$LOG"

# Commit + Push
cd "$WEBAPP_REPO"
git add data/*.csv figures/*.svg 2>>"$LOG"
if git diff --cached --quiet; then
    echo "No changes to commit." | tee -a "$LOG"
else
    git commit -m "Sync intermediate_data: kNN-Imputation-CSVs aus dem PPMI-Re-Run

D (SubtypePredictions kNN-Re-Run) wurde abgeschlossen. Diese CSVs
ueberschreiben die alten median-imputation-Versionen, sodass die
Supplementary Figures (SF1-SF4) konsistent zum kNN-Imputation-Claim
im Main-Methods-Text sind.

Files synchronisiert: $copied / ${#FILES[@]}
- Score combinations (Greedy + Random)
- Per-score AUC
- Follow-up sensitivity
- Missingness sensitivity (plus _bootstrap, _luxpark, _full)
- Missingness x Follow-up Grid (plus _luxpark, _full)

Publication-grade SVGs (figures/*.svg) ebenfalls neu generiert mit
den aktualisierten Daten." 2>>"$LOG"
    git push 2>>"$LOG"
    echo "Committed and pushed at $(date)" | tee -a "$LOG"
fi

echo "=== sync_intermediate_data done: $(date) ===" | tee -a "$LOG"
