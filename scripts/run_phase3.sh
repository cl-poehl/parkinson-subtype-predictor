#!/usr/bin/env bash
# Orchestrator fuer Phase 3 + Item 8: laufende Hyperparameter-Tuning,
# SHAP-Stabilitaet, Stress-Test, True Bootstrap. Aktiviert das urodoc
# Conda-Env und sequenziert die Skripte. Loggt nach logs/.

set -u
ROOT="$( cd "$(dirname "$0")/.." && pwd )"
cd "$ROOT"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
ENV_NAME=urodoc

run() {
    local script="$1"
    local LOG="logs/${script%.py}_${TS}.log"
    echo "=== $script start: $(date) ===" | tee -a "$LOG"
    /opt/homebrew/Caskroom/miniforge/base/envs/$ENV_NAME/bin/python "scripts/$script" >> "$LOG" 2>&1
    echo "=== $script done: $(date), exit=$? ===" | tee -a "$LOG"
}

# Stress-Test laeuft schon separat (start_time before this). Hier nur die
# anderen drei.
run shap_stability.py
run hyperparameter_tuning.py
run true_bootstrap.py
echo "=== Phase 3 + Item 8 COMPLETE: $(date) ==="
