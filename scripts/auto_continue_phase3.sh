#!/usr/bin/env bash
# Wartet bis stress_test.py fertig ist, dann laeuft Phase-3-Orchestrator.
# Loggt nach logs/.
set -u
ROOT="$( cd "$(dirname "$0")/.." && pwd )"
cd "$ROOT"
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/auto_continue_${TS}.log"

echo "=== Auto-continue Phase 3 start: $(date) ===" | tee -a "$LOG"
# Warte bis kein stress_test mehr laeuft
while pgrep -f "scripts/stress_test.py" > /dev/null 2>&1; do
    sleep 30
done
echo "stress_test finished at $(date)" | tee -a "$LOG"

bash "$ROOT/scripts/run_phase3.sh" 2>&1 | tee -a "$LOG"
echo "=== Auto-continue Phase 3 done: $(date) ===" | tee -a "$LOG"
