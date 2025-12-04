#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/test_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/bounded_test_run.log"
PROGRESS_LOG="${LOG_DIR}/pytest_progress.log"
export PYTEST_PROGRESS_LOG="$PROGRESS_LOG"

echo "=== TEST RUN START $(date) ===" | tee "$LOG_FILE"
echo "Logs: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Progress: $PROGRESS_LOG" | tee -a "$LOG_FILE"

run_pytest() {
  local desc="$1"; shift
  echo "--- BEGIN: $desc @ $(date) ---" | tee -a "$LOG_FILE"
  # Harden each invocation with an overall timeout (in addition to per-test timeouts)
  if ! timeout 300s env PYTEST_TIMEOUT=30 pytest -q "$@" -vv -s --maxfail=1 --durations=10 2>&1 | tee -a "$LOG_FILE"; then
    echo "--- FAIL: $desc @ $(date) ---" | tee -a "$LOG_FILE"
    return 1
  fi
  echo "--- PASS: $desc @ $(date) ---" | tee -a "$LOG_FILE"
}

# 1) Collection-only sanity
run_pytest "collect-only" --collect-only || exit 1

# 2) Unit tests
run_pytest "unit tests" tests/unit || exit 1

# 3) Integration tests (one file at a time)
for f in \
  tests/integration/test_cfg_generation.py \
  tests/integration/test_gt_extraction.py \
  tests/integration/test_predictions_smoke.py \
  tests/integration/test_metrics_and_aggregation.py \
  tests/integration/test_validate_summary.py; do
  run_pytest "integration $f" "$f" || exit 1
done

echo "=== TEST RUN END $(date) ===" | tee -a "$LOG_FILE"


