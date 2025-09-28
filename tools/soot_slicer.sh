#!/usr/bin/env bash
set -euo pipefail

# Soot-based slicer interface for CFWR
# Uses the Java-based SootSlicer for proper bytecode slicing
# Accepts:
#   --projectRoot <path>
#   --targetFile <relative-or-abs .java>
#   --line <number>
#   --output <dir>
#   --member <class#sig>
#   --decompiler <vineflower.jar> (optional)

PROJECT_ROOT=""
TARGET_FILE=""
LINE_NO=""
OUTPUT_DIR=""
MEMBER_SIG=""
DECOMPILER_JAR=""
PREDICTION_MODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --projectRoot) PROJECT_ROOT="$2"; shift 2;;
    --targetFile)  TARGET_FILE="$2"; shift 2;;
    --line)        LINE_NO="$2"; shift 2;;
    --output)      OUTPUT_DIR="$2"; shift 2;;
    --member)      MEMBER_SIG="$2"; shift 2;;
    --decompiler)  DECOMPILER_JAR="$2"; shift 2;;
    --prediction-mode) PREDICTION_MODE="--prediction-mode"; shift 1;;
    *) echo "[soot_slicer] Unknown arg: $1"; shift 1;;
  esac
done

if [[ -z "$PROJECT_ROOT" || -z "$TARGET_FILE" || -z "$LINE_NO" || -z "$OUTPUT_DIR" || -z "$MEMBER_SIG" ]]; then
  echo "[soot_slicer] Missing required args" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFWR_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if the SootSlicer JAR exists
SOOT_JAR="$CFWR_ROOT/build/libs/CFWR-all.jar"
if [[ ! -f "$SOOT_JAR" ]]; then
  echo "[soot_slicer] SootSlicer JAR not found at $SOOT_JAR" >&2
  echo "[soot_slicer] Please run './gradlew build' first" >&2
  exit 1
fi

# Build the command to run the Java-based SootSlicer
CMD=("java" "-cp" "$SOOT_JAR" "cfwr.SootSlicer")
CMD+=("--projectRoot" "$PROJECT_ROOT")
CMD+=("--targetFile" "$TARGET_FILE")
CMD+=("--line" "$LINE_NO")
CMD+=("--output" "$OUTPUT_DIR")
CMD+=("--member" "$MEMBER_SIG")

if [[ -n "$DECOMPILER_JAR" ]]; then
  CMD+=("--decompiler" "$DECOMPILER_JAR")
fi

if [[ -n "$PREDICTION_MODE" ]]; then
  CMD+=("$PREDICTION_MODE")
fi

echo "[soot_slicer] Running: ${CMD[*]}"

# Execute the SootSlicer
if "${CMD[@]}"; then
  echo "[soot_slicer] Slicing completed successfully"
else
  echo "[soot_slicer] Slicing failed, exit code: $?" >&2
  exit 1
fi
