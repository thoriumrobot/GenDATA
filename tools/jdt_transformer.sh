#!/bin/bash
#
# JDT Transformer Service Wrapper
# 
# Wrapper script for the JDT semantic transformer service that replaces regex-based
# transformations with robust Eclipse JDT AST-based transformations.
#

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default JAR path
JDT_TRANSFORMER_JAR="${PROJECT_ROOT}/build/libs/jdt-transformer-all.jar"

# Check if JAR exists
if [ ! -f "$JDT_TRANSFORMER_JAR" ]; then
    echo "Error: JDT transformer JAR not found at $JDT_TRANSFORMER_JAR" >&2
    echo "Please build it first with: ./gradlew jdtTransformerJar" >&2
    exit 1
fi

# Run the JDT transformer service
java -jar "$JDT_TRANSFORMER_JAR" "$@"
