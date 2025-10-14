#!/bin/bash
#
# JDT Parser Service Wrapper
# 
# Wrapper script for the JDT parser service that replaces regex-based parsing
# with robust Eclipse JDT AST parsing.
#

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default JAR path
JDT_PARSER_JAR="${PROJECT_ROOT}/build/libs/jdt-parser-all.jar"

# Check if JAR exists
if [ ! -f "$JDT_PARSER_JAR" ]; then
    echo "Error: JDT parser JAR not found at $JDT_PARSER_JAR" >&2
    echo "Please build it first with: ./gradlew jdtParserJar" >&2
    exit 1
fi

# Run the JDT parser service
java -jar "$JDT_PARSER_JAR" "$@"
