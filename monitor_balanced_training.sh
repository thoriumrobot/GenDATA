#!/bin/bash
# Monitor balanced training progress and generate report when complete

GEN_DATA_ROOT="/home/ubuntu/GenDATA"
SQL_QUOTES_LOG="/tmp/generate_sql_balanced_datasets.log"
SIGNATURE_STRING_LOG="/tmp/generate_signature_balanced_datasets.log"
REPORT_SCRIPT="$GEN_DATA_ROOT/generate_balanced_training_metrics_report.py"

# Expected model counts
SQL_QUOTES_EXPECTED=14
SIGNATURE_STRING_EXPECTED=21

echo "Monitoring balanced training progress..."
echo "Expected: SQL Quotes=$SQL_QUOTES_EXPECTED, Signature String=$SIGNATURE_STRING_EXPECTED"
echo ""

while true; do
    # Count trained models
    sql_quotes_trained=$(find "$GEN_DATA_ROOT/models_annotation_types_sql_quotes" -name "*balanced*.pth" 2>/dev/null | wc -l)
    signature_string_trained=$(find "$GEN_DATA_ROOT/models_annotation_types_signature_string" -name "*balanced*.pth" 2>/dev/null | wc -l)
    
    # Check if training processes are still running
    sql_quotes_running=$(ps aux | grep "train_balanced_sql_quotes" | grep python | grep -v grep | wc -l)
    signature_string_running=$(ps aux | grep "train_balanced_signature_string" | grep python | grep -v grep | wc -l)
    
    echo "$(date): SQL Quotes: $sql_quotes_trained/$SQL_QUOTES_EXPECTED models, Signature String: $signature_string_trained/$SIGNATURE_STRING_EXPECTED models"
    
    # Check if training is complete
    if [ $sql_quotes_running -eq 0 ] && [ $signature_string_running -eq 0 ]; then
        echo ""
        echo "Training processes have completed. Generating final report..."
        python3 "$REPORT_SCRIPT"
        echo ""
        echo "Final status:"
        echo "  SQL Quotes: $sql_quotes_trained/$SQL_QUOTES_EXPECTED models"
        echo "  Signature String: $signature_string_trained/$SIGNATURE_STRING_EXPECTED models"
        break
    fi
    
    # Wait before next check
    sleep 300  # Check every 5 minutes
done

