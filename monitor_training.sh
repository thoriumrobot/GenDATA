#!/bin/bash
# Monitor training progress for all checkers

echo "=== Training Progress Monitor ==="
echo ""

# Check running processes
echo "Running Training Processes:"
ps aux | grep -E "train_all|train_sql|train_signature" | grep -v grep || echo "No training processes found"
echo ""

# Check log files
echo "Training Log Files:"
if [ -d "training_logs" ]; then
    for log in training_logs/*.log; do
        if [ -f "$log" ]; then
            size=$(du -h "$log" | cut -f1)
            lines=$(wc -l < "$log" 2>/dev/null || echo "0")
            echo "  $(basename $log): $size, $lines lines"
        fi
    done
else
    echo "  No training_logs directory found"
fi
echo ""

# Check model counts
echo "Model Counts:"
python3 -c "
from checker_evaluation_config import get_checker_config, get_all_checker_names, build_model_name
from pathlib import Path

models_dir = Path('models_annotation_types')
if not models_dir.exists():
    print('Models directory does not exist')
    exit(1)

for checker_name in get_all_checker_names():
    config = get_checker_config(checker_name)
    if not config:
        continue
    
    annotation_types = config.get('annotation_types', [])
    base_models = ['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']
    expected = len(annotation_types) * len(base_models)
    
    found = 0
    all_models = list(models_dir.glob('*.pth')) + list(models_dir.glob('*.pkl'))
    
    for ann_type in annotation_types:
        for base_model in base_models:
            model_name = build_model_name(checker_name, ann_type, base_model)
            found_model = any(
                model_name.replace('_', '').lower() in f.name.replace('_', '').lower().replace('.pth', '').replace('.pkl', '')
                for f in all_models
            )
            if found_model:
                found += 1
    
    pct = (found / expected * 100) if expected > 0 else 0
    print(f'  {checker_name}: {found}/{expected} ({pct:.1f}%)')
" 2>/dev/null || echo "  Error checking model counts"
echo ""

# Show recent log entries
echo "Recent Training Activity (last 10 lines from each log):"
if [ -d "training_logs" ]; then
    for log in training_logs/*.log; do
        if [ -f "$log" ] && [ -s "$log" ]; then
            echo ""
            echo "=== $(basename $log) ==="
            tail -10 "$log" 2>/dev/null | sed 's/^/  /'
        fi
    done
fi

