#!/usr/bin/env python3
"""
Demo script showing how to use the predictions viewer
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show its output"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Show first 50 lines of output
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines[:50]):
                print(line)
            if len(lines) > 50:
                print(f"... ({len(lines) - 50} more lines)")
        else:
            print(f"Error: {result.stderr}")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    print("🎯 PREDICTIONS VIEWER DEMO")
    print("=" * 50)
    print("This demo shows how to use the predictions viewer script")
    print("to examine model predictions with their code context.")
    
    # Check if prediction files exist
    predictions_dir = "predictions_annotation_types"
    if not os.path.exists(predictions_dir):
        print(f"\n❌ Predictions directory '{predictions_dir}' not found!")
        print("Please run the training pipeline first to generate predictions.")
        return
    
    # Show summary
    run_command(
        ["python", "show_predictions_with_context.py", "--summary"],
        "Summary of All Predictions"
    )
    
    # Show predictions for a specific file with low confidence threshold
    run_command(
        ["python", "show_predictions_with_context.py", 
         "--file", "predictions_annotation_types/Collections.java.predictions.json",
         "--confidence-threshold", "0.55", "--context-lines", "2"],
        "Predictions for Collections.java (confidence > 55%)"
    )
    
    # Show only high confidence predictions
    run_command(
        ["python", "show_predictions_with_context.py", 
         "--file", "predictions_annotation_types/Collections.java.predictions.json",
         "--confidence-threshold", "0.7", "--context-lines", "1"],
        "High Confidence Predictions for Collections.java (confidence > 70%)"
    )
    
    # Show predictions filtered by model type
    run_command(
        ["python", "show_predictions_with_context.py", 
         "--filter-model", "enhanced_causal", "--confidence-threshold", "0.6"],
        "All Predictions by Enhanced Causal Model (confidence > 60%)"
    )
    
    print(f"\n{'='*60}")
    print("🎉 DEMO COMPLETE")
    print(f"{'='*60}")
    print("\n📚 USAGE EXAMPLES:")
    print("1. Show all predictions with context:")
    print("   python show_predictions_with_context.py")
    print("\n2. Show predictions for a specific file:")
    print("   python show_predictions_with_context.py --file predictions_annotation_types/Collections.java.predictions.json")
    print("\n3. Show only high confidence predictions:")
    print("   python show_predictions_with_context.py --confidence-threshold 0.8")
    print("\n4. Show predictions by specific model:")
    print("   python show_predictions_with_context.py --filter-model enhanced_causal")
    print("\n5. Show predictions by annotation type:")
    print("   python show_predictions_with_context.py --filter-annotation @Positive")
    print("\n6. Show summary only:")
    print("   python show_predictions_with_context.py --summary")

if __name__ == '__main__':
    main()
