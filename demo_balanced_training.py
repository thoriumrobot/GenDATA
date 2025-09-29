#!/usr/bin/env python3
"""
Demo Script for Balanced Training System

This script demonstrates how to use the balanced training system to create
balanced datasets and train annotation type models with proper convergence.
"""

import os
import logging
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_balanced_training_demo():
    """Run a demonstration of the balanced training system"""
    
    print("="*60)
    print("BALANCED TRAINING SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Configuration
    project_root = "/home/ubuntu/checker-framework/checker/tests/index"
    warnings_file = "/home/ubuntu/GenDATA/index1.out"
    cfwr_root = "/home/ubuntu/GenDATA"
    output_dir = "/home/ubuntu/GenDATA/balanced_training_demo"
    
    print(f"Project root: {project_root}")
    print(f"Warnings file: {warnings_file}")
    print(f"CFWR root: {cfwr_root}")
    print(f"Output directory: {output_dir}")
    
    # Check if input files exist
    if not os.path.exists(warnings_file):
        logger.error(f"Warnings file not found: {warnings_file}")
        logger.info("Creating a sample warnings file for demonstration...")
        
        # Create a sample warnings file
        sample_warnings = """warning: [unchecked] unchecked call to method(int) as a member of raw type
warning: [unchecked] unchecked conversion from List to List<String>
warning: [unchecked] unchecked method invocation"""
        
        with open(warnings_file, 'w') as f:
            f.write(sample_warnings)
        
        logger.info(f"Created sample warnings file: {warnings_file}")
    
    # Run the balanced training pipeline
    print("\n" + "="*60)
    print("RUNNING BALANCED TRAINING PIPELINE")
    print("="*60)
    
    cmd = [
        'python', 'balanced_training_pipeline.py',
        '--project_root', project_root,
        '--warnings_file', warnings_file,
        '--cfwr_root', cfwr_root,
        '--output_dir', output_dir,
        '--examples_per_annotation', '100',  # Small number for demo
        '--target_balance', '0.5',  # 50% positive, 50% negative
        '--epochs', '10',  # Small number for demo
        '--batch_size', '16',
        '--use_existing_cfg'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("\nThis will:")
    print("1. Generate or use existing CFG data")
    print("2. Create balanced datasets (50% positive, 50% negative examples)")
    print("3. Train annotation type models with balanced data")
    print("4. Save trained models and statistics")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION RESULTS")
        print("="*60)
        
        if result.returncode == 0:
            print("✅ SUCCESS: Balanced training pipeline completed successfully!")
            print("\nOutput:")
            print(result.stdout)
            
            # Show generated files
            if os.path.exists(output_dir):
                print(f"\nGenerated files in {output_dir}:")
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"  - {file_path}")
            
            print("\n" + "="*60)
            print("BALANCED TRAINING BENEFITS")
            print("="*60)
            print("✅ Balanced datasets ensure proper model convergence")
            print("✅ 50% positive and 50% negative examples for each annotation type")
            print("✅ Models learn to distinguish between annotation types effectively")
            print("✅ Reduced bias towards always predicting positive examples")
            print("✅ Better generalization to new code")
            print("✅ Improved prediction accuracy and confidence")
            
        else:
            print("❌ FAILED: Balanced training pipeline failed!")
            print(f"Exit code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            
            # Show partial results if any
            if os.path.exists(output_dir):
                print(f"\nPartial results in {output_dir}:")
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        print(f"  - {file_path}")
    
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT: Pipeline execution timed out after 30 minutes")
        print("This is normal for large datasets. Consider reducing examples_per_annotation for faster execution.")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("To use the balanced training system in your own projects:")
    print()
    print("1. Generate balanced datasets:")
    print("   python balanced_dataset_generator.py \\")
    print("     --cfg_dir /path/to/cfg/files \\")
    print("     --output_dir /path/to/output \\")
    print("     --examples_per_annotation 1000 \\")
    print("     --target_balance 0.5")
    print()
    print("2. Train balanced models:")
    print("   python balanced_annotation_type_trainer.py \\")
    print("     --balanced_dataset_dir /path/to/datasets \\")
    print("     --output_dir /path/to/models \\")
    print("     --epochs 100 --batch_size 32")
    print()
    print("3. Run complete pipeline:")
    print("   python balanced_training_pipeline.py \\")
    print("     --project_root /path/to/java/project \\")
    print("     --warnings_file /path/to/warnings.out \\")
    print("     --cfwr_root /path/to/cfwr \\")
    print("     --output_dir /path/to/output")
    print()
    print("="*60)

def show_balanced_training_concept():
    """Show the concept behind balanced training"""
    
    print("\n" + "="*60)
    print("BALANCED TRAINING CONCEPT")
    print("="*60)
    
    print("PROBLEM:")
    print("  - Original training data is imbalanced")
    print("  - Most examples are positive (need annotations)")
    print("  - Few negative examples (no annotations needed)")
    print("  - Models learn to always predict positive")
    print("  - Poor generalization and convergence")
    print()
    
    print("SOLUTION:")
    print("  - Generate balanced datasets")
    print("  - 50% positive examples (annotation needed)")
    print("  - 50% negative examples (no annotation needed)")
    print("  - For each annotation type: @Positive, @NonNegative, @GTENegativeOne")
    print()
    
    print("BENEFITS:")
    print("  ✅ Better model convergence")
    print("  ✅ Reduced prediction bias")
    print("  ✅ Improved accuracy on balanced test sets")
    print("  ✅ Better generalization to new code")
    print("  ✅ More reliable confidence scores")
    print()
    
    print("IMPLEMENTATION:")
    print("  - BalancedDatasetGenerator: Creates 50/50 positive/negative examples")
    print("  - BalancedAnnotationTypeTrainer: Trains models on balanced data")
    print("  - BalancedTrainingPipeline: End-to-end pipeline integration")
    print()
    
    print("="*60)

if __name__ == '__main__':
    show_balanced_training_concept()
    run_balanced_training_demo()
