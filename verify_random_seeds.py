#!/usr/bin/env python3
"""
Verification script for random seed fixes.

This script verifies that training with fixed random seeds produces
deterministic results (same training run twice gives identical results).
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_deterministic_training(dataset_file: str, num_runs: int = 2) -> bool:
    """
    Verify that training produces identical results across multiple runs
    
    Args:
        dataset_file: Path to balanced dataset JSON file
        num_runs: Number of training runs to compare (default: 2)
        
    Returns:
        True if all runs produce identical results, False otherwise
    """
    from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeTrainer
    
    logger.info(f"Verifying deterministic training with {num_runs} runs...")
    logger.info(f"Dataset: {dataset_file}")
    
    results = []
    
    for run_num in range(num_runs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Run {run_num + 1}/{num_runs}")
        logger.info(f"{'=' * 60}")
        
        trainer = ImprovedBalancedAnnotationTypeTrainer(device='cpu')
        
        result = trainer.train_model(
            dataset_file=dataset_file,
            epochs=5,  # Short run for verification
            batch_size=32,
            validation_split=0.2
        )
        
        if not result.get('success'):
            logger.error(f"Run {run_num + 1} failed: {result.get('error')}")
            return False
        
        # Extract key metrics
        run_metrics = {
            'train_accuracy': result.get('train_accuracy'),
            'val_accuracy': result.get('val_accuracy'),
            'best_val_accuracy': result.get('best_val_accuracy'),
            'final_train_loss': result.get('final_train_loss'),
            'final_val_loss': result.get('final_val_loss'),
            'epochs_completed': result.get('epochs_completed', 0)
        }
        
        results.append(run_metrics)
        logger.info(f"Run {run_num + 1} metrics: {run_metrics}")
    
    # Compare results
    logger.info(f"\n{'=' * 60}")
    logger.info("COMPARISON")
    logger.info(f"{'=' * 60}")
    
    all_identical = True
    for i in range(1, len(results)):
        run1 = results[0]
        run2 = results[i]
        
        logger.info(f"\nComparing Run 1 vs Run {i + 1}:")
        
        for key in run1.keys():
            val1 = run1[key]
            val2 = run2[key]
            
            if val1 is None or val2 is None:
                if val1 != val2:
                    logger.warning(f"  {key}: {val1} vs {val2} (one is None)")
                    all_identical = False
                else:
                    logger.info(f"  {key}: Both None (identical)")
            elif isinstance(val1, float) and isinstance(val2, float):
                # Allow small floating point differences
                diff = abs(val1 - val2)
                if diff < 1e-6:
                    logger.info(f"  {key}: {val1:.6f} vs {val2:.6f} (identical, diff={diff:.2e})")
                else:
                    logger.warning(f"  {key}: {val1:.6f} vs {val2:.6f} (DIFFERENT, diff={diff:.2e})")
                    all_identical = False
            else:
                if val1 == val2:
                    logger.info(f"  {key}: {val1} vs {val2} (identical)")
                else:
                    logger.warning(f"  {key}: {val1} vs {val2} (DIFFERENT)")
                    all_identical = False
    
    if all_identical:
        logger.info("\n✅ VERIFICATION PASSED: All runs produced identical results")
        logger.info("Random seed fixes are working correctly!")
    else:
        logger.warning("\n⚠️  VERIFICATION FAILED: Runs produced different results")
        logger.warning("Random seed fixes may need adjustment")
    
    return all_identical


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify random seed fixes produce deterministic results'
    )
    parser.add_argument(
        '--dataset_file',
        type=str,
        required=True,
        help='Path to balanced dataset JSON file'
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=2,
        help='Number of training runs to compare (default: 2)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_file):
        logger.error(f"Dataset file not found: {args.dataset_file}")
        return 1
    
    success = verify_deterministic_training(args.dataset_file, args.num_runs)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

