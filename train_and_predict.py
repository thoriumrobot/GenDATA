#!/usr/bin/env python3
"""
Train Models and Generate Predictions

This script trains models for checkers that need training and then generates predictions.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
MODELS_DIR = GEN_DATA_ROOT / 'models_annotation_types'

def check_models_exist(checker_name: str, expected_count: int) -> bool:
    """Check if models exist for a checker"""
    if not MODELS_DIR.exists():
        return False
    
    from checker_evaluation_config import get_checker_config, build_model_name
    
    config = get_checker_config(checker_name)
    if not config:
        return False
    
    found_count = 0
    all_models = list(MODELS_DIR.glob('*.pth')) + list(MODELS_DIR.glob('*.pkl'))
    
    for ann_type in config.get('annotation_types', []):
        for base_model in config.get('base_models', []):
            model_name = build_model_name(checker_name, ann_type, base_model)
            # Check if model name matches any existing model file
            # Model files can be: {model_name}_model.pth or variations
            found = any(
                model_name.replace('_', '') in f.name.replace('_model.pth', '').replace('_model.pkl', '').replace('_', '')
                or f.name.startswith(model_name.split('_')[0])  # Match annotation prefix
                for f in all_models
            )
            if found:
                found_count += 1
    
    logger.info(f"Found {found_count}/{expected_count} models for {checker_name}")
    return found_count >= max(1, expected_count * 0.3)  # At least 30% or 1 model

def train_signature_string_models(episodes: int = 50):
    """Train Signature String models"""
    logger.info("=" * 80)
    logger.info("Training Signature String Checker Models")
    logger.info("=" * 80)
    
    test_suite = Path('/home/ubuntu/checker-framework/checker/tests/signature')
    if not test_suite.exists():
        logger.error("Signature String test suite not found")
        return False
    
    # Check if models already exist
    if check_models_exist('signature_string', 21):
        logger.info("Signature String models already exist, skipping training")
        return True
    
    # Run training script
    cmd = ['python3', str(GEN_DATA_ROOT / 'train_signature_string_models.py'),
           '--episodes', str(episodes)]
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("✅ Signature String models trained successfully")
        return True
    else:
        logger.error(f"❌ Training failed: {result.stderr}")
        return False

def generate_predictions(checker_name: str, project_name: str):
    """Generate predictions for a checker on a project"""
    logger.info("=" * 80)
    logger.info(f"Generating Predictions: {checker_name} on {project_name}")
    logger.info("=" * 80)
    
    from evaluate_multi_checker import evaluate_checker_on_project
    
    result = evaluate_checker_on_project(checker_name, project_name)
    
    if isinstance(result, dict):
        status = result.get('status')
        if status == 'success':
            logger.info(f"✅ Predictions generated successfully")
            metrics_file = result.get('metrics_file')
            if metrics_file:
                logger.info(f"Metrics saved to: {metrics_file}")
            return True
        elif status == 'no_warnings':
            logger.warning(f"⚠️ Project has no warnings - cannot generate predictions")
            return False
        else:
            logger.error(f"❌ Evaluation failed: {status}")
            return False
    else:
        logger.error("❌ Invalid evaluation result")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train models and generate predictions')
    parser.add_argument('--checker', choices=['lower_bound', 'sql_quotes', 'signature_string', 'all'],
                       default='all', help='Checker to train/evaluate')
    parser.add_argument('--project', help='Project to evaluate on (default: guava)')
    parser.add_argument('--episodes', type=int, default=50, help='Training episodes')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, only generate predictions')
    
    args = parser.parse_args()
    
    project = args.project or 'guava'
    
    # Train models
    if not args.skip_training:
        if args.checker in ['signature_string', 'all']:
            train_signature_string_models(episodes=args.episodes)
        
        if args.checker in ['sql_quotes', 'all']:
            logger.warning("SQL Quotes Checker: Test suite not found, cannot train models")
    
    # Generate predictions
    checkers_to_evaluate = []
    if args.checker == 'all':
        checkers_to_evaluate = ['lower_bound', 'signature_string']  # Skip SQL Quotes (no models)
    else:
        checkers_to_evaluate = [args.checker]
    
    for checker_name in checkers_to_evaluate:
        # Check if models exist
        from checker_evaluation_config import get_checker_config
        config = get_checker_config(checker_name)
        expected_models = config.get('expected_models', 0) if config else 0
        
        if not check_models_exist(checker_name, expected_models):
            logger.warning(f"Skipping {checker_name}: Models not available")
            continue
        
        generate_predictions(checker_name, project)
    
    logger.info("=" * 80)
    logger.info("Training and Prediction Complete")
    logger.info("=" * 80)
    
    return 0

if __name__ == '__main__':
    exit(main())

