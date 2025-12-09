#!/usr/bin/env python3
"""
Train All Checkers - Comprehensive Training Script

This script trains all models for all checkers:
- Lower Bound Checker: 21 models (7 base models × 3 annotation types)
- SQL Quotes Checker: 14 models (7 base models × 2 annotation types)
- Signature String Checker: 21 models (7 base models × 3 annotation types)

It checks for existing models and only trains missing ones.
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
MODELS_DIR = GEN_DATA_ROOT / 'models_annotation_types'
LOG_DIR = GEN_DATA_ROOT / 'training_logs'

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True)

def check_models_exist(checker_name: str) -> Tuple[int, int]:
    """Check how many models exist for a checker"""
    from checker_evaluation_config import get_checker_config, build_model_name
    
    config = get_checker_config(checker_name)
    if not config:
        return 0, 0
    
    annotation_types = config.get('annotation_types', [])
    base_models = ['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']
    expected = len(annotation_types) * len(base_models)
    
    found = 0
    all_models = list(MODELS_DIR.glob('*.pth')) + list(MODELS_DIR.glob('*.pkl'))
    
    for ann_type in annotation_types:
        for base_model in base_models:
            model_name = build_model_name(checker_name, ann_type, base_model)
            # Check if model file exists (flexible matching)
            found_model = any(
                model_name.replace('_', '').lower() in f.name.replace('_', '').lower().replace('.pth', '').replace('.pkl', '')
                for f in all_models
            )
            if found_model:
                found += 1
    
    return found, expected

def train_lower_bound_models(episodes: int = 100, background: bool = True) -> bool:
    """Train Lower Bound Checker models"""
    logger.info("=" * 80)
    logger.info("Training Lower Bound Checker Models")
    logger.info("=" * 80)
    
    found, expected = check_models_exist('lower_bound')
    logger.info(f"Found {found}/{expected} Lower Bound models")
    
    if found >= expected * 0.8:  # 80% threshold
        logger.info("✅ Lower Bound models already exist (≥80%), skipping training")
        return True
    
    script = GEN_DATA_ROOT / 'train_all_21_models.py'
    if not script.exists():
        logger.error(f"Training script not found: {script}")
        return False
    
    log_file = LOG_DIR / 'train_lower_bound.log'
    
    cmd = ['python3', str(script), '--episodes', str(episodes)]
    
    if background:
        logger.info(f"Starting Lower Bound training in background (log: {log_file})")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(GEN_DATA_ROOT)
            )
        logger.info(f"✅ Lower Bound training started (PID: {process.pid})")
        logger.info(f"Monitor progress: tail -f {log_file}")
        return True
    else:
        logger.info("Starting Lower Bound training (foreground)...")
        result = subprocess.run(cmd, cwd=str(GEN_DATA_ROOT))
        return result.returncode == 0

def train_sql_quotes_models(episodes: int = 100, background: bool = True) -> bool:
    """Train SQL Quotes Checker models"""
    logger.info("=" * 80)
    logger.info("Training SQL Quotes Checker Models")
    logger.info("=" * 80)
    
    # Check if test suite exists
    test_suite = Path('/home/ubuntu/checker-framework/checker/tests/quotes')
    if not test_suite.exists():
        logger.warning("⚠️ SQL Quotes test suite not found, cannot train models")
        return False
    
    found, expected = check_models_exist('sql_quotes')
    logger.info(f"Found {found}/{expected} SQL Quotes models")
    
    if found >= expected * 0.8:
        logger.info("✅ SQL Quotes models already exist (≥80%), skipping training")
        return True
    
    script = GEN_DATA_ROOT / 'train_sql_quotes_models.py'
    if not script.exists():
        logger.error(f"Training script not found: {script}")
        return False
    
    log_file = LOG_DIR / 'train_sql_quotes.log'
    
    cmd = ['python3', str(script), '--episodes', str(episodes)]
    
    if background:
        logger.info(f"Starting SQL Quotes training in background (log: {log_file})")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(GEN_DATA_ROOT)
            )
        logger.info(f"✅ SQL Quotes training started (PID: {process.pid})")
        logger.info(f"Monitor progress: tail -f {log_file}")
        return True
    else:
        logger.info("Starting SQL Quotes training (foreground)...")
        result = subprocess.run(cmd, cwd=str(GEN_DATA_ROOT))
        return result.returncode == 0

def train_signature_string_models(episodes: int = 100, background: bool = True) -> bool:
    """Train Signature String Checker models"""
    logger.info("=" * 80)
    logger.info("Training Signature String Checker Models")
    logger.info("=" * 80)
    
    # Check if test suite exists
    test_suite = Path('/home/ubuntu/checker-framework/checker/tests/signature')
    if not test_suite.exists():
        logger.warning("⚠️ Signature String test suite not found, cannot train models")
        return False
    
    found, expected = check_models_exist('signature_string')
    logger.info(f"Found {found}/{expected} Signature String models")
    
    if found >= expected * 0.8:
        logger.info("✅ Signature String models already exist (≥80%), skipping training")
        return True
    
    script = GEN_DATA_ROOT / 'train_signature_string_models.py'
    if not script.exists():
        logger.error(f"Training script not found: {script}")
        return False
    
    log_file = LOG_DIR / 'train_signature_string.log'
    
    cmd = ['python3', str(script), '--episodes', str(episodes)]
    
    if background:
        logger.info(f"Starting Signature String training in background (log: {log_file})")
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(GEN_DATA_ROOT)
            )
        logger.info(f"✅ Signature String training started (PID: {process.pid})")
        logger.info(f"Monitor progress: tail -f {log_file}")
        return True
    else:
        logger.info("Starting Signature String training (foreground)...")
        result = subprocess.run(cmd, cwd=str(GEN_DATA_ROOT))
        return result.returncode == 0

def generate_training_report() -> Dict:
    """Generate a report of training status"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'checkers': {}
    }
    
    for checker_name in ['lower_bound', 'sql_quotes', 'signature_string']:
        found, expected = check_models_exist(checker_name)
        report['checkers'][checker_name] = {
            'found': found,
            'expected': expected,
            'percentage': (found / expected * 100) if expected > 0 else 0,
            'status': 'complete' if found >= expected * 0.8 else 'incomplete'
        }
    
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train all checker models')
    parser.add_argument('--checker', choices=['lower_bound', 'sql_quotes', 'signature_string', 'all'],
                       default='all', help='Checker to train')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--foreground', action='store_true', help='Run in foreground (not background)')
    parser.add_argument('--report-only', action='store_true', help='Only generate report, do not train')
    parser.add_argument('--generate-warnings', action='store_true', help='Generate warning files from test suites before training')
    
    args = parser.parse_args()
    
    background = not args.foreground
    
    # Generate initial report
    logger.info("=" * 80)
    logger.info("Training Status Report")
    logger.info("=" * 80)
    report = generate_training_report()
    for checker_name, status in report['checkers'].items():
        logger.info(f"{checker_name}: {status['found']}/{status['expected']} models ({status['percentage']:.1f}%) - {status['status']}")
    
    if args.report_only:
        # Save report
        report_file = LOG_DIR / 'training_status.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to: {report_file}")
        return 0
    
    # Generate warning files if requested
    if args.generate_warnings:
        logger.info("=" * 80)
        logger.info("Generating Warning Files from Test Suites")
        logger.info("=" * 80)
        try:
            sys.path.insert(0, str(GEN_DATA_ROOT))
            from generate_checker_warning_files import generate_all_warning_files
            results = generate_all_warning_files()
            logger.info("Warning file generation completed")
            logger.info("")
        except Exception as e:
            logger.error(f"Failed to generate warning files: {e}")
            logger.warning("Continuing with training anyway (using existing warning files if available)")
            logger.info("")
    
    # Train models
    results = {}
    
    if args.checker in ['lower_bound', 'all']:
        results['lower_bound'] = train_lower_bound_models(episodes=args.episodes, background=background)
    
    if args.checker in ['sql_quotes', 'all']:
        results['sql_quotes'] = train_sql_quotes_models(episodes=args.episodes, background=background)
    
    if args.checker in ['signature_string', 'all']:
        results['signature_string'] = train_signature_string_models(episodes=args.episodes, background=background)
    
    # Final report
    logger.info("=" * 80)
    logger.info("Training Summary")
    logger.info("=" * 80)
    for checker_name, success in results.items():
        status = "✅ Started" if success else "❌ Failed"
        logger.info(f"{checker_name}: {status}")
    
    if background:
        logger.info("")
        logger.info("Training is running in background. Monitor progress with:")
        logger.info(f"  tail -f {LOG_DIR}/train_*.log")
        logger.info("")
        logger.info("Check training status with:")
        logger.info("  python3 train_all_checkers.py --report-only")
    
    return 0

if __name__ == '__main__':
    exit(main())

