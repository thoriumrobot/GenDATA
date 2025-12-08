#!/usr/bin/env python3
"""
Verify Checker Training Status

This script verifies whether models exist and training data is available
for SQL Quotes and Signature String checkers.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = Path('/home/ubuntu/GenDATA/models_annotation_types')
CHECKER_FRAMEWORK_ROOT = Path('/home/ubuntu/checker-framework-3.42.0')

def verify_sql_quotes_training() -> Dict[str, any]:
    """Verify SQL Quotes Checker training status"""
    logger.info("=" * 80)
    logger.info("Verifying SQL Quotes Checker Training Status")
    logger.info("=" * 80)
    
    status = {
        'checker_name': 'sql_quotes',
        'test_suite_exists': False,
        'test_suite_path': None,
        'models_exist': False,
        'model_count': 0,
        'expected_models': 14,  # 7 base models × 2 annotation types
        'models_found': [],
        'models_missing': [],
        'annotation_types': ['sqlevenquotes', 'sqloddquotes'],
        'base_models': ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n']
    }
    
    # Check test suite
    test_suite_paths = [
        CHECKER_FRAMEWORK_ROOT / 'checker' / 'tests' / 'quotes',
        Path('/home/ubuntu/checker-framework/checker/tests/quotes'),
    ]
    
    for path in test_suite_paths:
        if path.exists():
            status['test_suite_exists'] = True
            status['test_suite_path'] = str(path)
            logger.info(f"✅ Test suite found: {path}")
            break
    
    if not status['test_suite_exists']:
        logger.warning("⚠️ SQL Quotes test suite not found")
    
    # Check for models
    if MODELS_DIR.exists():
        all_files = list(MODELS_DIR.rglob('*'))
        
        # Look for SQL Quotes models
        for ann_type in status['annotation_types']:
            for base_model in status['base_models']:
                model_patterns = [
                    f"{ann_type}_{base_model}",
                    f"{base_model}_{ann_type}",
                    f"sql_quotes_{ann_type}_{base_model}",
                    f"{ann_type.replace('sql', 'sqleven')}_{base_model}",
                    f"{ann_type.replace('sql', 'sqlodd')}_{base_model}"
                ]
                
                found = False
                for pattern in model_patterns:
                    matching_files = [f for f in all_files if pattern.lower() in str(f).lower() and (f.suffix in ['.pth', '.pt', '.model'] or f.is_dir())]
                    if matching_files:
                        status['models_found'].append(f"{ann_type}_{base_model}")
                        status['model_count'] += 1
                        found = True
                        break
                
                if not found:
                    status['models_missing'].append(f"{ann_type}_{base_model}")
        
        status['models_exist'] = status['model_count'] > 0
        logger.info(f"Found {status['model_count']}/{status['expected_models']} SQL Quotes models")
        
        if status['models_found']:
            logger.info(f"✅ Models found: {', '.join(status['models_found'][:5])}{'...' if len(status['models_found']) > 5 else ''}")
        if status['models_missing']:
            logger.warning(f"⚠️ Missing models: {len(status['models_missing'])} models")
    else:
        logger.warning(f"⚠️ Models directory not found: {MODELS_DIR}")
    
    return status

def verify_signature_string_training() -> Dict[str, any]:
    """Verify Signature String Checker training status"""
    logger.info("=" * 80)
    logger.info("Verifying Signature String Checker Training Status")
    logger.info("=" * 80)
    
    status = {
        'checker_name': 'signature_string',
        'test_suite_exists': False,
        'test_suite_path': None,
        'models_exist': False,
        'model_count': 0,
        'expected_models': 21,  # 7 base models × 3 annotation types
        'models_found': [],
        'models_missing': [],
        'annotation_types': ['fullyqualifiedname', 'binaryname', 'fielddescriptor'],
        'base_models': ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n']
    }
    
    # Check test suite
    test_suite_paths = [
        CHECKER_FRAMEWORK_ROOT / 'checker' / 'tests' / 'signature',
        Path('/home/ubuntu/checker-framework/checker/tests/signature'),
    ]
    
    for path in test_suite_paths:
        if path.exists():
            status['test_suite_exists'] = True
            status['test_suite_path'] = str(path)
            logger.info(f"✅ Test suite found: {path}")
            break
    
    if not status['test_suite_exists']:
        logger.warning("⚠️ Signature String test suite not found")
    
    # Check for models
    if MODELS_DIR.exists():
        all_files = list(MODELS_DIR.rglob('*'))
        
        # Look for Signature String models
        for ann_type in status['annotation_types']:
            for base_model in status['base_models']:
                model_patterns = [
                    f"{ann_type}_{base_model}",
                    f"{base_model}_{ann_type}",
                    f"signature_string_{ann_type}_{base_model}",
                    f"signature_{ann_type}_{base_model}"
                ]
                
                found = False
                for pattern in model_patterns:
                    matching_files = [f for f in all_files if pattern.lower() in str(f).lower() and (f.suffix in ['.pth', '.pt', '.model'] or f.is_dir())]
                    if matching_files:
                        status['models_found'].append(f"{ann_type}_{base_model}")
                        status['model_count'] += 1
                        found = True
                        break
                
                if not found:
                    status['models_missing'].append(f"{ann_type}_{base_model}")
        
        status['models_exist'] = status['model_count'] > 0
        logger.info(f"Found {status['model_count']}/{status['expected_models']} Signature String models")
        
        if status['models_found']:
            logger.info(f"✅ Models found: {', '.join(status['models_found'][:5])}{'...' if len(status['models_found']) > 5 else ''}")
        if status['models_missing']:
            logger.warning(f"⚠️ Missing models: {len(status['models_missing'])} models")
    else:
        logger.warning(f"⚠️ Models directory not found: {MODELS_DIR}")
    
    return status

def main():
    """Main function"""
    logger.info("Starting checker training verification...")
    
    sql_status = verify_sql_quotes_training()
    signature_status = verify_signature_string_training()
    
    # Summary
    logger.info("=" * 80)
    logger.info("Verification Summary")
    logger.info("=" * 80)
    
    logger.info(f"\nSQL Quotes Checker:")
    logger.info(f"  Test Suite: {'✅ Found' if sql_status['test_suite_exists'] else '❌ Missing'}")
    logger.info(f"  Models: {sql_status['model_count']}/{sql_status['expected_models']} ({'✅ Available' if sql_status['models_exist'] else '❌ Missing'})")
    
    logger.info(f"\nSignature String Checker:")
    logger.info(f"  Test Suite: {'✅ Found' if signature_status['test_suite_exists'] else '❌ Missing'}")
    logger.info(f"  Models: {signature_status['model_count']}/{signature_status['expected_models']} ({'✅ Available' if signature_status['models_exist'] else '❌ Missing'})")
    
    return {
        'sql_quotes': sql_status,
        'signature_string': signature_status
    }

if __name__ == '__main__':
    import json
    results = main()
    print("\n" + "=" * 80)
    print("JSON Results:")
    print(json.dumps(results, indent=2))

