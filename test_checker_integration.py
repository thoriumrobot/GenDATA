#!/usr/bin/env python3
"""
Checker Integration Test Suite

This script performs integration tests for checker execution and evaluation workflows.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'

def test_checker_execution() -> Dict[str, Any]:
    """Test running each checker on a known project"""
    logger.info("=" * 80)
    logger.info("Testing Checker Execution Pipeline")
    logger.info("=" * 80)
    
    results = {}
    
    # Test Lower Bound Checker
    logger.info("\nTesting Lower Bound Checker execution...")
    try:
        from checker_framework_runner import CheckerFrameworkRunner
        
        project_path = CASE_STUDIES_DIR / 'guava'
        if not project_path.exists():
            logger.warning("Guava project not found, skipping Lower Bound Checker test")
            results['lower_bound'] = {'status': 'skipped', 'reason': 'project_not_found'}
        else:
            warnings_file = project_path / 'guava_lower_bound_test_warnings.out'
            checker_cp = ':'.join([
                '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker-qual.jar',
                '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker.jar'
            ])
            
            runner = CheckerFrameworkRunner(checker_cp=checker_cp, checker_name='lower_bound')
            success = runner.run_checker_on_project(
                project_root=str(project_path),
                output_file=str(warnings_file),
                max_files=10  # Limit for testing
            )
            
            if success and warnings_file.exists():
                warning_count = runner.parse_warnings_file(str(warnings_file)).get('total_warnings', 0)
                results['lower_bound'] = {
                    'status': 'success',
                    'warnings_file': str(warnings_file),
                    'warning_count': warning_count
                }
                logger.info(f"✅ Lower Bound Checker: Generated {warning_count} warnings")
            else:
                results['lower_bound'] = {'status': 'failed', 'reason': 'warnings_file_not_created'}
                logger.warning("⚠️ Lower Bound Checker: Failed to generate warnings file")
    except Exception as e:
        results['lower_bound'] = {'status': 'error', 'error': str(e)}
        logger.error(f"❌ Lower Bound Checker test failed: {e}")
    
    # Test SQL Quotes Checker
    logger.info("\nTesting SQL Quotes Checker execution...")
    try:
        project_path = CASE_STUDIES_DIR / 'guava'
        if not project_path.exists():
            logger.warning("Guava project not found, skipping SQL Quotes Checker test")
            results['sql_quotes'] = {'status': 'skipped', 'reason': 'project_not_found'}
        else:
            warnings_file = project_path / 'guava_sql_quotes_test_warnings.out'
            checker_cp = ':'.join([
                '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker-qual.jar',
                '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker.jar'
            ])
            
            runner = CheckerFrameworkRunner(checker_cp=checker_cp, checker_name='sql_quotes')
            success = runner.run_checker_on_project(
                project_root=str(project_path),
                output_file=str(warnings_file),
                max_files=10  # Limit for testing
            )
            
            if success and warnings_file.exists():
                warning_count = runner.parse_warnings_file(str(warnings_file)).get('total_warnings', 0)
                results['sql_quotes'] = {
                    'status': 'success',
                    'warnings_file': str(warnings_file),
                    'warning_count': warning_count
                }
                logger.info(f"✅ SQL Quotes Checker: Generated {warning_count} warnings")
            else:
                results['sql_quotes'] = {'status': 'failed', 'reason': 'warnings_file_not_created'}
                logger.warning("⚠️ SQL Quotes Checker: Failed to generate warnings file")
    except Exception as e:
        results['sql_quotes'] = {'status': 'error', 'error': str(e)}
        logger.error(f"❌ SQL Quotes Checker test failed: {e}")
    
    # Test Signature String Checker
    logger.info("\nTesting Signature String Checker execution...")
    try:
        project_path = CASE_STUDIES_DIR / 'guava'
        if not project_path.exists():
            logger.warning("Guava project not found, skipping Signature String Checker test")
            results['signature_string'] = {'status': 'skipped', 'reason': 'project_not_found'}
        else:
            warnings_file = project_path / 'guava_signature_string_test_warnings.out'
            checker_cp = ':'.join([
                '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker-qual.jar',
                '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker.jar'
            ])
            
            runner = CheckerFrameworkRunner(checker_cp=checker_cp, checker_name='signature_string')
            success = runner.run_checker_on_project(
                project_root=str(project_path),
                output_file=str(warnings_file),
                max_files=10  # Limit for testing
            )
            
            if success and warnings_file.exists():
                warning_count = runner.parse_warnings_file(str(warnings_file)).get('total_warnings', 0)
                results['signature_string'] = {
                    'status': 'success',
                    'warnings_file': str(warnings_file),
                    'warning_count': warning_count
                }
                logger.info(f"✅ Signature String Checker: Generated {warning_count} warnings")
            else:
                results['signature_string'] = {'status': 'failed', 'reason': 'warnings_file_not_created'}
                logger.warning("⚠️ Signature String Checker: Failed to generate warnings file")
    except Exception as e:
        results['signature_string'] = {'status': 'error', 'error': str(e)}
        logger.error(f"❌ Signature String Checker test failed: {e}")
    
    return results

def test_project_identification() -> Dict[str, Any]:
    """Test project identification script"""
    logger.info("=" * 80)
    logger.info("Testing Project Identification")
    logger.info("=" * 80)
    
    results = {}
    
    try:
        import subprocess
        result = subprocess.run(
            ['python3', str(GEN_DATA_ROOT / 'identify_checker_projects.py')],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            results['status'] = 'success'
            results['output'] = result.stdout
            logger.info("✅ Project identification script executed successfully")
        else:
            results['status'] = 'failed'
            results['error'] = result.stderr
            logger.warning(f"⚠️ Project identification script failed: {result.stderr}")
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        logger.error(f"❌ Project identification test failed: {e}")
    
    return results

def test_project_preparation() -> Dict[str, Any]:
    """Test project preparation script"""
    logger.info("=" * 80)
    logger.info("Testing Project Preparation")
    logger.info("=" * 80)
    
    results = {}
    
    try:
        import subprocess
        result = subprocess.run(
            ['python3', str(GEN_DATA_ROOT / 'prepare_checker_projects.py')],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        
        if result.returncode == 0:
            results['status'] = 'success'
            results['output'] = result.stdout
            logger.info("✅ Project preparation script executed successfully")
        else:
            results['status'] = 'failed'
            results['error'] = result.stderr
            logger.warning(f"⚠️ Project preparation script failed: {result.stderr}")
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        logger.error(f"❌ Project preparation test failed: {e}")
    
    return results

def test_multi_checker_evaluation() -> Dict[str, Any]:
    """Test multi-checker evaluation script"""
    logger.info("=" * 80)
    logger.info("Testing Multi-Checker Evaluation")
    logger.info("=" * 80)
    
    results = {}
    
    try:
        from evaluate_multi_checker import evaluate_checker_on_project
        
        # Test single checker evaluation
        logger.info("Testing single checker evaluation...")
        result = evaluate_checker_on_project('lower_bound', 'guava')
        
        if isinstance(result, dict) and 'status' in result:
            results['single_checker'] = {
                'status': 'success',
                'evaluation_status': result.get('status'),
                'has_warnings': result.get('has_warnings', False)
            }
            logger.info(f"✅ Single checker evaluation: {result.get('status')}")
        else:
            results['single_checker'] = {'status': 'failed', 'reason': 'invalid_result_format'}
            logger.warning("⚠️ Single checker evaluation returned invalid format")
    except Exception as e:
        results['single_checker'] = {'status': 'error', 'error': str(e)}
        logger.error(f"❌ Single checker evaluation test failed: {e}")
    
    return results

def main():
    """Main integration test function"""
    logger.info("=" * 80)
    logger.info("Checker Integration Test Suite")
    logger.info("=" * 80)
    
    all_results = {}
    
    # Run integration tests
    all_results['checker_execution'] = test_checker_execution()
    all_results['project_identification'] = test_project_identification()
    all_results['project_preparation'] = test_project_preparation()
    all_results['multi_checker_evaluation'] = test_multi_checker_evaluation()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Integration Test Summary")
    logger.info("=" * 80)
    
    for test_name, result in all_results.items():
        if isinstance(result, dict):
            # Handle nested results (e.g., checker_execution has nested checker results)
            if 'status' in result:
                status = result.get('status', 'unknown')
                logger.info(f"{test_name}: {status}")
            elif test_name == 'checker_execution':
                # Check if all checkers succeeded
                all_success = all(
                    r.get('status') == 'success' or r.get('status') == 'skipped'
                    for r in result.values() if isinstance(r, dict)
                )
                status = 'success' if all_success else 'partial'
                logger.info(f"{test_name}: {status}")
            elif test_name == 'multi_checker_evaluation':
                # Extract status from nested result
                nested_status = result.get('single_checker', {}).get('status', 'unknown')
                logger.info(f"{test_name}: {nested_status}")
            else:
                logger.info(f"{test_name}: {type(result)}")
        else:
            logger.info(f"{test_name}: {type(result)}")
    
    return 0

if __name__ == '__main__':
    exit(main())

