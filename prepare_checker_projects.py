#!/usr/bin/env python3
"""
Prepare Projects for Checker Evaluation

This script prepares projects for evaluation with specific checkers by:
1. Running the appropriate checker on each project
2. Generating warnings files
3. Validating that warnings contain actual checker warnings
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from checker_evaluation_config import get_checker_config, get_evaluation_projects
from checker_framework_runner import CheckerFrameworkRunner, count_checker_warnings

GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'

def prepare_project_for_checker(project_name: str, checker_name: str) -> Dict[str, any]:
    """Prepare a project for evaluation with a specific checker."""
    logger.info(f"Preparing {project_name} for {checker_name} checker...")
    
    project_path = CASE_STUDIES_DIR / project_name
    if not project_path.exists():
        return {
            'success': False,
            'error': f'Project not found: {project_name}'
        }
    
    warnings_file = project_path / f'{project_name}_{checker_name}_warnings.out'
    
    # Check if warnings file already exists and is valid
    if warnings_file.exists():
        warning_count = count_checker_warnings(str(warnings_file))
        if warning_count > 0:
            logger.info(f"Warnings file already exists with {warning_count} warnings")
            return {
                'success': True,
                'warnings_file': str(warnings_file),
                'warning_count': warning_count,
                'already_existed': True
            }
    
    # Run checker
    try:
        checker_cp = ':'.join([
            '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker-qual.jar',
            '/home/ubuntu/checker-framework-3.42.0/checker/dist/checker.jar'
        ])
        
        runner = CheckerFrameworkRunner(
            checker_cp=checker_cp,
            checker_name=checker_name
        )
        
        success = runner.run_checker_on_project(
            project_root=str(project_path),
            output_file=str(warnings_file),
            max_files=1000
        )
        
        if success and warnings_file.exists():
            warning_count = count_checker_warnings(str(warnings_file))
            
            if warning_count > 0:
                logger.info(f"✅ Successfully prepared {project_name} for {checker_name} ({warning_count} warnings)")
                return {
                    'success': True,
                    'warnings_file': str(warnings_file),
                    'warning_count': warning_count,
                    'already_existed': False
                }
            else:
                logger.info(f"⚠️ {project_name} has no {checker_name} warnings (may be well-annotated)")
                return {
                    'success': True,
                    'warnings_file': str(warnings_file),
                    'warning_count': 0,
                    'already_existed': False,
                    'no_warnings': True
                }
        else:
            return {
                'success': False,
                'error': 'Failed to generate warnings file'
            }
    except Exception as e:
        logger.error(f"Error preparing {project_name} for {checker_name}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def prepare_all_checkers() -> Dict[str, Dict[str, Dict[str, any]]]:
    """Prepare all projects for all checkers."""
    logger.info("=" * 80)
    logger.info("Preparing Projects for All Checkers")
    logger.info("=" * 80)
    
    from checker_evaluation_config import get_all_checker_names
    
    results = {}
    
    for checker_name in get_all_checker_names():
        logger.info(f"\n{'='*80}")
        logger.info(f"Preparing projects for {checker_name} checker")
        logger.info(f"{'='*80}\n")
        
        projects = get_evaluation_projects(checker_name)
        if not projects:
            logger.warning(f"No projects configured for {checker_name}")
            continue
        
        results[checker_name] = {}
        
        for project_name in projects:
            result = prepare_project_for_checker(project_name, checker_name)
            results[checker_name][project_name] = result
    
    return results

def main():
    """Main function"""
    results = prepare_all_checkers()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Preparation Summary")
    logger.info("=" * 80)
    
    for checker_name, checker_results in results.items():
        logger.info(f"\n{checker_name.upper()} Checker:")
        successful = len([r for r in checker_results.values() if r.get('success')])
        with_warnings = len([r for r in checker_results.values() if r.get('warning_count', 0) > 0])
        logger.info(f"  Projects prepared: {successful}/{len(checker_results)}")
        logger.info(f"  Projects with warnings: {with_warnings}")
    
    # Save results
    results_file = GEN_DATA_ROOT / 'checker_project_preparation.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    return 0

if __name__ == '__main__':
    exit(main())

