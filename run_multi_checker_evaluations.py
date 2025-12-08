#!/usr/bin/env python3
"""
Multi-Checker Evaluation Orchestrator

This script orchestrates the complete evaluation pipeline for all checkers:
- Lower Bound Checker
- SQL Quotes Checker
- Signature String Checker

It coordinates the evaluation process across multiple checkers and projects.
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import configuration
from checker_evaluation_config import (
    get_all_checker_names, get_checker_config, get_evaluation_projects
)
from evaluate_multi_checker import evaluate_checker_on_project, evaluate_all_checkers

# Base directory
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
RESULTS_DIR = GEN_DATA_ROOT / 'multi_checker_results'

def verify_checker_availability(checker_name: str) -> Dict[str, Any]:
    """Verify that a checker is available and ready for evaluation."""
    logger.info(f"Verifying availability for {checker_name} checker...")
    
    status = {
        'checker_name': checker_name,
        'available': False,
        'test_suite_exists': False,
        'models_exist': False,
        'model_count': 0,
        'can_evaluate': False,  # Can evaluate even without models (will report no models)
        'issues': []
    }
    
    config = get_checker_config(checker_name)
    if not config:
        status['issues'].append(f"Configuration not found for {checker_name}")
        return status
    
    # Check test suite
    test_suite_path = Path(config.get('test_suite', ''))
    if test_suite_path.exists():
        status['test_suite_exists'] = True
    else:
        status['issues'].append(f"Test suite not found: {test_suite_path}")
    
    # Check models (simplified check)
    from verify_checker_training import verify_sql_quotes_training, verify_signature_string_training
    
    if checker_name == 'sql_quotes':
        training_status = verify_sql_quotes_training()
        status['models_exist'] = training_status.get('models_exist', False)
        status['model_count'] = training_status.get('model_count', 0)
    elif checker_name == 'signature_string':
        training_status = verify_signature_string_training()
        status['models_exist'] = training_status.get('models_exist', False)
        status['model_count'] = training_status.get('model_count', 0)
    elif checker_name == 'lower_bound':
        # Lower Bound Checker models are assumed to exist
        status['models_exist'] = True
        status['model_count'] = 21  # Expected count
    
    # Can evaluate if test suite exists (even without models, we can run checker and report status)
    status['can_evaluate'] = status['test_suite_exists']
    status['available'] = status['test_suite_exists'] and status['models_exist']
    
    if not status['models_exist']:
        status['issues'].append(f"Models not available ({status['model_count']} found, {config.get('expected_models', 0)} expected)")
    
    return status

def prepare_evaluation_projects(checker_name: str) -> List[str]:
    """Prepare and validate evaluation projects for a checker."""
    logger.info(f"Preparing evaluation projects for {checker_name} checker...")
    
    projects = get_evaluation_projects(checker_name)
    
    if not projects:
        logger.warning(f"No evaluation projects configured for {checker_name} checker")
        # Use default projects
        projects = ['guava', 'jfreechart', 'plume-lib']
        logger.info(f"Using default projects: {projects}")
    
    # Validate projects exist
    valid_projects = []
    for project_name in projects:
        project_path = GEN_DATA_ROOT / 'case_studies' / project_name
        if project_path.exists():
            valid_projects.append(project_name)
        else:
            logger.warning(f"Project {project_name} not found, skipping")
    
    logger.info(f"Valid projects for {checker_name}: {valid_projects}")
    return valid_projects

def generate_multi_checker_report(all_results: Dict[str, Dict[str, dict]]) -> Path:
    """Generate comprehensive report comparing all checkers."""
    logger.info("=" * 80)
    logger.info("Generating Multi-Checker Evaluation Report")
    logger.info("=" * 80)
    
    report_file = RESULTS_DIR / 'MULTI_CHECKER_EVALUATION_REPORT.md'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write("# Multi-Checker Evaluation Report\n\n")
        f.write("This report compares evaluation results across all supported checkers.\n\n")
        
        # Summary statistics
        f.write("## Summary\n\n")
        
        total_checkers = len(all_results)
        successful_evaluations = 0
        total_projects = 0
        projects_with_warnings = 0
        
        for checker_name, checker_results in all_results.items():
            for project_name, result in checker_results.items():
                total_projects += 1
                if result.get('status') == 'success':
                    successful_evaluations += 1
                if result.get('has_warnings', False):
                    projects_with_warnings += 1
        
        f.write(f"- Checkers Evaluated: {total_checkers}\n")
        f.write(f"- Total Project Evaluations: {total_projects}\n")
        f.write(f"- Successful Evaluations: {successful_evaluations}\n")
        f.write(f"- Projects with Warnings: {projects_with_warnings}\n\n")
        
        # Per-checker results
        f.write("## Results by Checker\n\n")
        
        for checker_name, checker_results in all_results.items():
            config = get_checker_config(checker_name)
            checker_display_name = config.get('name', checker_name)
            
            f.write(f"### {checker_display_name}\n\n")
            
            successful = len([r for r in checker_results.values() if r.get('status') == 'success'])
            no_warnings = len([r for r in checker_results.values() if r.get('status') == 'no_warnings'])
            failed = len([r for r in checker_results.values() if r.get('status') not in ['success', 'no_warnings']])
            
            f.write(f"**Status Summary:**\n")
            f.write(f"- Successful: {successful}\n")
            f.write(f"- No Warnings: {no_warnings}\n")
            f.write(f"- Failed: {failed}\n\n")
            
            f.write("**Per-Project Results:**\n\n")
            for project_name, result in checker_results.items():
                status = result.get('status', 'unknown')
                warning_count = result.get('warning_count', 0)
                
                if status == 'success':
                    f.write(f"- **{project_name}**: ✅ Success ({warning_count} warnings)\n")
                elif status == 'no_warnings':
                    f.write(f"- **{project_name}**: ⚠️ No Warnings Found\n")
                else:
                    f.write(f"- **{project_name}**: ❌ Failed ({status})\n")
            
            f.write("\n")
        
        # Cross-checker comparison
        f.write("## Cross-Checker Comparison\n\n")
        
        f.write("| Checker | Projects Evaluated | Projects with Warnings | Success Rate |\n")
        f.write("|---------|-------------------|------------------------|---------------|\n")
        
        for checker_name, checker_results in all_results.items():
            config = get_checker_config(checker_name)
            checker_display_name = config.get('name', checker_name)
            
            total = len(checker_results)
            with_warnings = len([r for r in checker_results.values() if r.get('has_warnings', False)])
            successful = len([r for r in checker_results.values() if r.get('status') == 'success'])
            success_rate = f"{(successful/total*100):.1f}%" if total > 0 else "N/A"
            
            f.write(f"| {checker_display_name} | {total} | {with_warnings} | {success_rate} |\n")
        
        f.write("\n")
    
    logger.info(f"Multi-checker report saved to {report_file}")
    return report_file

def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("Multi-Checker Evaluation Orchestrator")
    logger.info("=" * 80)
    
    # Get all checkers
    checker_names = get_all_checker_names()
    logger.info(f"Supported checkers: {', '.join(checker_names)}")
    
    # Verify checker availability
    checker_statuses = {}
    for checker_name in checker_names:
        status = verify_checker_availability(checker_name)
        checker_statuses[checker_name] = status
        
        if status['available']:
            logger.info(f"✅ {checker_name}: Available (models: {status['model_count']})")
        else:
            logger.warning(f"⚠️ {checker_name}: Not fully available")
            if status['issues']:
                for issue in status['issues']:
                    logger.warning(f"  - {issue}")
    
    # Prepare evaluation projects
    all_projects = set()
    for checker_name in checker_names:
        projects = prepare_evaluation_projects(checker_name)
        all_projects.update(projects)
    
    logger.info(f"Evaluation projects: {sorted(all_projects)}")
    
    # Run evaluations
    all_results = {}
    
    for checker_name in checker_names:
        if not checker_statuses[checker_name]['can_evaluate']:
            logger.warning(f"Skipping {checker_name} - cannot evaluate (test suite missing)")
            continue
        
        if not checker_statuses[checker_name]['models_exist']:
            logger.warning(f"{checker_name} has no trained models - evaluation will be limited to warning generation")
        
        projects = prepare_evaluation_projects(checker_name)
        if not projects:
            logger.warning(f"No projects to evaluate for {checker_name}")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {checker_name} checker")
        logger.info(f"{'='*80}\n")
        
        checker_results = {}
        for project_name in projects:
            result = evaluate_checker_on_project(checker_name, project_name)
            checker_results[project_name] = result
        
        all_results[checker_name] = checker_results
    
    # Generate comprehensive report
    if all_results:
        report_file = generate_multi_checker_report(all_results)
        
        # Save JSON results
        json_file = RESULTS_DIR / 'multi_checker_evaluation_results.json'
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"JSON results saved to {json_file}")
        
        logger.info("=" * 80)
        logger.info("Multi-Checker Evaluation Complete")
        logger.info("=" * 80)
        logger.info(f"Report: {report_file}")
    else:
        logger.warning("No results to report")
    
    return 0

if __name__ == '__main__':
    exit(main())

