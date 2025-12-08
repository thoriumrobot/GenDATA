#!/usr/bin/env python3
"""
Multi-Checker Evaluation Script

This script extends the evaluation pipeline to support multiple checkers:
- Lower Bound Checker
- SQL Quotes Checker  
- Signature String Checker

It reuses the existing evaluation infrastructure while making it checker-aware.
"""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directory
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'
MODELS_DIR = GEN_DATA_ROOT / 'models_annotation_types'

# Supported checkers
SUPPORTED_CHECKERS = ['lower_bound', 'sql_quotes', 'signature_string']

def get_checker_interface(checker_name: str):
    """Get checker interface instance"""
    try:
        from checker_registry import get_checker
        checker = get_checker(checker_name)
        if checker:
            return checker
        else:
            logger.error(f"Checker '{checker_name}' not found in registry")
            return None
    except Exception as e:
        logger.error(f"Error loading checker '{checker_name}': {e}")
        return None

def validate_project(project_path: Path, project_name: str) -> bool:
    """Validate that project exists and has required structure."""
    if not project_path.exists():
        logger.error(f"Project {project_name} not found at {project_path}")
        return False
    
    # Check for Java files
    java_files = list(project_path.rglob('*.java'))
    if not java_files:
        logger.warning(f"No Java files found in {project_name}")
        return False
    
    logger.info(f"Found {len(java_files)} Java files in {project_name}")
    return True

def run_checker_on_project(project_path: Path, project_name: str, checker_name: str) -> tuple:
    """Run checker on a project. Returns (warnings_file, validation_info)."""
    checker_interface = get_checker_interface(checker_name)
    if not checker_interface:
        return None, {'has_warnings': False, 'warning_count': 0}
    
    warnings_file = project_path / f'{project_name}_{checker_name}_warnings.out'
    
    if warnings_file.exists():
        logger.info(f"Warnings file already exists for {project_name} ({checker_name}): {warnings_file}")
        validation_info = validate_warnings_file(warnings_file, checker_name)
        return warnings_file, validation_info
    
    logger.info(f"Running {checker_name} checker on {project_name}...")
    
    try:
        from checker_framework_runner import CheckerFrameworkRunner
        
        # Set up checker classpath
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
            logger.info(f"Successfully generated warnings file for {project_name} ({checker_name})")
            validation_info = validate_warnings_file(warnings_file, checker_name)
            
            if validation_info['has_warnings']:
                logger.info(f"Found {validation_info['warning_count']} checker warnings in {project_name}")
            else:
                logger.info(f"No checker warnings found in {project_name} (may have {validation_info.get('compilation_errors', 0)} compilation errors)")
            
            return warnings_file, validation_info
        else:
            logger.warning(f"Failed to generate warnings for {project_name}")
            return None, {'has_warnings': False, 'warning_count': 0}
    except Exception as e:
        logger.error(f"Error running checker on {project_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None, {'has_warnings': False, 'warning_count': 0}

def validate_warnings_file(warnings_file: Path, checker_name: str) -> dict:
    """Validate warnings file and count actual checker warnings."""
    if not warnings_file.exists():
        return {'has_warnings': False, 'warning_count': 0, 'compilation_errors': 0}
    
    try:
        from checker_framework_runner import CheckerFrameworkRunner, count_checker_warnings
        
        # Count checker warnings
        warning_count = count_checker_warnings(str(warnings_file))
        
        # Parse warnings file for more details
        checker_interface = get_checker_interface(checker_name)
        if checker_interface:
            # Use checker-specific parser
            parsed_warnings = checker_interface.parse_warnings(str(warnings_file))
            warning_count = len(parsed_warnings)
        
        # Also use generic parser for compilation errors
        runner = CheckerFrameworkRunner(checker_name=checker_name)
        warnings_info = runner.parse_warnings_file(str(warnings_file))
        
        return {
            'has_warnings': warning_count > 0,
            'warning_count': warning_count,
            'compilation_errors': warnings_info.get('total_compilation_errors', 0),
            'files_with_warnings': len(warnings_info.get('files_with_warnings', []))
        }
    except Exception as e:
        logger.warning(f"Error validating warnings file: {e}")
        return {'has_warnings': False, 'warning_count': 0, 'compilation_errors': 0}

def generate_slices(project_path: Path, warnings_file: Path, project_name: str, checker_name: str, has_warnings: bool) -> bool:
    """Generate slices using Soot slicer."""
    if not has_warnings:
        logger.info(f"Skipping slice generation for {project_name} ({checker_name}) - no checker warnings found")
        return False
    
    slices_dir = project_path / 'slices' / checker_name
    slices_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating slices for {project_name} ({checker_name})...")
    
    try:
        from pipeline import run_slicing
        
        run_slicing(
            project_root=str(project_path),
            warnings_file=str(warnings_file),
            cfwr_root=str(GEN_DATA_ROOT),
            base_slices_dir=str(slices_dir),
            slicer_type='soot'
        )
        
        # Check if slices were generated
        slices_soot_dir = slices_dir / 'slices_soot'
        if slices_soot_dir.exists() and any(slices_soot_dir.iterdir()):
            logger.info(f"Successfully generated slices for {project_name} ({checker_name})")
            return True
        else:
            logger.warning(f"No slices generated for {project_name} ({checker_name})")
            return False
    except Exception as e:
        logger.error(f"Error generating slices for {project_name} ({checker_name}): {e}")
        return False

def generate_cfgs(project_path: Path, project_name: str, checker_name: str) -> bool:
    """Generate CFGs from slices."""
    slices_dir = project_path / 'slices' / checker_name / 'slices_soot'
    cfg_output_dir = project_path / 'cfg_output' / checker_name
    cfg_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not slices_dir.exists() or not any(slices_dir.iterdir()):
        logger.warning(f"Slices directory not found or empty for {project_name} ({checker_name}): {slices_dir}")
        return False
    
    logger.info(f"Generating CFGs for {project_name} ({checker_name})...")
    
    try:
        from pipeline import run_cfg_generation
        
        run_cfg_generation(str(slices_dir), str(cfg_output_dir))
        
        # Check if CFGs were generated
        if cfg_output_dir.exists() and any(cfg_output_dir.iterdir()):
            logger.info(f"Successfully generated CFGs for {project_name} ({checker_name})")
            return True
        else:
            logger.warning(f"No CFGs generated for {project_name} ({checker_name})")
            return False
    except Exception as e:
        logger.error(f"Error generating CFGs for {project_name} ({checker_name}): {e}")
        return False

def get_available_models_for_checker(checker_name: str) -> List[str]:
    """Get list of available models for a specific checker."""
    checker_interface = get_checker_interface(checker_name)
    if not checker_interface:
        return []
    
    annotation_types = checker_interface.get_annotation_types()
    base_models = ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n']
    
    # Build expected model names
    expected_models = []
    for base_model in base_models:
        for ann_type in annotation_types:
            # Remove @ symbol and convert to lowercase
            ann_name = ann_type.replace('@', '').lower()
            model_name = f"{ann_name}_{base_model}"
            expected_models.append(model_name)
    
    # Check which models actually exist
    available_models = []
    if MODELS_DIR.exists():
        for model_name in expected_models:
            # Check for model files
            model_files = list(MODELS_DIR.rglob(f"*{model_name}*"))
            if model_files:
                available_models.append(model_name)
    
    logger.info(f"Found {len(available_models)}/{len(expected_models)} models for {checker_name}")
    return available_models

def generate_predictions(project_path: Path, project_name: str, checker_name: str, model_names: List[str]) -> bool:
    """Generate predictions for all models using CFG files."""
    cfg_output_dir = project_path / 'cfg_output' / checker_name
    
    if not cfg_output_dir.exists() or not any(cfg_output_dir.iterdir()):
        logger.warning(f"No CFGs found for {project_name} ({checker_name})")
        return False
    
    logger.info(f"Generating predictions for {project_name} ({checker_name}) with {len(model_names)} models...")
    
    # Find all Java files that have CFGs
    java_files_with_cfgs = []
    for java_file in project_path.rglob('*.java'):
        java_basename = java_file.stem
        cfg_file = cfg_output_dir / java_basename / 'cfg.json'
        if cfg_file.exists():
            java_files_with_cfgs.append(java_file)
    
    if not java_files_with_cfgs:
        logger.warning(f"No Java files with CFGs found for {project_name} ({checker_name})")
        return False
    
    logger.info(f"Found {len(java_files_with_cfgs)} Java files with CFGs")
    
    # Note: Prediction generation would need to be checker-aware
    # For now, we'll use a simplified approach that works with the existing infrastructure
    # Full implementation would require checker-specific prediction logic
    
    logger.info(f"Prediction generation for {checker_name} checker - implementation pending")
    return False  # Placeholder - full implementation needed

def compute_metrics(project_path: Path, project_name: str, checker_name: str, model_names: List[str]) -> dict:
    """Compute evaluation metrics for all models."""
    logger.info(f"Computing metrics for {project_name} ({checker_name})...")
    
    # Placeholder - full implementation needed
    return {}

def evaluate_checker_on_project(checker_name: str, project_name: str) -> dict:
    """Evaluate a single checker on a single project."""
    logger.info("=" * 80)
    logger.info(f"Evaluating {checker_name} checker on project: {project_name}")
    logger.info("=" * 80)
    
    project_path = CASE_STUDIES_DIR / project_name
    evaluation_status = {
        'checker': checker_name,
        'project': project_name,
        'status': 'failed',
        'steps_completed': [],
        'steps_failed': [],
        'warning_count': 0,
        'has_warnings': False,
        'metrics': None
    }
    
    # Validate project
    if not validate_project(project_path, project_name):
        evaluation_status['steps_failed'].append('validation')
        return evaluation_status
    
    evaluation_status['steps_completed'].append('validation')
    
    # Step 1: Run checker
    warnings_file, validation_info = run_checker_on_project(project_path, project_name, checker_name)
    if not warnings_file:
        evaluation_status['steps_failed'].append('warnings_generation')
        evaluation_status['status'] = 'no_warnings_file'
        return evaluation_status
    
    evaluation_status['steps_completed'].append('warnings_generation')
    evaluation_status['warning_count'] = validation_info.get('warning_count', 0)
    evaluation_status['has_warnings'] = validation_info.get('has_warnings', False)
    
    # If no warnings, report this gracefully
    if not evaluation_status['has_warnings']:
        logger.info(f"Project {project_name} has no {checker_name} checker warnings.")
        evaluation_status['status'] = 'no_warnings'
        evaluation_status['steps_completed'].append('evaluation_complete')
        return evaluation_status
    
    # Step 2: Generate slices
    if not generate_slices(project_path, warnings_file, project_name, checker_name, evaluation_status['has_warnings']):
        evaluation_status['steps_failed'].append('slice_generation')
        evaluation_status['status'] = 'slice_generation_failed'
        return evaluation_status
    
    evaluation_status['steps_completed'].append('slice_generation')
    
    # Step 3: Generate CFGs
    if not generate_cfgs(project_path, project_name, checker_name):
        evaluation_status['steps_failed'].append('cfg_generation')
        evaluation_status['status'] = 'cfg_generation_failed'
        return evaluation_status
    
    evaluation_status['steps_completed'].append('cfg_generation')
    
    # Step 4: Get available models
    model_names = get_available_models_for_checker(checker_name)
    logger.info(f"Found {len(model_names)} models for {checker_name} checker")
    
    if not model_names:
        logger.warning(f"No models available for {checker_name} checker")
        evaluation_status['steps_failed'].append('no_models')
        evaluation_status['status'] = 'no_models_available'
        return evaluation_status
    
    # Step 5: Generate predictions
    prediction_success = generate_predictions(project_path, project_name, checker_name, model_names)
    if not prediction_success:
        logger.warning(f"Prediction generation not yet implemented for {checker_name}")
        evaluation_status['steps_failed'].append('prediction_generation')
        evaluation_status['status'] = 'prediction_not_implemented'
        return evaluation_status
    
    evaluation_status['steps_completed'].append('prediction_generation')
    
    # Step 6: Compute metrics
    metrics = compute_metrics(project_path, project_name, checker_name, model_names)
    evaluation_status['metrics'] = metrics
    evaluation_status['steps_completed'].append('metrics_computation')
    evaluation_status['status'] = 'success'
    
    logger.info(f"Successfully evaluated {checker_name} checker on {project_name}")
    return evaluation_status

def evaluate_all_checkers(project_names: List[str]) -> Dict[str, Dict[str, dict]]:
    """Evaluate all checkers on specified projects."""
    results = {}
    
    for checker_name in SUPPORTED_CHECKERS:
        results[checker_name] = {}
        for project_name in project_names:
            evaluation_status = evaluate_checker_on_project(checker_name, project_name)
            results[checker_name][project_name] = evaluation_status
    
    return results

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate multiple checkers on projects')
    parser.add_argument('--checker', choices=SUPPORTED_CHECKERS, help='Specific checker to evaluate')
    parser.add_argument('--projects', nargs='+', help='Projects to evaluate')
    parser.add_argument('--all-checkers', action='store_true', help='Evaluate all checkers')
    
    args = parser.parse_args()
    
    # Default projects
    if not args.projects:
        args.projects = ['guava', 'jfreechart', 'plume-lib']
    
    if args.checker:
        # Evaluate single checker
        results = {}
        for project_name in args.projects:
            results[project_name] = evaluate_checker_on_project(args.checker, project_name)
        
        # Save results
        results_file = GEN_DATA_ROOT / f'{args.checker}_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    elif args.all_checkers:
        # Evaluate all checkers
        results = evaluate_all_checkers(args.projects)
        
        # Save results
        results_file = GEN_DATA_ROOT / 'multi_checker_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

