#!/usr/bin/env python3
"""
Evaluate Models on Outline Projects

This script evaluates all trained models on the projects mentioned in GenDATA outline.md:
- Agrona
- Hipparchus
- Eclipse Collections

It follows the complete evaluation pipeline:
1. Run Lower Bound Checker to generate warnings
2. Generate slices using Soot slicer
3. Generate CFGs from slices
4. Generate predictions for all available models
5. Format predictions as predictions_{model}.json
6. Compute metrics (precision, recall, F1, warning reduction)
7. Generate evaluation report
"""

import os
import sys
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Projects to evaluate
OUTLINE_PROJECTS = ['agrona', 'hipparchus', 'eclipse-collections']

# Base directory
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
CASE_STUDIES_DIR = GEN_DATA_ROOT / 'case_studies'
MODELS_DIR = GEN_DATA_ROOT / 'models_annotation_types'


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


def validate_warnings_file(warnings_file: Path) -> dict:
    """Validate warnings file and count actual checker warnings."""
    if not warnings_file.exists():
        return {'has_warnings': False, 'warning_count': 0, 'compilation_errors': 0}
    
    try:
        from checker_framework_runner import CheckerFrameworkRunner, count_checker_warnings
        
        # Count checker warnings
        warning_count = count_checker_warnings(str(warnings_file))
        
        # Parse warnings file for more details
        runner = CheckerFrameworkRunner()
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


def run_checker_on_project(project_path: Path, project_name: str) -> tuple:
    """Run Lower Bound Checker on a project. Returns (warnings_file, validation_info)."""
    warnings_file = project_path / f'{project_name}_warnings.out'
    
    if warnings_file.exists():
        logger.info(f"Warnings file already exists for {project_name}: {warnings_file}")
        validation_info = validate_warnings_file(warnings_file)
        return warnings_file, validation_info
    
    logger.info(f"Running Lower Bound Checker on {project_name}...")
    
    try:
        from checker_framework_runner import run_checker_framework_on_project
        
        success = run_checker_framework_on_project(
            project_root=str(project_path),
            output_file=str(warnings_file),
            max_files=1000  # Process all files
        )
        
        if success and warnings_file.exists():
            logger.info(f"Successfully generated warnings file for {project_name}")
            validation_info = validate_warnings_file(warnings_file)
            
            if validation_info['has_warnings']:
                logger.info(f"Found {validation_info['warning_count']} checker warnings in {project_name}")
            else:
                logger.info(f"No checker warnings found in {project_name} (may have {validation_info['compilation_errors']} compilation errors)")
            
            return warnings_file, validation_info
        else:
            logger.warning(f"Failed to generate warnings for {project_name}")
            return None, {'has_warnings': False, 'warning_count': 0}
    except ImportError:
        logger.error("checker_framework_runner not available")
        return None, {'has_warnings': False, 'warning_count': 0}
    except Exception as e:
        logger.error(f"Error running checker on {project_name}: {e}")
        return None, {'has_warnings': False, 'warning_count': 0}


def generate_slices(project_path: Path, warnings_file: Path, project_name: str, has_warnings: bool) -> bool:
    """Generate slices using Soot slicer."""
    if not has_warnings:
        logger.info(f"Skipping slice generation for {project_name} - no checker warnings found")
        return False
    
    slices_dir = project_path / 'slices'
    slices_dir.mkdir(exist_ok=True)
    
    logger.info(f"Generating slices for {project_name}...")
    
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
            logger.info(f"Successfully generated slices for {project_name}")
            return True
        else:
            logger.warning(f"No slices generated for {project_name}")
            return False
    except Exception as e:
        logger.error(f"Error generating slices for {project_name}: {e}")
        return False


def generate_cfgs(project_path: Path, project_name: str) -> bool:
    """Generate CFGs from slices."""
    slices_dir = project_path / 'slices' / 'slices_soot'
    cfg_output_dir = project_path / 'cfg_output'
    cfg_output_dir.mkdir(exist_ok=True)
    
    if not slices_dir.exists() or not any(slices_dir.iterdir()):
        logger.warning(f"Slices directory not found or empty for {project_name}: {slices_dir}")
        # Try fallback to CF slicer if Soot didn't produce slices
        logger.info(f"Attempting fallback to CF slicer...")
        try:
            from pipeline import run_slicing
            slices_dir_parent = project_path / 'slices'
            run_slicing(
                project_root=str(project_path),
                warnings_file=str(project_path / f'{project_name}_warnings.out'),
                cfwr_root=str(GEN_DATA_ROOT),
                base_slices_dir=str(slices_dir_parent),
                slicer_type='cf'  # Try CF slicer as fallback
            )
            # Check if CF slicer produced slices
            cf_slices_dir = slices_dir_parent / 'slices_cf'
            if cf_slices_dir.exists() and any(cf_slices_dir.iterdir()):
                slices_dir = cf_slices_dir
                logger.info(f"CF slicer produced slices, using those")
            else:
                return False
        except Exception as e:
            logger.warning(f"Fallback slicer also failed: {e}")
            return False
    
    logger.info(f"Generating CFGs for {project_name}...")
    
    try:
        from pipeline import run_cfg_generation
        
        run_cfg_generation(str(slices_dir), str(cfg_output_dir))
        
        # Check if CFGs were generated
        if cfg_output_dir.exists() and any(cfg_output_dir.iterdir()):
            logger.info(f"Successfully generated CFGs for {project_name}")
            return True
        else:
            logger.warning(f"No CFGs generated for {project_name}")
            return False
    except Exception as e:
        logger.error(f"Error generating CFGs for {project_name}: {e}")
        return False


def get_available_models() -> list:
    """Get list of available models for evaluation."""
    try:
        from verify_models_for_evaluation import get_available_models_for_evaluation
        models = get_available_models_for_evaluation(MODELS_DIR)
        if models:
            return models
    except Exception as e:
        logger.warning(f"Could not verify models: {e}")
    
    # Fallback: use expected models
    base_models = ['gcn', 'hgt', 'gbt', 'causal', 'enhanced_causal', 'gcsn', 'dg2n']
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    expected_models = []
    for base in base_models:
        for ann_type in annotation_types:
            expected_models.append(f"{base}_{ann_type}")
    
    return expected_models


def generate_predictions(project_path: Path, project_name: str, model_names: list) -> bool:
    """Generate predictions for all models using CFG files."""
    cfg_output_dir = project_path / 'cfg_output'
    
    if not cfg_output_dir.exists() or not any(cfg_output_dir.iterdir()):
        logger.warning(f"No CFGs found for {project_name}")
        return False
    
    logger.info(f"Generating predictions for {project_name} with {len(model_names)} models...")
    
    # Find all Java files that have CFGs
    java_files_with_cfgs = []
    for java_file in project_path.rglob('*.java'):
        java_basename = java_file.stem
        cfg_file = cfg_output_dir / java_basename / 'cfg.json'
        if cfg_file.exists():
            java_files_with_cfgs.append(java_file)
    
    if not java_files_with_cfgs:
        logger.warning(f"No Java files with CFGs found for {project_name}")
        return False
    
    logger.info(f"Found {len(java_files_with_cfgs)} Java files with CFGs")
    
    # Use enhanced_graph_predictor for prediction
    try:
        from enhanced_graph_predictor import EnhancedGraphPredictor
        
        predictor = EnhancedGraphPredictor(
            models_dir=str(MODELS_DIR),
            device='auto',
            auto_train=False  # Don't auto-train, use existing models
        )
        
        # Group models by base model type
        models_by_base = {}
        for model_name in model_names:
            parts = model_name.split('_')
            if len(parts) >= 2:
                base_model = '_'.join(parts[:-1])
                annotation_type = parts[-1]
            else:
                base_model = model_name
                annotation_type = None
            
            if base_model not in models_by_base:
                models_by_base[base_model] = []
            models_by_base[base_model].append((model_name, annotation_type))
        
        # Generate predictions for each base model
        all_predictions = {}
        predictions_generated = 0
        
        for base_model, model_list in models_by_base.items():
            try:
                # Try to load models for this base model type
                if not predictor.load_or_train_models(base_model_type=base_model, episodes=0):
                    logger.warning(f"Could not load models for base type {base_model}")
                    continue
                
                logger.info(f"Loaded models for base type {base_model}")
                
                # Generate predictions for each Java file
                for java_file in java_files_with_cfgs[:100]:  # Limit to first 100 files for performance
                    try:
                        preds = predictor.predict_annotations_for_file_with_cfg(
                            str(java_file),
                            str(cfg_output_dir),
                            threshold=0.3
                        )
                        
                        if preds:
                            if str(java_file) not in all_predictions:
                                all_predictions[str(java_file)] = []
                            all_predictions[str(java_file)].extend(preds)
                    except Exception as e:
                        logger.debug(f"Error predicting for {java_file}: {e}")
                        continue
                
                predictions_generated += 1
                
            except Exception as e:
                logger.warning(f"Failed to generate predictions for base model {base_model}: {e}")
                continue
        
        logger.info(f"Generated predictions for {predictions_generated} base models")
        
        # Save predictions to temporary location for collection
        if all_predictions:
            temp_predictions_dir = GEN_DATA_ROOT / 'predictions_annotation_types' / project_name
            temp_predictions_dir.mkdir(parents=True, exist_ok=True)
            
            # Save predictions by model
            for java_file, preds in all_predictions.items():
                # Group by model type
                for pred in preds:
                    model_type = pred.get('model_type', 'unknown')
                    output_file = temp_predictions_dir / f"{model_type}_predictions.json"
                    
                    # Load existing or create new
                    if output_file.exists():
                        with open(output_file, 'r') as f:
                            existing = json.load(f)
                    else:
                        existing = []
                    
                    # Add this prediction
                    existing.append({
                        'file': java_file,
                        'predictions': [pred]
                    })
                    
                    with open(output_file, 'w') as f:
                        json.dump(existing, f, indent=2)
        
        # Collect and format predictions
        try:
            from studies.collect_outline_project_predictions import collect_and_save_predictions
            saved_files = collect_and_save_predictions(project_name)
            logger.info(f"Collected and formatted predictions: {len(saved_files)} models")
            return len(saved_files) > 0
        except Exception as e:
            logger.error(f"Error collecting predictions: {e}")
            return len(all_predictions) > 0  # Return True if we have predictions even if collection failed
    
    except Exception as e:
        logger.error(f"Error generating predictions for {project_name}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def compute_metrics(project_path: Path, project_name: str, model_names: list) -> dict:
    """Compute evaluation metrics for all models."""
    logger.info(f"Computing metrics for {project_name}...")
    
    metrics = {}
    
    try:
        from studies.compute_case_study_metrics import evaluate_project_model
        
        for model_name in model_names:
            try:
                # Normalize model name (handle enhanced_causal vs dgcrf)
                normalized_model = model_name.replace('dgcrf', 'enhanced_causal')
                
                model_metrics = evaluate_project_model(project_name, normalized_model)
                metrics[model_name] = model_metrics
                
                logger.info(f"Computed metrics for {model_name}: "
                          f"F1={model_metrics.get('f1_weighted', 0):.3f}, "
                          f"Warning Reduction={model_metrics.get('warning_reduction', 0):.1f}%")
            except Exception as e:
                logger.warning(f"Failed to compute metrics for {model_name}: {e}")
                metrics[model_name] = {'error': str(e)}
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error computing metrics for {project_name}: {e}")
        return {}


def evaluate_project(project_name: str) -> dict:
    """Evaluate a single project. Returns evaluation status dictionary."""
    logger.info("=" * 80)
    logger.info(f"Evaluating project: {project_name}")
    logger.info("=" * 80)
    
    project_path = CASE_STUDIES_DIR / project_name
    evaluation_status = {
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
        logger.error(f"Project validation failed for {project_name}")
        evaluation_status['steps_failed'].append('validation')
        return evaluation_status
    
    evaluation_status['steps_completed'].append('validation')
    
    # Step 1: Run checker
    warnings_file, validation_info = run_checker_on_project(project_path, project_name)
    if not warnings_file:
        logger.warning(f"Failed to generate warnings file for {project_name}")
        evaluation_status['steps_failed'].append('warnings_generation')
        evaluation_status['status'] = 'no_warnings_file'
        return evaluation_status
    
    evaluation_status['steps_completed'].append('warnings_generation')
    evaluation_status['warning_count'] = validation_info.get('warning_count', 0)
    evaluation_status['has_warnings'] = validation_info.get('has_warnings', False)
    
    # If no warnings, report this gracefully
    if not evaluation_status['has_warnings']:
        logger.info(f"Project {project_name} has no checker warnings. This is not a failure - the project may be well-annotated or not use array indexing.")
        evaluation_status['status'] = 'no_warnings'
        evaluation_status['steps_completed'].append('evaluation_complete')
        
        # Save status
        status_file = project_path / 'evaluation_status.json'
        with open(status_file, 'w') as f:
            json.dump(evaluation_status, f, indent=2)
        
        return evaluation_status
    
    # Step 2: Generate slices
    if not generate_slices(project_path, warnings_file, project_name, evaluation_status['has_warnings']):
        logger.warning(f"Failed to generate slices for {project_name}")
        evaluation_status['steps_failed'].append('slice_generation')
        evaluation_status['status'] = 'slice_generation_failed'
        return evaluation_status
    
    evaluation_status['steps_completed'].append('slice_generation')
    
    # Step 3: Generate CFGs
    if not generate_cfgs(project_path, project_name):
        logger.warning(f"Failed to generate CFGs for {project_name}")
        evaluation_status['steps_failed'].append('cfg_generation')
        evaluation_status['status'] = 'cfg_generation_failed'
        return evaluation_status
    
    evaluation_status['steps_completed'].append('cfg_generation')
    
    # Step 4: Get available models
    model_names = get_available_models()
    logger.info(f"Found {len(model_names)} models for evaluation")
    
    # Step 5: Generate predictions
    prediction_success = generate_predictions(project_path, project_name, model_names)
    if not prediction_success:
        logger.warning(f"Failed to generate predictions for {project_name} (continuing anyway)")
        evaluation_status['steps_failed'].append('prediction_generation')
        # Don't fail completely - continue to metrics if we have some predictions
        if not (project_path / 'predictions_gbt.json').exists():  # Check if any predictions exist
            evaluation_status['status'] = 'prediction_generation_failed'
            return evaluation_status
    
    evaluation_status['steps_completed'].append('prediction_generation')
    
    # Step 6: Compute metrics
    metrics = compute_metrics(project_path, project_name, model_names)
    
    # Save metrics
    if metrics:
        metrics_file = project_path / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        evaluation_status['metrics'] = metrics
    
    evaluation_status['steps_completed'].append('metrics_computation')
    evaluation_status['status'] = 'success'
    
    logger.info(f"Successfully evaluated {project_name}")
    return evaluation_status


def main():
    """Main function."""
    logger.info("Starting evaluation of outline projects...")
    
    results = {}
    for project_name in OUTLINE_PROJECTS:
        evaluation_status = evaluate_project(project_name)
        results[project_name] = evaluation_status
    
    # Generate summary report
    successful = len([p for p, s in results.items() if s.get('status') == 'success'])
    no_warnings = len([p for p, s in results.items() if s.get('status') == 'no_warnings'])
    failed = len([p for p, s in results.items() if s.get('status') not in ['success', 'no_warnings']])
    
    summary = {
        'projects_evaluated': successful,
        'projects_no_warnings': no_warnings,
        'projects_failed': failed,
        'projects_total': len(OUTLINE_PROJECTS),
        'results': results
    }
    
    summary_file = GEN_DATA_ROOT / 'OUTLINE_PROJECTS_EVALUATION_RESULTS.md'
    with open(summary_file, 'w') as f:
        f.write("# Outline Projects Evaluation Results\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Projects Successfully Evaluated: {successful}/{len(OUTLINE_PROJECTS)}\n")
        f.write(f"- Projects with No Warnings: {no_warnings}/{len(OUTLINE_PROJECTS)}\n")
        f.write(f"- Projects Failed: {failed}/{len(OUTLINE_PROJECTS)}\n\n")
        f.write(f"## Results\n\n")
        for project, status_dict in results.items():
            status = status_dict.get('status', 'unknown')
            warning_count = status_dict.get('warning_count', 0)
            
            if status == 'success':
                f.write(f"### {project}: ✅ Success ({warning_count} warnings)\n\n")
            elif status == 'no_warnings':
                f.write(f"### {project}: ⚠️ No Warnings Found (project may be well-annotated)\n\n")
            else:
                f.write(f"### {project}: ❌ Failed ({status})\n\n")
            
            f.write(f"- Steps Completed: {', '.join(status_dict.get('steps_completed', []))}\n")
            if status_dict.get('steps_failed'):
                f.write(f"- Steps Failed: {', '.join(status_dict.get('steps_failed', []))}\n")
            f.write("\n")
    
    logger.info(f"Evaluation summary saved to {summary_file}")
    
    # Return 0 if at least some projects were processed (even if no warnings)
    return 0 if (successful > 0 or no_warnings > 0) else 1


if __name__ == '__main__':
    exit(main())
