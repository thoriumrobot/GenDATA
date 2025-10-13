#!/usr/bin/env python3
"""
Run Ablation Studies

Main execution script for running ablation studies on the GenDATA pipeline.
Supports running individual ablation studies or comprehensive analysis.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Import ablation study components
from ablation_study_pipeline import AblationStudyPipeline
from ablation_study_evaluator import AblationStudyEvaluator
from ablation_study_report_generator import AblationStudyReportGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_baseline_study(project_root: str, warnings_file: str, cfwr_root: str, 
                      output_dir: str, device: str) -> bool:
    """Run baseline study"""
    logger.info("=== Running Baseline Study ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_baseline_study()
    return bool(results)

def run_no_augmentation_study(project_root: str, warnings_file: str, cfwr_root: str, 
                             output_dir: str, device: str) -> bool:
    """Run no augmentation study"""
    logger.info("=== Running No Augmentation Study ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_no_augmentation_study()
    return bool(results)

def run_no_random_walk_study(project_root: str, warnings_file: str, cfwr_root: str, 
                            output_dir: str, device: str) -> bool:
    """Run no random walk study"""
    logger.info("=== Running No Random Walk Study ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_no_random_walk_study()
    return bool(results)

def run_transformation_ablations(project_root: str, warnings_file: str, cfwr_root: str, 
                                output_dir: str, device: str, transform_names: Optional[List[str]] = None) -> bool:
    """Run transformation ablation studies"""
    logger.info("=== Running Transformation Ablation Studies ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    # Get transformation list
    if transform_names is None:
        transform_names = pipeline.transformation_ablations
    
    success_count = 0
    total_count = len(transform_names)
    
    for i, transform_name in enumerate(transform_names, 1):
        logger.info(f"Running transformation ablation {i}/{total_count}: {transform_name}")
        
        try:
            results = pipeline.run_transformation_ablation_study(transform_name)
            if results:
                success_count += 1
                logger.info(f"✓ Successfully completed ablation for {transform_name}")
            else:
                logger.error(f"✗ Failed ablation for {transform_name}")
        except Exception as e:
            logger.error(f"✗ Error in ablation for {transform_name}: {e}")
    
    logger.info(f"Transformation ablations completed: {success_count}/{total_count} successful")
    return success_count > 0

def run_all_ablations(project_root: str, warnings_file: str, cfwr_root: str, 
                     output_dir: str, device: str) -> bool:
    """Run all ablation studies"""
    logger.info("=== Running All Ablation Studies ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_all_ablations()
    return bool(results)

def evaluate_results(results_dir: str) -> bool:
    """Evaluate ablation study results"""
    logger.info("=== Evaluating Ablation Study Results ===")
    
    try:
        evaluator = AblationStudyEvaluator(results_dir)
        analysis_file = evaluator.save_analysis_report()
        
        logger.info(f"Analysis report saved to: {analysis_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error evaluating results: {e}")
        return False

def generate_reports(results_dir: str, report_only: bool = False, 
                    visualizations_only: bool = False) -> bool:
    """Generate reports and visualizations"""
    logger.info("=== Generating Reports and Visualizations ===")
    
    try:
        generator = AblationStudyReportGenerator(results_dir)
        
        if visualizations_only:
            visualization_files = generator.generate_all_visualizations()
            logger.info(f"Generated {len(visualization_files)} visualization files")
        elif report_only:
            report_file = generator.generate_markdown_report()
            logger.info(f"Generated markdown report: {report_file}")
        else:
            report_file = generator.generate_complete_report()
            logger.info(f"Generated complete report: {report_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        return False

def validate_environment(project_root: str, warnings_file: str) -> bool:
    """Validate that required files and directories exist"""
    logger.info("=== Validating Environment ===")
    
    # Check project root
    if not os.path.exists(project_root):
        logger.error(f"Project root does not exist: {project_root}")
        return False
    
    # Check warnings file
    if not os.path.exists(warnings_file):
        logger.error(f"Warnings file does not exist: {warnings_file}")
        return False
    
    # Check for Java files in project
    java_files = []
    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    if not java_files:
        logger.error(f"No Java files found in project root: {project_root}")
        return False
    
    logger.info(f"Environment validation passed. Found {len(java_files)} Java files.")
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run ablation studies on GenDATA pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablation studies
  python run_ablation_studies.py --mode all

  # Run only baseline study
  python run_ablation_studies.py --mode baseline

  # Run only no augmentation study
  python run_ablation_studies.py --mode no_aug

  # Run transformation ablations for specific transformations
  python run_ablation_studies.py --mode transformations --transform_names loop_conversion guard_reversal

  # Evaluate existing results
  python run_ablation_studies.py --mode evaluate --results_dir ablation_studies

  # Generate reports from existing results
  python run_ablation_studies.py --mode report --results_dir ablation_studies
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['all', 'baseline', 'no_aug', 'no_rw', 'transformations', 'evaluate', 'report'],
                       default='all',
                       help='Ablation study mode')
    
    # Input parameters
    parser.add_argument('--project_root', 
                       default='/home/ubuntu/checker-framework/checker/tests/index',
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', 
                       default='index1.out',
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', 
                       default='/home/ubuntu/GenDATA',
                       help='Root directory of GenDATA project')
    
    # Output parameters
    parser.add_argument('--output_dir', 
                       default='ablation_studies',
                       help='Output directory for ablation studies')
    parser.add_argument('--device', 
                       default='cuda',
                       help='Device to use (cuda/cpu/auto, default: cuda)')
    
    # Transformation-specific parameters
    parser.add_argument('--transform_names', 
                       nargs='*',
                       help='Specific transformation names for transformation mode')
    
    # Evaluation and reporting parameters
    parser.add_argument('--results_dir', 
                       help='Results directory for evaluation/reporting modes')
    parser.add_argument('--report_only', 
                       action='store_true',
                       help='Generate only markdown report (no visualizations)')
    parser.add_argument('--visualizations_only', 
                       action='store_true',
                       help='Generate only visualizations (no markdown report)')
    
    # Utility flags
    parser.add_argument('--validate_only', 
                       action='store_true',
                       help='Only validate environment, do not run studies')
    parser.add_argument('--skip_validation', 
                       action='store_true',
                       help='Skip environment validation')
    
    args = parser.parse_args()
    
    # Set up paths
    if not os.path.isabs(args.warnings_file):
        args.warnings_file = os.path.join(args.cfwr_root, args.warnings_file)
    
    # Validate environment unless skipped
    if not args.skip_validation and not args.validate_only:
        if not validate_environment(args.project_root, args.warnings_file):
            logger.error("Environment validation failed")
            return 1
    
    if args.validate_only:
        logger.info("Environment validation completed successfully")
        return 0
    
    # Run selected mode
    success = False
    
    if args.mode == 'all':
        success = run_all_ablations(
            args.project_root, args.warnings_file, args.cfwr_root,
            args.output_dir, args.device
        )
        
    elif args.mode == 'baseline':
        success = run_baseline_study(
            args.project_root, args.warnings_file, args.cfwr_root,
            args.output_dir, args.device
        )
        
    elif args.mode == 'no_aug':
        success = run_no_augmentation_study(
            args.project_root, args.warnings_file, args.cfwr_root,
            args.output_dir, args.device
        )
        
    elif args.mode == 'no_rw':
        success = run_no_random_walk_study(
            args.project_root, args.warnings_file, args.cfwr_root,
            args.output_dir, args.device
        )
        
    elif args.mode == 'transformations':
        success = run_transformation_ablations(
            args.project_root, args.warnings_file, args.cfwr_root,
            args.output_dir, args.device, args.transform_names
        )
        
    elif args.mode == 'evaluate':
        results_dir = args.results_dir or args.output_dir
        success = evaluate_results(results_dir)
        
    elif args.mode == 'report':
        results_dir = args.results_dir or args.output_dir
        success = generate_reports(results_dir, args.report_only, args.visualizations_only)
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    if success:
        logger.info("Ablation study completed successfully")
        
        # Auto-generate evaluation and reports for study modes
        if args.mode in ['all', 'baseline', 'no_aug', 'no_rw', 'transformations']:
            logger.info("Auto-generating evaluation and reports...")
            
            try:
                evaluate_results(args.output_dir)
                generate_reports(args.output_dir)
                logger.info("Evaluation and reports generated successfully")
            except Exception as e:
                logger.warning(f"Could not auto-generate evaluation/reports: {e}")
        
        return 0
    else:
        logger.error("Ablation study failed")
        return 1

if __name__ == '__main__':
    exit(main())
