#!/usr/bin/env python3
"""
Run Comprehensive Ablation Studies

Main execution script for running ablation studies on the GenDATA pipeline.
Supports running individual studies or all studies with configurable episodes.
"""

import os
import sys
import json
import argparse
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ablation_study_pipeline import AblationStudyPipeline
from ablation_study_evaluator import AblationStudyEvaluator
from ablation_study_report_generator import AblationStudyReportGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_optimal_device() -> str:
    """Detect optimal device for training"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        return 'cuda'
    else:
        logger.info("No GPU detected, using CPU")
        return 'cpu'

def verify_environment(project_root: str, warnings_file: str, cfwr_root: str) -> bool:
    """Verify that the environment is properly set up"""
    logger.info("=== Verifying Environment ===")
    
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
    
    # Check GenDATA directory
    if not os.path.exists(cfwr_root):
        logger.error(f"GenDATA root does not exist: {cfwr_root}")
        return False
    
    # Check for required files
    required_files = [
        'ablation_study_pipeline.py',
        'simple_annotation_type_pipeline.py',
        'checker_framework_integration.py'
    ]
    
    for file in required_files:
        file_path = os.path.join(cfwr_root, file)
        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            return False
    
    logger.info(f"Environment verification passed. Found {len(java_files)} Java files.")
    return True

def run_baseline_study(project_root: str, warnings_file: str, cfwr_root: str, 
                      output_dir: str, device: str, episodes: int) -> bool:
    """Run baseline study"""
    logger.info("=== Running Baseline Study ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_baseline_study(episodes=episodes)
    return bool(results)

def run_no_augmentation_study(project_root: str, warnings_file: str, cfwr_root: str, 
                             output_dir: str, device: str, episodes: int) -> bool:
    """Run no augmentation study"""
    logger.info("=== Running No Augmentation Study ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_no_augmentation_study(episodes=episodes)
    return bool(results)

def run_no_random_walk_study(project_root: str, warnings_file: str, cfwr_root: str, 
                            output_dir: str, device: str, episodes: int) -> bool:
    """Run no random walk study"""
    logger.info("=== Running No Random Walk Study ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_no_random_walk_study(episodes=episodes)
    return bool(results)

def run_transformation_ablations(project_root: str, warnings_file: str, cfwr_root: str, 
                                output_dir: str, device: str, episodes: int,
                                transform_subset: Optional[List[str]] = None) -> bool:
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
    if transform_subset is None:
        transform_names = pipeline.transformation_ablations
    else:
        transform_names = transform_subset
    
    success_count = 0
    total_count = len(transform_names)
    
    for i, transform_name in enumerate(transform_names, 1):
        logger.info(f"Running transformation ablation {i}/{total_count}: {transform_name}")
        
        try:
            results = pipeline.run_transformation_ablation_study(transform_name, episodes=episodes)
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
                     output_dir: str, device: str, episodes: int) -> bool:
    """Run all ablation studies"""
    logger.info("=== Running All Ablation Studies ===")
    
    pipeline = AblationStudyPipeline(
        project_root=project_root,
        warnings_file=warnings_file,
        cfwr_root=cfwr_root,
        output_dir=output_dir,
        device=device
    )
    
    results = pipeline.run_all_ablations(episodes=episodes)
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

def create_timestamped_output_dir(base_name: str) -> str:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive ablation studies on GenDATA pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablation studies with 10 episodes
  python run_comprehensive_ablation_studies.py --mode all --episodes 10

  # Run only baseline study
  python run_comprehensive_ablation_studies.py --mode baseline --episodes 10

  # Run only no augmentation study
  python run_comprehensive_ablation_studies.py --mode no_aug --episodes 10

  # Run transformation ablations for specific transformations
  python run_comprehensive_ablation_studies.py --mode transformations --transform_subset loop_conversion guard_reversal --episodes 10

  # Run with GPU acceleration
  python run_comprehensive_ablation_studies.py --mode all --device cuda --episodes 10

  # Evaluate existing results
  python run_comprehensive_ablation_studies.py --mode evaluate --results_dir ablation_studies_soot_gpu_final

  # Generate reports from existing results
  python run_comprehensive_ablation_studies.py --mode report --results_dir ablation_studies_soot_gpu_final
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['all', 'baseline', 'no_aug', 'no_rw', 'transformations', 'evaluate', 'report', 'compare'],
                       default='compare',
                       help='Ablation study mode (default compares augmentation vs no augmentation)')
    
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
                       default='ablation_studies_soot_gpu_final',
                       help='Output directory for ablation studies')
    parser.add_argument('--device', 
                       default='auto',
                       help='Device to use (cuda/cpu/auto, default: auto)')
    parser.add_argument('--episodes', 
                       type=int,
                       default=10,
                       help='Number of training episodes (default: 10)')
    parser.add_argument('--use_subset', 
                       action='store_true',
                       help='Use a smaller real-data subset warnings file (pre-generated)')
    parser.add_argument('--subset_file',
                       help='Path to a pre-generated real-data subset warnings file (.out)')
    parser.add_argument('--fast_compare', 
                       action='store_true',
                       help='Speed up compare by disabling random walk in baseline')
    
    # Transformation-specific parameters
    parser.add_argument('--transform_subset', 
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
    parser.add_argument('--timestamp_output', 
                       action='store_true',
                       help='Create timestamped output directory')
    
    args = parser.parse_args()
    
    # Set up paths
    if not os.path.isabs(args.warnings_file):
        args.warnings_file = os.path.join(args.cfwr_root, args.warnings_file)

    # Optionally create a tiny subset warnings file from real data
    subset_warnings_file = None
    if args.use_subset:
        if args.subset_file and os.path.exists(args.subset_file):
            subset_warnings_file = args.subset_file
            logger.info(f"Using provided subset warnings file: {subset_warnings_file}")
        else:
            logger.error("--use_subset specified but --subset_file is missing or does not exist")
            return 1
    
    # Detect optimal device
    if args.device == 'auto':
        args.device = detect_optimal_device()
    
    # Create timestamped output directory if requested
    if args.timestamp_output and args.mode in ['all', 'baseline', 'no_aug', 'no_rw', 'transformations']:
        args.output_dir = create_timestamped_output_dir(args.output_dir)
    
    # Validate environment unless skipped
    if not args.skip_validation and not args.validate_only:
        if not verify_environment(args.project_root, args.warnings_file, args.cfwr_root):
            logger.error("Environment validation failed")
            return 1
    
    if args.validate_only:
        logger.info("Environment validation completed successfully")
        return 0
    
    # Run selected mode
    success = False
    
    if args.mode == 'all':
        success = run_all_ablations(
            args.project_root, subset_warnings_file or args.warnings_file, args.cfwr_root,
            args.output_dir, args.device, args.episodes
        )
        
    elif args.mode == 'baseline':
        success = run_baseline_study(
            args.project_root, subset_warnings_file or args.warnings_file, args.cfwr_root,
            args.output_dir, args.device, args.episodes
        )
        
    elif args.mode == 'no_aug':
        success = run_no_augmentation_study(
            args.project_root, subset_warnings_file or args.warnings_file, args.cfwr_root,
            args.output_dir, args.device, args.episodes
        )
        
    elif args.mode == 'no_rw':
        success = run_no_random_walk_study(
            args.project_root, subset_warnings_file or args.warnings_file, args.cfwr_root,
            args.output_dir, args.device, args.episodes
        )
        
    elif args.mode == 'transformations':
        success = run_transformation_ablations(
            args.project_root, subset_warnings_file or args.warnings_file, args.cfwr_root,
            args.output_dir, args.device, args.episodes, args.transform_subset
        )
    
    elif args.mode == 'compare':
        # Fast comparison: with augmentation vs without augmentation
        logger.info("=== Compare: augmentation vs no augmentation ===")
        warnings_to_use = subset_warnings_file or args.warnings_file
        pipeline = AblationStudyPipeline(
            project_root=args.project_root,
            warnings_file=warnings_to_use,
            cfwr_root=args.cfwr_root,
            output_dir=args.output_dir,
            device=args.device
        )
        if args.fast_compare:
            logger.info("Compare mode: using no-random-walk baseline for speed")
            with_aug = pipeline.run_no_random_walk_study(episodes=args.episodes)
        else:
            with_aug = pipeline.run_baseline_study(episodes=args.episodes)
        without_aug = pipeline.run_no_augmentation_study(episodes=args.episodes)
        comparison = pipeline._calculate_performance_comparison({
            'baseline': with_aug,
            'no_augmentation': without_aug
        })
        out_path = os.path.join(args.output_dir, 'augmentation_vs_no_augmentation.json')
        with open(out_path, 'w') as f:
            json.dump({
                'with_augmentation': with_aug,
                'without_augmentation': without_aug,
                'performance_comparison': comparison
            }, f, indent=2)
        logger.info(f"Compare saved to {out_path}")
        success = True
        
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
