#!/usr/bin/env python3
"""
Run First Ablation Study: With vs Without Augmentation

This script runs the first ablation study comparing performance with vs without augmentation
on the index1.small.subset.out warnings file.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_first_ablation_study(
    time_limit_hours: int = 6,
    max_files: int = 50,
    max_variants: int = 10,
    log_interval: int = 100,
):
    """Run the first ablation study: baseline vs no augmentation

    Args:
        time_limit_hours: hard wall-clock limit for the entire run
        max_files: sample at most this many Java files from the project
        max_variants: cap augmentation variants per file
        log_interval: progress log interval for long loops
    """
    
    # Configuration
    project_root = '/home/ubuntu/checker-framework/checker/tests/index'
    warnings_file = '/home/ubuntu/GenDATA/index1.small.subset.out'
    cfwr_root = '/home/ubuntu/GenDATA'
    output_dir = '/home/ubuntu/GenDATA/ablation_studies_first'
    # Auto-detect device
    try:
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            logger.info("💻 Using CPU (CUDA not available)")
    except ImportError:
        device = 'cpu'
        logger.info("💻 Using CPU (PyTorch not available)")
    
    logger.info("🚀 Starting First Ablation Study: With vs Without Augmentation")
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"📄 Warnings file: {warnings_file}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"⏱️ Time limit (hours): {time_limit_hours}")
    logger.info(f"📦 Max files: {max_files} | 🔁 Max variants: {max_variants}")
    logger.info(f"📝 Log interval: {log_interval}")

    start_time = time.time()
    deadline_ts = start_time + (time_limit_hours * 3600)
    
    # Verify inputs
    if not os.path.exists(project_root):
        logger.error(f"Project root not found: {project_root}")
        return False
    
    if not os.path.exists(warnings_file):
        logger.error(f"Warnings file not found: {warnings_file}")
        return False
    
    # Import and run ablation study
    try:
        from ablation_study_pipeline import AblationStudyPipeline
        
        # Initialize pipeline
        pipeline = AblationStudyPipeline(
            project_root=project_root,
            warnings_file=warnings_file,
            cfwr_root=cfwr_root,
            output_dir=output_dir,
            device=device,
            run_checker_on_target=False,  # Use provided warnings file
            max_files_to_process=max_files,
            max_variants_per_file=max_variants,
            time_limit_hours=time_limit_hours,
            log_interval=log_interval,
        )
        
        logger.info("✅ AblationStudyPipeline initialized successfully")
        
        # Run baseline study (with augmentation)
        logger.info("🔬 Running baseline study (with augmentation)...")
        baseline_start = time.time()
        baseline_results = pipeline.run_baseline_study(episodes=5)  # Reduced episodes for faster execution
        logger.info(f"⏱️ Baseline duration: {time.time() - baseline_start:.2f}s")

        # Check global deadline after baseline
        if time.time() > deadline_ts:
            logger.error("⏰ Time limit reached after baseline. Aborting remaining studies.")
            return False
        
        if baseline_results:
            logger.info("✅ Baseline study completed successfully")
            logger.info(f"📊 Baseline results: {baseline_results}")
        else:
            logger.error("❌ Baseline study failed")
            return False
        
        # Run no augmentation study
        logger.info("🔬 Running no augmentation study...")
        noaug_start = time.time()
        no_aug_results = pipeline.run_no_augmentation_study(episodes=5)  # Reduced episodes for faster execution
        logger.info(f"⏱️ No-augmentation duration: {time.time() - noaug_start:.2f}s")
        
        if no_aug_results:
            logger.info("✅ No augmentation study completed successfully")
            logger.info(f"📊 No augmentation results: {no_aug_results}")
        else:
            logger.error("❌ No augmentation study failed")
            return False
        
        # Generate comparison
        logger.info("📈 Generating performance comparison...")
        
        baseline_metrics = baseline_results.get('metrics', {})
        no_aug_metrics = no_aug_results.get('metrics', {})
        
        baseline_reduction = baseline_metrics.get('reduction_percentage', 0)
        no_aug_reduction = no_aug_metrics.get('reduction_percentage', 0)
        
        improvement = baseline_reduction - no_aug_reduction
        
        logger.info("=" * 60)
        logger.info("📊 ABLATION STUDY RESULTS")
        logger.info("=" * 60)
        logger.info(f"Baseline (with augmentation): {baseline_reduction:.2f}% warning reduction")
        logger.info(f"No augmentation: {no_aug_reduction:.2f}% warning reduction")
        logger.info(f"Improvement from augmentation: {improvement:.2f}%")
        logger.info(f"Baseline training time: {baseline_metrics.get('training_time_seconds', 0):.2f}s")
        logger.info(f"No augmentation training time: {no_aug_metrics.get('training_time_seconds', 0):.2f}s")
        logger.info("=" * 60)
        
        if improvement > 0:
            logger.info("✅ Augmentation improves performance!")
        elif improvement == 0:
            logger.info("⚖️ Augmentation has no effect on performance")
        else:
            logger.info("⚠️ Augmentation decreases performance")
        
        total_elapsed = time.time() - start_time
        logger.info(f"⏱️ Total elapsed: {total_elapsed:.2f}s (limit {time_limit_hours*3600:.0f}s)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running ablation study: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_first_ablation_fast(time_limit_hours: int = 6, max_files: int = 30, max_variants: int = 1, log_interval: int = 100) -> bool:
    """Run fast ablation: slice-first with CF slicer and strict budgets."""
    project_root = '/home/ubuntu/checker-framework/checker/tests/index'
    warnings_file = '/home/ubuntu/GenDATA/index1.out'
    cfwr_root = '/home/ubuntu/GenDATA'
    output_dir = '/home/ubuntu/GenDATA/ablation_studies_first'
    logger.info("⚡ Running fast ablation (slice-first, CF slicer)")
    try:
        from ablation_study_pipeline import AblationStudyPipeline
        pipeline = AblationStudyPipeline(
            project_root=project_root,
            warnings_file=warnings_file,
            cfwr_root=cfwr_root,
            output_dir=output_dir,
            device='auto',
            max_files_to_process=max_files,
            max_variants_per_file=max_variants,
            time_limit_hours=time_limit_hours,
            log_interval=log_interval,
        )
        return bool(pipeline.run_all_ablations_fast(episodes=3))
    except Exception as e:
        logger.error(f"Fast ablation failed to start: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run First Ablation Study with runtime/sampling controls')
    parser.add_argument('--time_limit_hours', type=int, default=6)
    parser.add_argument('--max_files', type=int, default=50)
    parser.add_argument('--max_variants', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--fast', action='store_true', help='Run fast ablation (slice-first; Soot default slicer)')
    args = parser.parse_args()

    if args.fast:
        success = run_first_ablation_fast(
            time_limit_hours=args.time_limit_hours,
            max_files=args.max_files,
            max_variants=args.max_variants,
            log_interval=args.log_interval,
        )
    else:
        success = run_first_ablation_study(
            time_limit_hours=args.time_limit_hours,
            max_files=args.max_files,
            max_variants=args.max_variants,
            log_interval=args.log_interval,
        )
    if success:
        logger.info("🎉 First ablation study completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 First ablation study failed!")
        sys.exit(1)
