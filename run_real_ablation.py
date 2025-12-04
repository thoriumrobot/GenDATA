#!/usr/bin/env python3
"""
Real Ablation Study: With vs Without Augmentation

This script runs a real ablation study with actual pipeline execution,
measuring real performance metrics from augmentation vs no augmentation.
"""

import os
import sys
import json
import time
import logging
import glob
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_device():
    """Detect available device (cuda or cpu)"""
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
    
    return device

def count_files_in_directory(directory, pattern="*"):
    """Count files in directory matching pattern"""
    if not os.path.exists(directory):
        return 0
    
    files = glob.glob(os.path.join(directory, pattern))
    return len(files)

def count_warnings_in_file(warnings_file):
    """Count the number of warnings in the warnings file"""
    try:
        with open(warnings_file, 'r') as f:
            lines = f.readlines()
        
        warning_count = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Look for Checker Framework warning patterns
                if 'compiler.' in line and ('.warn.' in line or '.err.' in line):
                    warning_count += 1
        
        return warning_count
    except Exception as e:
        logger.error(f"Error counting warnings: {e}")
        return 0

def run_real_ablation_study():
    """Run the real ablation study with measured performance"""
    
    # Configuration
    project_root = '/home/ubuntu/checker-framework/checker/tests/index'
    warnings_file = '/home/ubuntu/GenDATA/index1.small.subset.out'
    cfwr_root = '/home/ubuntu/GenDATA'
    output_dir = '/home/ubuntu/GenDATA/real_ablation_results'
    device = detect_device()
    
    logger.info("🚀 Starting Real Ablation Study: With vs Without Augmentation")
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"📄 Warnings file: {warnings_file}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🖥️ Device: {device}")
    
    # Verify inputs
    if not os.path.exists(project_root):
        logger.error(f"Project root not found: {project_root}")
        return False
    
    if not os.path.exists(warnings_file):
        logger.error(f"Warnings file not found: {warnings_file}")
        return False
    
    # Count baseline warnings
    baseline_warnings = count_warnings_in_file(warnings_file)
    logger.info(f"📊 Baseline warnings: {baseline_warnings}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import the pipeline
        from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
        
        results = {}
        
        # Run baseline study (with augmentation)
        logger.info("🔬 Running baseline study (with augmentation)...")
        baseline_dir = os.path.join(output_dir, 'baseline')
        os.makedirs(baseline_dir, exist_ok=True)
        
        baseline_pipeline = SimpleAnnotationTypePipeline(
            project_root=project_root,
            warnings_file=warnings_file,
            cfwr_root=cfwr_root,
            mode='train',
            device=device,
            augment_first=True,  # Enable augmentation
            disable_random_walk=False,  # Enable random walk for augmentation
            run_checker_on_target=False
        )
        
        # Update directories
        baseline_pipeline.slices_dir = os.path.join(baseline_dir, 'slices')
        baseline_pipeline.cfg_dir = os.path.join(baseline_dir, 'cfg_output')
        baseline_pipeline.models_dir = os.path.join(baseline_dir, 'models')
        
        # Create directories
        os.makedirs(baseline_pipeline.slices_dir, exist_ok=True)
        os.makedirs(baseline_pipeline.cfg_dir, exist_ok=True)
        os.makedirs(baseline_pipeline.models_dir, exist_ok=True)
        
        # Run baseline training
        start_time = time.time()
        baseline_success = baseline_pipeline.run_training_pipeline(episodes=1, base_model='gcn')
        baseline_time = time.time() - start_time
        
        if baseline_success:
            logger.info(f"✅ Baseline study completed in {baseline_time:.2f}s")
            
            # Collect baseline metrics
            baseline_slices = count_files_in_directory(baseline_pipeline.slices_dir)
            baseline_cfgs = count_files_in_directory(baseline_pipeline.cfg_dir, "*.json")
            baseline_models = count_files_in_directory(baseline_pipeline.models_dir)
            
            results['baseline'] = {
                'case_name': 'baseline',
                'training_time': baseline_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': {
                    'slices_generated': baseline_slices,
                    'cfgs_generated': baseline_cfgs,
                    'models_trained': baseline_models,
                    'training_time_seconds': baseline_time,
                    'baseline_warnings': baseline_warnings,
                    'models_found': baseline_models,
                    'augmentation_used': True
                }
            }
            
            logger.info(f"📊 Baseline metrics: {baseline_slices} slices, {baseline_cfgs} CFGs, {baseline_models} models")
        else:
            logger.error("❌ Baseline study failed")
            return False
        
        # Run no augmentation study
        logger.info("🔬 Running no augmentation study...")
        no_aug_dir = os.path.join(output_dir, 'no_augmentation')
        os.makedirs(no_aug_dir, exist_ok=True)
        
        no_aug_pipeline = SimpleAnnotationTypePipeline(
            project_root=project_root,
            warnings_file=warnings_file,
            cfwr_root=cfwr_root,
            mode='train',
            device=device,
            augment_first=False,  # Disable augmentation
            disable_random_walk=True,
            run_checker_on_target=False
        )
        
        # Update directories
        no_aug_pipeline.slices_dir = os.path.join(no_aug_dir, 'slices')
        no_aug_pipeline.cfg_dir = os.path.join(no_aug_dir, 'cfg_output')
        no_aug_pipeline.models_dir = os.path.join(no_aug_dir, 'models')
        
        # Create directories
        os.makedirs(no_aug_pipeline.slices_dir, exist_ok=True)
        os.makedirs(no_aug_pipeline.cfg_dir, exist_ok=True)
        os.makedirs(no_aug_pipeline.models_dir, exist_ok=True)
        
        # Run no augmentation training
        start_time = time.time()
        no_aug_success = no_aug_pipeline.run_training_pipeline(episodes=1, base_model='gcn')
        no_aug_time = time.time() - start_time
        
        if no_aug_success:
            logger.info(f"✅ No augmentation study completed in {no_aug_time:.2f}s")
            
            # Collect no augmentation metrics
            no_aug_slices = count_files_in_directory(no_aug_pipeline.slices_dir)
            no_aug_cfgs = count_files_in_directory(no_aug_pipeline.cfg_dir, "*.json")
            no_aug_models = count_files_in_directory(no_aug_pipeline.models_dir)
            
            results['no_augmentation'] = {
                'case_name': 'no_augmentation',
                'training_time': no_aug_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': {
                    'slices_generated': no_aug_slices,
                    'cfgs_generated': no_aug_cfgs,
                    'models_trained': no_aug_models,
                    'training_time_seconds': no_aug_time,
                    'baseline_warnings': baseline_warnings,
                    'models_found': no_aug_models,
                    'augmentation_used': False
                }
            }
            
            logger.info(f"📊 No augmentation metrics: {no_aug_slices} slices, {no_aug_cfgs} CFGs, {no_aug_models} models")
        else:
            logger.error("❌ No augmentation study failed")
            return False
        
        # Generate comparison
        logger.info("📈 Generating performance comparison...")
        
        baseline_metrics = results['baseline']['metrics']
        no_aug_metrics = results['no_augmentation']['metrics']
        
        # Calculate improvements
        slice_improvement = baseline_slices - no_aug_slices
        cfg_improvement = baseline_cfgs - no_aug_cfgs
        time_overhead = baseline_time - no_aug_time
        
        slice_improvement_pct = (slice_improvement / max(no_aug_slices, 1)) * 100
        cfg_improvement_pct = (cfg_improvement / max(no_aug_cfgs, 1)) * 100
        
        logger.info("=" * 60)
        logger.info("📊 REAL ABLATION STUDY RESULTS")
        logger.info("=" * 60)
        logger.info(f"Baseline warnings: {baseline_warnings}")
        logger.info(f"Baseline (with augmentation): {baseline_slices} slices, {baseline_cfgs} CFGs, {baseline_time:.2f}s")
        logger.info(f"No augmentation: {no_aug_slices} slices, {no_aug_cfgs} CFGs, {no_aug_time:.2f}s")
        logger.info(f"Improvement from augmentation:")
        logger.info(f"  - Slices: +{slice_improvement} (+{slice_improvement_pct:.1f}%)")
        logger.info(f"  - CFGs: +{cfg_improvement} (+{cfg_improvement_pct:.1f}%)")
        logger.info(f"  - Training time overhead: +{time_overhead:.2f}s")
        logger.info("=" * 60)
        
        if slice_improvement > 0:
            logger.info("✅ Augmentation increases data diversity!")
        else:
            logger.info("⚠️ Augmentation did not increase data diversity")
        
        # Save comprehensive results
        comprehensive_results = {
            'ablation_study_summary': {
                'total_studies': 2,
                'total_time_seconds': baseline_time + no_aug_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'project_root': project_root,
                'warnings_file': warnings_file,
                'device_used': device,
                'study_type': 'real_measured_performance'
            },
            'individual_results': results,
            'performance_comparison': {
                'baseline': baseline_metrics,
                'no_augmentation': no_aug_metrics,
                'improvements': {
                    'slices_improvement': slice_improvement,
                    'slices_improvement_percentage': slice_improvement_pct,
                    'cfgs_improvement': cfg_improvement,
                    'cfgs_improvement_percentage': cfg_improvement_pct,
                    'training_time_overhead': time_overhead,
                    'data_diversity_increase': slice_improvement > 0
                }
            }
        }
        
        # Save to results file
        results_file = os.path.join(output_dir, 'real_ablation_results.json')
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"📄 Comprehensive results saved to: {results_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running real ablation study: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_real_ablation_study()
    if success:
        logger.info("🎉 Real ablation study completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Real ablation study failed!")
        sys.exit(1)
