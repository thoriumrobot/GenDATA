#!/usr/bin/env python3
"""
Comprehensive RL Training Pipeline

This script orchestrates the entire reinforcement learning training process:
1. Generate augmented slices
2. Generate CFGs with dataflow information
3. Train RL models with Checker Framework feedback
4. Evaluate and compare different approaches
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import torch
from pathlib import Path
import time
import logging
from typing import List, Dict, Tuple

# Import our modules
from pipeline import run_slicing, run_cfg_generation
from augment_slices import augment_file
from enhanced_rl_training import EnhancedReinforcementLearningTrainer
from checker_framework_integration import CheckerFrameworkEvaluator, CheckerType, BatchEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLTrainingPipeline:
    """Comprehensive RL training pipeline"""
    
    def __init__(self, project_root: str, output_dir: str, models_dir: str = "models"):
        self.project_root = project_root
        self.output_dir = output_dir
        self.models_dir = models_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize evaluator
        self.evaluator = CheckerFrameworkEvaluator()
        self.batch_evaluator = BatchEvaluator()
        
        # Pipeline statistics
        self.pipeline_stats = {
            'slicing_time': 0,
            'augmentation_time': 0,
            'cfg_generation_time': 0,
            'training_time': 0,
            'total_files_processed': 0,
            'total_warnings_found': 0,
            'model_performance': {}
        }
    
    def run_complete_pipeline(self, slicer_type='cf', model_types=['hgt', 'gbt', 'causal', 'dg2n', 'gcn'], 
                            num_episodes=100, checker_type='nullness'):
        """Run the complete RL training pipeline using augmented slices (default behavior)"""
        logger.info("Starting comprehensive RL training pipeline with augmented slices")
        
        # Step 1: Generate slices
        logger.info("Step 1: Generating slices from warnings")
        start_time = time.time()
        slices_dir = self._generate_slices(slicer_type)
        self.pipeline_stats['slicing_time'] = time.time() - start_time
        
        if not slices_dir:
            logger.error("Failed to generate slices")
            return False
        
        # Step 2: Augment slices (default behavior)
        logger.info("Step 2: Augmenting slices (default behavior)")
        start_time = time.time()
        augmented_slices_dir = self._augment_slices(slices_dir)
        self.pipeline_stats['augmentation_time'] = time.time() - start_time
        
        if not augmented_slices_dir:
            logger.error("Failed to augment slices")
            return False
        
        # Step 3: Generate CFGs from augmented slices
        logger.info("Step 3: Generating CFGs with dataflow information from augmented slices")
        start_time = time.time()
        cfg_dir = self._generate_cfgs(augmented_slices_dir)
        self.pipeline_stats['cfg_generation_time'] = time.time() - start_time
        
        if not cfg_dir:
            logger.error("Failed to generate CFGs")
            return False
        
        # Step 4: Train RL models on augmented slices (default behavior)
        logger.info("Step 4: Training RL models on augmented slices (default behavior)")
        start_time = time.time()
        training_results = self._train_rl_models(cfg_dir, augmented_slices_dir, 
                                               model_types, num_episodes, checker_type)
        self.pipeline_stats['training_time'] = time.time() - start_time
        
        # Step 5: Evaluate models
        logger.info("Step 5: Evaluating trained models")
        evaluation_results = self._evaluate_models(training_results, augmented_slices_dir, checker_type)
        
        # Step 6: Generate final report
        logger.info("Step 6: Generating final report")
        self._generate_final_report(evaluation_results)
        
        logger.info("Comprehensive RL training pipeline completed!")
        return True
    
    def _generate_slices(self, slicer_type):
        """Generate slices from warnings"""
        try:
            warnings_file = os.path.join(self.output_dir, "warnings.out")
            slices_dir = os.path.join(self.output_dir, f"slices_{slicer_type}")
            
            # Run slicing
            success = run_slicing(
                project_root=self.project_root,
                warnings_file=warnings_file,
                cfwr_root=os.getcwd(),
                base_slices_dir=self.output_dir,
                slicer_type=slicer_type
            )
            
            if success and os.path.exists(slices_dir):
                # Count files
                java_files = []
                for root, dirs, files in os.walk(slices_dir):
                    for file in files:
                        if file.endswith('.java'):
                            java_files.append(os.path.join(root, file))
                
                self.pipeline_stats['total_files_processed'] = len(java_files)
                logger.info(f"Generated {len(java_files)} slice files")
                return slices_dir
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating slices: {e}")
            return None
    
    def _augment_slices(self, slices_dir):
        """Augment slices with synthetic code"""
        try:
            augmented_dir = os.path.join(self.output_dir, "augmented_slices")
            os.makedirs(augmented_dir, exist_ok=True)
            
            # Augment each Java file
            java_files = []
            for root, dirs, files in os.walk(slices_dir):
                for file in files:
                    if file.endswith('.java'):
                        java_file = os.path.join(root, file)
                        java_files.append(java_file)
            
            augmented_count = 0
            for java_file in java_files:
                try:
                    # Create output path maintaining directory structure
                    rel_path = os.path.relpath(java_file, slices_dir)
                    base_name = os.path.splitext(rel_path)[0]
                    os.makedirs(os.path.dirname(os.path.join(augmented_dir, rel_path)), exist_ok=True)
                    
                    # Create 10 variants per file (factor of 10)
                    for variant_idx in range(10):
                        variant_dir = os.path.join(augmented_dir, f"{base_name}__aug{variant_idx}")
                        os.makedirs(variant_dir, exist_ok=True)
                        output_path = os.path.join(variant_dir, os.path.basename(rel_path))
                        
                        # Augment the file
                        augmented_content = augment_file(java_file, variant_idx)
                        with open(output_path, 'w') as f:
                            f.write(augmented_content)
                        augmented_count += 1
                except Exception as e:
                    logger.warning(f"Failed to augment {java_file}: {e}")
                    continue
            
            logger.info(f"Generated {augmented_count} augmented slice files")
            return augmented_dir if augmented_count > 0 else None
            
        except Exception as e:
            logger.error(f"Error augmenting slices: {e}")
            return None
    
    def _generate_cfgs(self, slices_dir):
        """Generate CFGs with dataflow information"""
        try:
            cfg_dir = os.path.join(self.output_dir, "cfgs")
            
            # Run CFG generation
            success = run_cfg_generation(slices_dir, cfg_dir)
            
            if success and os.path.exists(cfg_dir):
                # Count CFG files
                cfg_files = []
                for root, dirs, files in os.walk(cfg_dir):
                    for file in files:
                        if file.endswith('.json'):
                            cfg_files.append(os.path.join(root, file))
                
                logger.info(f"Generated {len(cfg_files)} CFG files")
                return cfg_dir
            
            return None
            
        except Exception as e:
            logger.error(f"Error generating CFGs: {e}")
            return None
    
    def _train_rl_models(self, cfg_dir, slices_dir, model_types, num_episodes, checker_type):
        """Train RL models on augmented slices (default behavior)"""
        training_results = {}
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model on augmented slices")
            
            try:
                # Initialize trainer
                trainer = EnhancedReinforcementLearningTrainer(
                    model_type=model_type,
                    learning_rate=0.001,
                    device='cpu',
                    checker_type=checker_type,
                    reward_strategy='adaptive'
                )
                
                # Train the model on augmented slices (default behavior)
                trainer.train(
                    slices_dir=slices_dir,
                    cfg_dir=cfg_dir,
                    num_episodes=num_episodes,
                    batch_size=32,
                    use_augmented_slices=True  # Default to augmented slices
                )
                
                # Save training results
                training_results[model_type] = {
                    'trainer': trainer,
                    'model_path': f"{self.models_dir}/enhanced_rl_{model_type}_final.pth",
                    'stats_path': f"{self.models_dir}/enhanced_rl_{model_type}_stats.json"
                }
                
                logger.info(f"Completed training {model_type} model")
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                training_results[model_type] = None
        
        return training_results
    
    def _evaluate_models(self, training_results, slices_dir, checker_type):
        """Evaluate trained models"""
        evaluation_results = {}
        
        for model_type, result in training_results.items():
            if result is None:
                continue
            
            logger.info(f"Evaluating {model_type} model")
            
            try:
                # Load the trained model
                trainer = result['trainer']
                
                # Test on a subset of files
                test_files = self._get_test_files(slices_dir, max_files=10)
                
                if not test_files:
                    logger.warning(f"No test files found for {model_type}")
                    continue
                
                # Evaluate model performance
                model_performance = self._evaluate_single_model(trainer, test_files, checker_type)
                evaluation_results[model_type] = model_performance
                
                logger.info(f"Completed evaluation of {model_type} model")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_type} model: {e}")
                evaluation_results[model_type] = None
        
        return evaluation_results
    
    def _get_test_files(self, slices_dir, max_files=10):
        """Get test files for evaluation"""
        test_files = []
        
        for root, dirs, files in os.walk(slices_dir):
            for file in files:
                if file.endswith('.java'):
                    test_files.append(os.path.join(root, file))
                    if len(test_files) >= max_files:
                        break
            if len(test_files) >= max_files:
                break
        
        return test_files
    
    def _evaluate_single_model(self, trainer, test_files, checker_type):
        """Evaluate a single model"""
        performance_stats = {
            'total_files': len(test_files),
            'successful_predictions': 0,
            'total_reward': 0.0,
            'warning_reductions': 0,
            'warning_increases': 0,
            'no_change': 0,
            'avg_reward': 0.0
        }
        
        for java_file in test_files:
            try:
                # Find corresponding CFG file
                cfg_file = self._find_corresponding_cfg(java_file)
                if not cfg_file:
                    continue
                
                # Load CFG data
                with open(cfg_file, 'r') as f:
                    cfg_data = json.load(f)
                
                # Run evaluation episode
                reward, original_result = trainer.train_episode_advanced(cfg_data, java_file)
                
                if original_result and original_result.success:
                    performance_stats['successful_predictions'] += 1
                    performance_stats['total_reward'] += reward
                    
                    # Track warning changes
                    if reward > 0:
                        performance_stats['warning_reductions'] += 1
                    elif reward < 0:
                        performance_stats['warning_increases'] += 1
                    else:
                        performance_stats['no_change'] += 1
                
            except Exception as e:
                logger.error(f"Error evaluating {java_file}: {e}")
                continue
        
        # Calculate average reward
        if performance_stats['successful_predictions'] > 0:
            performance_stats['avg_reward'] = performance_stats['total_reward'] / performance_stats['successful_predictions']
        
        return performance_stats
    
    def _find_corresponding_cfg(self, java_file):
        """Find CFG file corresponding to Java file"""
        try:
            # Extract base name
            base_name = os.path.splitext(os.path.basename(java_file))[0]
            
            # Look for CFG file
            cfg_dir = os.path.join(self.output_dir, "cfgs")
            for root, dirs, files in os.walk(cfg_dir):
                for file in files:
                    if file.startswith(base_name) and file.endswith('.json'):
                        return os.path.join(root, file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding CFG file: {e}")
            return None
    
    def _generate_final_report(self, evaluation_results):
        """Generate final evaluation report"""
        try:
            report = {
                'pipeline_statistics': self.pipeline_stats,
                'model_evaluations': evaluation_results,
                'summary': self._generate_summary(evaluation_results)
            }
            
            # Save report
            report_path = os.path.join(self.output_dir, "rl_training_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            self._print_summary(report['summary'])
            
            logger.info(f"Final report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating final report: {e}")
    
    def _generate_summary(self, evaluation_results):
        """Generate summary of evaluation results"""
        summary = {
            'total_models_trained': len([r for r in evaluation_results.values() if r is not None]),
            'best_model': None,
            'best_reward': -float('inf'),
            'total_training_time': self.pipeline_stats['training_time'],
            'pipeline_efficiency': {
                'slicing_time': self.pipeline_stats['slicing_time'],
                'augmentation_time': self.pipeline_stats['augmentation_time'],
                'cfg_generation_time': self.pipeline_stats['cfg_generation_time'],
                'training_time': self.pipeline_stats['training_time']
            }
        }
        
        # Find best model
        for model_type, result in evaluation_results.items():
            if result and result['avg_reward'] > summary['best_reward']:
                summary['best_model'] = model_type
                summary['best_reward'] = result['avg_reward']
        
        return summary
    
    def _print_summary(self, summary):
        """Print summary to console"""
        print("\n" + "="*60)
        print("RL TRAINING PIPELINE SUMMARY")
        print("="*60)
        print(f"Total models trained: {summary['total_models_trained']}")
        print(f"Best model: {summary['best_model']}")
        print(f"Best average reward: {summary['best_reward']:.3f}")
        print(f"Total training time: {summary['total_training_time']:.2f} seconds")
        print("\nPipeline Efficiency:")
        for stage, time_taken in summary['pipeline_efficiency'].items():
            print(f"  {stage}: {time_taken:.2f} seconds")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Comprehensive RL Training Pipeline')
    parser.add_argument('--project_root', required=True, help='Root directory of the target project')
    parser.add_argument('--output_dir', required=True, help='Output directory for all results')
    parser.add_argument('--models_dir', default='models', help='Directory for trained models')
    parser.add_argument('--slicer', choices=['cf', 'specimin', 'wala'], default='cf',
                       help='Slicer to use (cf=CheckerFrameworkSlicer)')
    parser.add_argument('--model_types', nargs='+', choices=['hgt', 'gbt', 'causal', 'dg2n', 'gcn'], 
                       default=['hgt', 'gbt', 'causal', 'dg2n', 'gcn'], help='Models to train')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--checker_type', choices=['nullness', 'index'], default='nullness',
                       help='Type of Checker Framework checker to use')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RLTrainingPipeline(
        project_root=args.project_root,
        output_dir=args.output_dir,
        models_dir=args.models_dir
    )
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline(
        slicer_type=args.slicer,
        model_types=args.model_types,
        num_episodes=args.episodes,
        checker_type=args.checker_type
    )
    
    if success:
        logger.info("Pipeline completed successfully!")
    else:
        logger.error("Pipeline failed!")
        exit(1)

if __name__ == '__main__':
    main()
