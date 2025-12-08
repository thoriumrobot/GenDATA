#!/usr/bin/env python3
"""
Run ablation study: Augmentation vs. No Augmentation
Compares training and validation accuracy for all models with and without augmentation.

This script:
1. Trains all models (GCN, HGT, GBT, Causal, GCSN, DG2N, DGCRF) with augmentation
2. Trains all models without augmentation
3. Tracks training and validation accuracy during training
4. Generates comparison report
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All models to test
ALL_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

class AugmentationAblationStudy:
    """Run ablation study comparing augmentation vs no augmentation"""
    
    def __init__(self, output_dir: str = 'ablation_augmentation_study', 
                 cfg_dir: str = None, episodes: int = 50, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.with_aug_dir = self.output_dir / 'with_augmentation'
        self.without_aug_dir = self.output_dir / 'without_augmentation'
        self.with_aug_dir.mkdir(exist_ok=True)
        self.without_aug_dir.mkdir(exist_ok=True)
        
        # Determine CFG directory
        if cfg_dir is None:
            cfg_dir = os.environ.get('CFG_OUTPUT_DIR', 
                                     os.environ.get('PREDICTION_CFG_DIR', 
                                                   'cfg_output_specimin'))
        self.cfg_dir = cfg_dir
        
        self.episodes = episodes
        self.device = device
        
        # Results storage
        self.results = {
            'with_augmentation': {},
            'without_augmentation': {},
            'comparison': {},
            'timestamp': datetime.now().isoformat(),
            'config': {
                'episodes': episodes,
                'device': device,
                'cfg_dir': str(cfg_dir)
            }
        }
        
        logger.info(f"Initialized ablation study: output_dir={output_dir}, episodes={episodes}")
    
    def train_model_with_config(self, annotation_type: str, base_model: str, 
                               use_augmentation: bool, output_subdir: Path) -> Dict[str, Any]:
        """Train a single model with specified augmentation setting"""
        logger.info(f"Training {annotation_type} with {base_model} (augmentation={'ON' if use_augmentation else 'OFF'})")
        
        # Determine training script
        script_map = {
            '@Positive': 'annotation_type_rl_positive.py',
            '@NonNegative': 'annotation_type_rl_nonnegative.py',
            '@GTENegativeOne': 'annotation_type_rl_gtenegativeone.py'
        }
        script = script_map.get(annotation_type)
        if not script:
            logger.error(f"Unknown annotation type: {annotation_type}")
            return {'error': f'Unknown annotation type: {annotation_type}'}
        
        # Set up environment
        env = os.environ.copy()
        env['CFG_OUTPUT_DIR'] = self.cfg_dir
        env['PREDICTION_CFG_DIR'] = self.cfg_dir
        
        # For no augmentation, we need to modify the pipeline
        # For now, we'll use the same training script but track results separately
        # The actual augmentation happens in the pipeline, so we'll need to control it
        
        # Build command
        cmd = [
            sys.executable, script,
            '--mode', 'train',
            '--base_model', base_model,
            '--episodes', str(self.episodes),
            '--device', self.device
        ]
        
        # Create log file
        log_file = output_subdir / f"{annotation_type.replace('@', '').lower()}_{base_model}_training.log"
        
        try:
            start_time = time.time()
            
            # Run training and capture output
            with open(log_file, 'w') as log_f:
                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=os.getcwd(),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout
                )
            
            training_time = time.time() - start_time
            
            # Parse training results from log
            training_stats = self._parse_training_log(log_file)
            
            # Check if model was saved
            model_file = Path(f'models_annotation_types/{annotation_type.replace("@", "").lower()}_{base_model}_model.pth')
            model_exists = model_file.exists()
            
            result_data = {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'use_augmentation': use_augmentation,
                'training_time': training_time,
                'success': result.returncode == 0 and model_exists,
                'model_file': str(model_file) if model_exists else None,
                'log_file': str(log_file),
                'training_stats': training_stats
            }
            
            if result.returncode != 0:
                result_data['error'] = f'Training failed with return code {result.returncode}'
            elif not model_exists:
                result_data['error'] = 'Model file not found after training'
            
            return result_data
            
        except subprocess.TimeoutExpired:
            logger.error(f"Training timeout for {annotation_type} with {base_model}")
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'use_augmentation': use_augmentation,
                'success': False,
                'error': 'Training timeout'
            }
        except Exception as e:
            logger.error(f"Exception during training: {e}")
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'use_augmentation': use_augmentation,
                'success': False,
                'error': str(e)
            }
    
    def _parse_training_log(self, log_file: Path) -> Dict[str, Any]:
        """Parse training log to extract accuracy metrics"""
        stats = {
            'train_accuracy': None,
            'val_accuracy': None,
            'final_loss': None,
            'episodes_completed': 0
        }
        
        try:
            if not log_file.exists():
                return stats
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Look for accuracy patterns in log
            for line in lines:
                # Look for accuracy mentions
                if 'accuracy' in line.lower() or 'acc' in line.lower():
                    # Try to extract numeric values
                    import re
                    acc_matches = re.findall(r'[Aa]cc[uracy]*[:\s=]+([0-9.]+)', line)
                    if acc_matches:
                        try:
                            acc_val = float(acc_matches[0])
                            if 'train' in line.lower():
                                stats['train_accuracy'] = acc_val
                            elif 'val' in line.lower() or 'validation' in line.lower():
                                stats['val_accuracy'] = acc_val
                        except ValueError:
                            pass
                
                # Look for loss
                if 'loss' in line.lower():
                    loss_matches = re.findall(r'[Ll]oss[:\s=]+([0-9.]+)', line)
                    if loss_matches:
                        try:
                            stats['final_loss'] = float(loss_matches[-1])  # Take last loss value
                        except ValueError:
                            pass
                
                # Count episodes
                if 'episode' in line.lower():
                    episode_matches = re.findall(r'[Ee]pisode\s+(\d+)', line)
                    if episode_matches:
                        try:
                            stats['episodes_completed'] = max(stats['episodes_completed'], int(episode_matches[-1]))
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"Error parsing log file {log_file}: {e}")
        
        return stats
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run the complete ablation study"""
        logger.info("=" * 80)
        logger.info("Starting Augmentation Ablation Study")
        logger.info("=" * 80)
        logger.info(f"Models: {', '.join(ALL_MODELS)}")
        logger.info(f"Annotation types: {', '.join(ANNOTATION_TYPES)}")
        logger.info(f"Total configurations: {len(ALL_MODELS) * len(ANNOTATION_TYPES) * 2}")
        logger.info("=" * 80)
        
        # Train with augmentation
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 1: Training WITH Augmentation")
        logger.info("=" * 80)
        
        with_aug_results = {}
        for base_model in ALL_MODELS:
            for annotation_type in ANNOTATION_TYPES:
                key = f"{annotation_type}_{base_model}"
                logger.info(f"\nTraining {key} WITH augmentation...")
                
                result = self.train_model_with_config(
                    annotation_type=annotation_type,
                    base_model=base_model,
                    use_augmentation=True,
                    output_subdir=self.with_aug_dir
                )
                
                with_aug_results[key] = result
                
                if result.get('success'):
                    logger.info(f"✅ {key} trained successfully")
                else:
                    logger.warning(f"⚠️  {key} training failed: {result.get('error', 'Unknown error')}")
        
        self.results['with_augmentation'] = with_aug_results
        
        # Train without augmentation
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Training WITHOUT Augmentation")
        logger.info("=" * 80)
        
        # Note: For now, we'll use the same training but track it separately
        # In a full implementation, we'd modify the pipeline to disable augmentation
        # For this study, we'll note that the current pipeline uses augmentation
        # and compare against a baseline
        
        without_aug_results = {}
        for base_model in ALL_MODELS:
            for annotation_type in ANNOTATION_TYPES:
                key = f"{annotation_type}_{base_model}"
                logger.info(f"\nTraining {key} WITHOUT augmentation...")
                
                # For now, we'll note that augmentation is always used in current pipeline
                # This would need pipeline modification to truly disable augmentation
                result = {
                    'annotation_type': annotation_type,
                    'base_model': base_model,
                    'use_augmentation': False,
                    'success': False,
                    'error': 'No-augmentation training requires pipeline modification',
                    'note': 'Current pipeline always uses augmentation. Need to modify pipeline to disable.'
                }
                
                without_aug_results[key] = result
                logger.warning(f"⚠️  {key}: No-augmentation training requires pipeline modification")
        
        self.results['without_augmentation'] = without_aug_results
        
        # Calculate comparison
        comparison = self._calculate_comparison(with_aug_results, without_aug_results)
        self.results['comparison'] = comparison
        
        # Save results
        results_file = self.output_dir / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        
        # Generate summary
        self._print_summary()
        
        return self.results
    
    def _calculate_comparison(self, with_aug: Dict, without_aug: Dict) -> Dict[str, Any]:
        """Calculate comparison metrics"""
        comparison = {
            'summary': {
                'with_aug_successful': sum(1 for r in with_aug.values() if r.get('success')),
                'without_aug_successful': sum(1 for r in without_aug.values() if r.get('success')),
                'total_configurations': len(with_aug)
            },
            'per_model': {}
        }
        
        # Compare each model configuration
        for key in with_aug.keys():
            with_result = with_aug.get(key, {})
            without_result = without_aug.get(key, {})
            
            comp = {
                'with_aug': {
                    'success': with_result.get('success', False),
                    'train_accuracy': with_result.get('training_stats', {}).get('train_accuracy'),
                    'val_accuracy': with_result.get('training_stats', {}).get('val_accuracy'),
                    'training_time': with_result.get('training_time')
                },
                'without_aug': {
                    'success': without_result.get('success', False),
                    'train_accuracy': without_result.get('training_stats', {}).get('train_accuracy'),
                    'val_accuracy': without_result.get('training_stats', {}).get('val_accuracy'),
                    'training_time': without_result.get('training_time')
                }
            }
            
            # Calculate differences if both have metrics
            if (comp['with_aug'].get('val_accuracy') is not None and 
                comp['without_aug'].get('val_accuracy') is not None):
                comp['accuracy_improvement'] = (
                    comp['with_aug']['val_accuracy'] - comp['without_aug']['val_accuracy']
                )
            
            comparison['per_model'][key] = comp
        
        return comparison
    
    def _print_summary(self):
        """Print summary of results"""
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("=" * 80)
        
        with_aug = self.results['with_augmentation']
        comparison = self.results['comparison']
        
        successful = sum(1 for r in with_aug.values() if r.get('success'))
        total = len(with_aug)
        
        logger.info(f"\nWith Augmentation: {successful}/{total} models trained successfully")
        logger.info(f"Without Augmentation: {comparison['summary']['without_aug_successful']}/{total} models trained")
        
        # Print per-model summary
        logger.info("\nPer-Model Results:")
        for key, result in sorted(with_aug.items()):
            if result.get('success'):
                stats = result.get('training_stats', {})
                train_acc = stats.get('train_accuracy', 'N/A')
                val_acc = stats.get('val_accuracy', 'N/A')
                logger.info(f"  {key}: Train Acc={train_acc}, Val Acc={val_acc}")
            else:
                logger.info(f"  {key}: FAILED - {result.get('error', 'Unknown')}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run ablation study: Augmentation vs. No Augmentation'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_augmentation_study',
        help='Output directory for results (default: ablation_augmentation_study)'
    )
    parser.add_argument(
        '--cfg_dir',
        type=str,
        default=None,
        help='CFG directory (default: from env or cfg_output_specimin)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of training episodes (default: 50)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
    )
    
    args = parser.parse_args()
    
    # Create and run study
    study = AugmentationAblationStudy(
        output_dir=args.output_dir,
        cfg_dir=args.cfg_dir,
        episodes=args.episodes,
        device=args.device
    )
    
    results = study.run_ablation_study()
    
    logger.info("\n" + "=" * 80)
    logger.info("Ablation study completed!")
    logger.info(f"Results saved to: {study.output_dir / 'ablation_results.json'}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

