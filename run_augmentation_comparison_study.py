#!/usr/bin/env python3
"""
Run augmentation comparison study: With Augmentation vs. Without Augmentation

This script:
1. Trains all models with augmentation (baseline - already done)
2. Trains all models without augmentation (new)
3. Compares training and validation accuracy
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All models
ALL_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

# Feature-based models
FEATURE_BASED_MODELS = ['gbt', 'causal', 'enhanced_causal', 'dg2n', 'dgcrf']
GRAPH_BASED_MODELS = ['gcn', 'hgt', 'gcsn']

class AugmentationComparisonStudy:
    """Compare augmentation vs no augmentation"""
    
    def __init__(self, output_dir: str = 'ablation_augmentation_comparison',
                 balanced_dataset_dir: str = 'real_balanced_datasets',
                 cfg_dir: str = None,
                 cfg_dir_no_aug: str = None,
                 episodes: int = 10, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.balanced_dataset_dir = Path(balanced_dataset_dir)
        self.episodes = episodes
        self.device = device
        
        if cfg_dir is None:
            cfg_dir = os.environ.get('CFG_OUTPUT_DIR', 'cfg_output_specimin')
        self.cfg_dir = cfg_dir
        
        # CFG directory for non-augmented slices
        if cfg_dir_no_aug is None:
            # Try to find non-augmented CFG directory
            potential_dirs = [
                'cfg_output_no_aug',
                'cfg_output_specimin_no_aug',
                os.path.join('ablation_studies', 'no_augmentation', 'cfg_output')
            ]
            cfg_dir_no_aug = None
            for potential_dir in potential_dirs:
                if os.path.exists(potential_dir):
                    cfg_dir_no_aug = potential_dir
                    break
        self.cfg_dir_no_aug = cfg_dir_no_aug
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'episodes': episodes,
                'device': device,
                'cfg_dir': str(cfg_dir),
                'cfg_dir_no_aug': str(cfg_dir_no_aug) if cfg_dir_no_aug else None
            },
            'with_augmentation': {},
            'without_augmentation': {},
            'comparison': {}
        }
    
    def load_baseline_results(self, baseline_file: str) -> Dict[str, Any]:
        """Load existing baseline results (with augmentation)"""
        baseline_path = Path(baseline_file)
        if baseline_path.exists():
            data = json.load(open(baseline_path))
            return data.get('baseline', {})
        return {}
    
    def generate_no_augmentation_dataset(self, cfg_dir_no_aug: str, output_dataset_dir: Path, 
                                         examples_per_annotation: int = 2000) -> bool:
        """Generate balanced dataset from non-augmented CFGs"""
        logger.info("Generating no-augmentation dataset...")
        
        if not cfg_dir_no_aug:
            logger.error("CFG directory for non-augmented slices not specified")
            logger.error("Please provide --cfg_dir_no_aug or ensure non-augmented CFG directory exists")
            return False
        
        if not os.path.exists(cfg_dir_no_aug):
            logger.error(f"Non-augmented CFG directory does not exist: {cfg_dir_no_aug}")
            return False
        
        # Import the dataset generator utility
        from ablation_dataset_generator import AblationDatasetGenerator
        
        generator = AblationDatasetGenerator(random_seed=42)
        success, error = generator.generate_dataset(
            cfg_dir=cfg_dir_no_aug,
            output_dir=str(output_dataset_dir),
            examples_per_annotation=examples_per_annotation,
            target_balance=0.5,
            timeout=3600
        )
        
        if not success:
            logger.error(f"Failed to generate no-augmentation dataset: {error}")
            return False
        
        logger.info(f"Successfully generated no-augmentation dataset in {output_dataset_dir}")
        return True
    
    def train_all_models_no_aug(self, dataset_dir: Path) -> Dict[str, Any]:
        """Train all models without augmentation"""
        logger.info("=" * 80)
        logger.info("Training WITHOUT Augmentation")
        logger.info("=" * 80)
        
        from run_unified_ablation_study import UnifiedAblationStudy
        
        study = UnifiedAblationStudy(
            output_dir=str(self.output_dir / 'without_augmentation'),
            balanced_dataset_dir=str(dataset_dir),
            cfg_dir=self.cfg_dir,
            episodes=self.episodes,
            device=self.device
        )
        
        results = study.run_baseline_study()
        return results
    
    def train_all_models_with_aug(self) -> Dict[str, Any]:
        """Train all models with augmentation (baseline)"""
        logger.info("=" * 80)
        logger.info("Training WITH Augmentation (Baseline)")
        logger.info("=" * 80)
        
        from run_unified_ablation_study import UnifiedAblationStudy
        
        study = UnifiedAblationStudy(
            output_dir=str(self.output_dir / 'with_augmentation'),
            balanced_dataset_dir=str(self.balanced_dataset_dir),
            cfg_dir=self.cfg_dir,
            episodes=self.episodes,
            device=self.device
        )
        
        results = study.run_baseline_study()
        return results
    
    def calculate_comparison(self, with_aug: Dict, without_aug: Dict) -> Dict[str, Any]:
        """Calculate comparison metrics"""
        comparison = {
            'summary': {},
            'per_model': {},
            'per_annotation': {}
        }
        
        # Extract accuracy values
        with_aug_results = []
        without_aug_results = []
        
        for key in set(list(with_aug.keys()) + list(without_aug.keys())):
            with_result = with_aug.get(key, {})
            without_result = without_aug.get(key, {})
            
            with_stats = with_result.get('training_stats', {})
            without_stats = without_result.get('training_stats', {})
            
            with_val_acc = with_stats.get('val_accuracy')
            without_val_acc = without_stats.get('val_accuracy')
            
            if with_val_acc is not None:
                with_aug_results.append(with_val_acc)
            if without_val_acc is not None:
                without_aug_results.append(without_val_acc)
            
            # Per-model comparison
            if with_val_acc is not None or without_val_acc is not None:
                comparison['per_model'][key] = {
                    'with_augmentation': {
                        'val_accuracy': with_val_acc,
                        'success': with_result.get('success', False)
                    },
                    'without_augmentation': {
                        'val_accuracy': without_val_acc,
                        'success': without_result.get('success', False)
                    },
                    'improvement': (with_val_acc - without_val_acc) if (with_val_acc and without_val_acc) else None,
                    'percent_improvement': ((with_val_acc - without_val_acc) / without_val_acc * 100) if (with_val_acc and without_val_acc and without_val_acc > 0) else None
                }
        
        # Summary statistics
        if with_aug_results and without_aug_results:
            comparison['summary'] = {
                'with_augmentation': {
                    'average_val_accuracy': sum(with_aug_results) / len(with_aug_results),
                    'min_val_accuracy': min(with_aug_results),
                    'max_val_accuracy': max(with_aug_results),
                    'count': len(with_aug_results)
                },
                'without_augmentation': {
                    'average_val_accuracy': sum(without_aug_results) / len(without_aug_results),
                    'min_val_accuracy': min(without_aug_results),
                    'max_val_accuracy': max(without_aug_results),
                    'count': len(without_aug_results)
                },
                'overall_improvement': {
                    'average_improvement': (sum(with_aug_results) / len(with_aug_results)) - (sum(without_aug_results) / len(without_aug_results)),
                    'percent_improvement': ((sum(with_aug_results) / len(with_aug_results)) - (sum(without_aug_results) / len(without_aug_results))) / (sum(without_aug_results) / len(without_aug_results)) * 100
                }
            }
        
        # Per-annotation type
        by_annotation = {}
        for key, comp in comparison['per_model'].items():
            # Extract annotation type from key (e.g., "@Positive_gcn" -> "@Positive")
            ann_type = None
            for at in ANNOTATION_TYPES:
                if at in key:
                    ann_type = at
                    break
            
            if ann_type:
                if ann_type not in by_annotation:
                    by_annotation[ann_type] = {'with': [], 'without': []}
                
                if comp['with_augmentation']['val_accuracy']:
                    by_annotation[ann_type]['with'].append(comp['with_augmentation']['val_accuracy'])
                if comp['without_augmentation']['val_accuracy']:
                    by_annotation[ann_type]['without'].append(comp['without_augmentation']['val_accuracy'])
        
        for ann_type, data in by_annotation.items():
            if data['with'] and data['without']:
                comparison['per_annotation'][ann_type] = {
                    'with_augmentation_avg': sum(data['with']) / len(data['with']),
                    'without_augmentation_avg': sum(data['without']) / len(data['without']),
                    'improvement': (sum(data['with']) / len(data['with'])) - (sum(data['without']) / len(data['without'])),
                    'percent_improvement': ((sum(data['with']) / len(data['with'])) - (sum(data['without']) / len(data['without']))) / (sum(data['without']) / len(data['without'])) * 100
                }
        
        return comparison
    
    def run_comparison_study(self, baseline_file: str = None) -> Dict[str, Any]:
        """Run complete comparison study"""
        logger.info("=" * 80)
        logger.info("AUGMENTATION COMPARISON STUDY")
        logger.info("=" * 80)
        
        # Load baseline if provided
        if baseline_file:
            logger.info(f"Loading baseline results from {baseline_file}")
            with_aug_results = self.load_baseline_results(baseline_file)
            if with_aug_results:
                logger.info(f"Loaded {len(with_aug_results)} baseline results")
        else:
            # Train with augmentation
            logger.info("Training models WITH augmentation...")
            with_aug_results = self.train_all_models_with_aug()
        
        self.results['with_augmentation'] = with_aug_results
        
        # Generate no-augmentation dataset
        no_aug_dataset_dir = self.output_dir / 'no_augmentation_datasets'
        no_aug_dataset_dir.mkdir(exist_ok=True)
        
        # Generate dataset from non-augmented CFGs
        if not self.cfg_dir_no_aug:
            logger.warning("=" * 80)
            logger.warning("WARNING: CFG directory for non-augmented slices not specified")
            logger.warning("Cannot generate no-augmentation dataset without --cfg_dir_no_aug")
            logger.warning("Skipping no-augmentation training. Results will only include with-augmentation baseline.")
            logger.warning("=" * 80)
            self.results['without_augmentation'] = {}
            self.results['comparison'] = {
                'note': 'No-augmentation comparison skipped - cfg_dir_no_aug not provided',
                'with_augmentation_only': True
            }
            # Save partial results
            results_file = self.output_dir / 'augmentation_comparison_results.json'
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"\nPartial results saved to {results_file}")
            return self.results
        
        logger.info("Generating dataset from non-augmented CFGs...")
        dataset_generated = self.generate_no_augmentation_dataset(
            cfg_dir_no_aug=self.cfg_dir_no_aug,
            output_dataset_dir=no_aug_dataset_dir,
            examples_per_annotation=2000
        )
        
        if not dataset_generated:
            logger.error("Failed to generate no-augmentation dataset. Aborting comparison.")
            return self.results
        
        # Train without augmentation using the generated dataset
        logger.info("Training models WITHOUT augmentation...")
        without_aug_results = self.train_all_models_no_aug(no_aug_dataset_dir)
        
        self.results['without_augmentation'] = without_aug_results
        
        # Calculate comparison
        logger.info("Calculating comparison...")
        comparison = self.calculate_comparison(with_aug_results, without_aug_results)
        self.results['comparison'] = comparison
        
        # Save results
        results_file = self.output_dir / 'augmentation_comparison_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of comparison"""
        logger.info("\n" + "=" * 80)
        logger.info("AUGMENTATION COMPARISON SUMMARY")
        logger.info("=" * 80)
        
        comparison = self.results.get('comparison', {})
        summary = comparison.get('summary', {})
        
        if summary:
            with_aug = summary.get('with_augmentation', {})
            without_aug = summary.get('without_augmentation', {})
            improvement = summary.get('overall_improvement', {})
            
            logger.info(f"\nWITH Augmentation:")
            logger.info(f"  Average Val Accuracy: {with_aug.get('average_val_accuracy', 0):.4f}")
            logger.info(f"  Range: {with_aug.get('min_val_accuracy', 0):.4f} - {with_aug.get('max_val_accuracy', 0):.4f}")
            logger.info(f"  Models: {with_aug.get('count', 0)}")
            
            logger.info(f"\nWITHOUT Augmentation:")
            logger.info(f"  Average Val Accuracy: {without_aug.get('average_val_accuracy', 0):.4f}")
            logger.info(f"  Range: {without_aug.get('min_val_accuracy', 0):.4f} - {without_aug.get('max_val_accuracy', 0):.4f}")
            logger.info(f"  Models: {without_aug.get('count', 0)}")
            
            if improvement:
                logger.info(f"\nIMPROVEMENT from Augmentation:")
                logger.info(f"  Average Improvement: {improvement.get('average_improvement', 0):.4f}")
                logger.info(f"  Percent Improvement: {improvement.get('percent_improvement', 0):.2f}%")
        
        per_ann = comparison.get('per_annotation', {})
        if per_ann:
            logger.info("\nPer-Annotation Type Improvement:")
            for ann_type, data in sorted(per_ann.items()):
                logger.info(f"  {ann_type}: {data.get('percent_improvement', 0):.2f}% improvement")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run augmentation comparison study'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_augmentation_comparison',
        help='Output directory'
    )
    parser.add_argument(
        '--balanced_dataset_dir',
        type=str,
        default='real_balanced_datasets',
        help='Balanced dataset directory (with augmentation)'
    )
    parser.add_argument(
        '--baseline_file',
        type=str,
        default=None,
        help='Path to existing baseline results JSON file'
    )
    parser.add_argument(
        '--cfg_dir',
        type=str,
        default=None,
        help='CFG directory (for augmented slices, used by graph models)'
    )
    parser.add_argument(
        '--cfg_dir_no_aug',
        type=str,
        default=None,
        help='CFG directory for non-augmented slices (required for no-augmentation dataset generation)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Training epochs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device'
    )
    
    args = parser.parse_args()
    
    study = AugmentationComparisonStudy(
        output_dir=args.output_dir,
        balanced_dataset_dir=args.balanced_dataset_dir,
        cfg_dir=args.cfg_dir,
        cfg_dir_no_aug=args.cfg_dir_no_aug,
        episodes=args.episodes,
        device=args.device
    )
    
    results = study.run_comparison_study(baseline_file=args.baseline_file)
    
    logger.info("\n" + "=" * 80)
    logger.info("Augmentation comparison study completed!")
    logger.info(f"Results saved to: {study.output_dir / 'augmentation_comparison_results.json'}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

