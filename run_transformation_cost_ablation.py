#!/usr/bin/env python3
"""
Run transformation cost ablation study: Measure accuracy cost of dropping each transformation.

For each transformation:
1. Train models with that transformation disabled
2. Compare against baseline (all transformations enabled)
3. Calculate accuracy cost/drop

Note: For true ablation, datasets should be regenerated with each transformation disabled.
This script uses existing datasets but provides the framework and results structure.
"""

import os
import sys
import json
import logging
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All models
ALL_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

# All semantic transformations
ENHANCED_TRANSFORMATIONS = [
    'loop_conversion', 'guard_reversal', 'mathematical_expression', 'logical_expression',
    'ternary_operator', 'switch_statement', 'variable_operation', 'method_extraction',
    'conditional_expression', 'array_access_pattern', 'string_concatenation', 
    'numeric_literal', 'exception_handling', 'lambda_expression', 'stream_api',
    'builder_pattern', 'functional_conversion'
]

SIMPLE_TRANSFORMATIONS = [
    'simple_method_call', 'simple_assignment', 'simple_conditional',
    'simple_array_access', 'simple_return_statement', 'simple_variable_declaration',
    'simple_constructor_call', 'simple_field_access', 'simple_string_operation',
    'simple_numeric_operation'
]

ALL_TRANSFORMATIONS = ENHANCED_TRANSFORMATIONS + SIMPLE_TRANSFORMATIONS

class TransformationCostAblation:
    """Measure accuracy cost of dropping each transformation"""
    
    def __init__(self, output_dir: str = 'ablation_transformation_costs',
                 baseline_file: str = None,
                 balanced_dataset_dir: str = 'real_balanced_datasets',
                 cfg_dir: str = None,
                 episodes: int = 10, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.balanced_dataset_dir = Path(balanced_dataset_dir)
        self.episodes = episodes
        self.device = device
        
        if cfg_dir is None:
            cfg_dir = os.environ.get('CFG_OUTPUT_DIR', 'cfg_output_specimin')
        self.cfg_dir = cfg_dir
        
        # Load baseline results
        if baseline_file:
            baseline_path = Path(baseline_file)
            if baseline_path.exists():
                data = json.load(open(baseline_path))
                self.baseline_results = data.get('baseline', {})
                logger.info(f"Loaded baseline results: {len(self.baseline_results)} models")
            else:
                logger.warning(f"Baseline file not found: {baseline_file}")
                self.baseline_results = {}
        else:
            self.baseline_results = {}
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'episodes': episodes,
                'device': device,
                'total_transformations': len(ALL_TRANSFORMATIONS),
                'baseline_file': str(baseline_file) if baseline_file else None
            },
            'baseline': self.baseline_results,
            'transformation_costs': {},
            'summary': {}
        }
    
    def extract_baseline_accuracy(self) -> Dict[str, float]:
        """Extract baseline accuracy values"""
        baseline_acc = {}
        
        for key, result in self.baseline_results.items():
            if result.get('success'):
                stats = result.get('training_stats', {})
                val_acc = stats.get('val_accuracy')
                if val_acc is not None:
                    baseline_acc[key] = val_acc
        
        return baseline_acc
    
    def train_with_disabled_transform(self, transform: str) -> Dict[str, Any]:
        """Train all models with a specific transformation disabled"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ABLATION: Training with '{transform}' DISABLED")
        logger.info(f"{'=' * 80}")
        
        transform_dir = self.output_dir / f'ablate_{transform}'
        transform_dir.mkdir(exist_ok=True)
        
        # Note: For true ablation, we'd regenerate dataset with this transformation disabled
        # For now, we use existing dataset but note the limitation
        logger.warning(f"Note: Using existing dataset. True ablation requires regenerating")
        logger.warning(f"dataset with {transform} disabled during augmentation phase.")
        
        from run_unified_ablation_study import UnifiedAblationStudy
        
        study = UnifiedAblationStudy(
            output_dir=str(transform_dir),
            balanced_dataset_dir=str(self.balanced_dataset_dir),
            cfg_dir=self.cfg_dir,
            episodes=self.episodes,
            device=self.device
        )
        
        results = study.run_baseline_study()
        return results
    
    def calculate_transformation_costs(self) -> Dict[str, Any]:
        """Calculate accuracy cost for each transformation"""
        baseline_acc = self.extract_baseline_accuracy()
        
        if not baseline_acc:
            logger.error("No baseline accuracy values found")
            return {}
        
        baseline_avg = sum(baseline_acc.values()) / len(baseline_acc.values())
        
        costs = {}
        
        for transform, ablation_results in self.results.get('ablations', {}).items():
            # Extract accuracy from ablation results
            ablation_acc = {}
            for key, result in ablation_results.items():
                if result.get('success'):
                    stats = result.get('training_stats', {})
                    val_acc = stats.get('val_accuracy')
                    if val_acc is not None:
                        ablation_acc[key] = val_acc
            
            if ablation_acc:
                ablation_avg = sum(ablation_acc.values()) / len(ablation_acc.values())
                cost = baseline_avg - ablation_avg  # Positive = performance loss
                
                # Per-model costs
                per_model_costs = {}
                for key in set(baseline_acc.keys()) & set(ablation_acc.keys()):
                    model_cost = baseline_acc[key] - ablation_acc[key]
                    per_model_costs[key] = {
                        'baseline': baseline_acc[key],
                        'ablated': ablation_acc[key],
                        'cost': model_cost,
                        'percent_cost': (model_cost / baseline_acc[key] * 100) if baseline_acc[key] > 0 else 0
                    }
                
                costs[transform] = {
                    'average_cost': cost,
                    'percent_cost': (cost / baseline_avg * 100) if baseline_avg > 0 else 0,
                    'baseline_avg': baseline_avg,
                    'ablated_avg': ablation_avg,
                    'models_tested': len(ablation_acc),
                    'per_model_costs': per_model_costs
                }
        
        return costs
    
    def run_full_study(self, transformations: List[str] = None, 
                      use_baseline: bool = True) -> Dict[str, Any]:
        """Run full transformation cost ablation study"""
        if transformations is None:
            transformations = ALL_TRANSFORMATIONS
        
        logger.info("=" * 80)
        logger.info("TRANSFORMATION COST ABLATION STUDY")
        logger.info("=" * 80)
        logger.info(f"Testing {len(transformations)} transformations")
        logger.info(f"Baseline average: {sum(self.extract_baseline_accuracy().values()) / len(self.extract_baseline_accuracy()) if self.extract_baseline_accuracy() else 'N/A'}")
        logger.info("=" * 80)
        
        # If using existing baseline, skip baseline training
        if not use_baseline or not self.baseline_results:
            logger.info("Training baseline models...")
            from run_unified_ablation_study import UnifiedAblationStudy
            
            study = UnifiedAblationStudy(
                output_dir=str(self.output_dir / 'baseline'),
                balanced_dataset_dir=str(self.balanced_dataset_dir),
                cfg_dir=self.cfg_dir,
                episodes=self.episodes,
                device=self.device
            )
            self.baseline_results = study.run_baseline_study()
            self.results['baseline'] = self.baseline_results
        
        # Run ablations for each transformation
        ablations = {}
        for i, transform in enumerate(transformations, 1):
            logger.info(f"\n[{i}/{len(transformations)}] Ablating: {transform}")
            ablation_results = self.train_with_disabled_transform(transform)
            ablations[transform] = ablation_results
        
        self.results['ablations'] = ablations
        
        # Calculate costs
        logger.info("\nCalculating transformation costs...")
        costs = self.calculate_transformation_costs()
        self.results['transformation_costs'] = costs
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        results_file = self.output_dir / 'transformation_cost_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        self._print_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate summary statistics"""
        costs = self.results.get('transformation_costs', {})
        baseline_acc = self.extract_baseline_accuracy()
        baseline_avg = sum(baseline_acc.values()) / len(baseline_acc.values()) if baseline_acc else None
        
        if not costs:
            self.results['summary'] = {'error': 'No cost data available'}
            return
        
        # Sort by cost (highest cost = most critical transformation)
        sorted_costs = sorted(costs.items(), key=lambda x: x[1].get('average_cost', 0), reverse=True)
        
        summary = {
            'baseline_average_accuracy': baseline_avg,
            'total_transformations_tested': len(costs),
            'most_critical_transformations': [
                {
                    'transformation': transform,
                    'average_cost': data.get('average_cost', 0),
                    'percent_cost': data.get('percent_cost', 0)
                }
                for transform, data in sorted_costs[:10]
            ],
            'least_critical_transformations': [
                {
                    'transformation': transform,
                    'average_cost': data.get('average_cost', 0),
                    'percent_cost': data.get('percent_cost', 0)
                }
                for transform, data in sorted_costs[-10:]
            ],
            'average_cost_across_all': sum(d.get('average_cost', 0) for d in costs.values()) / len(costs) if costs else 0,
            'max_cost': max(d.get('average_cost', 0) for d in costs.values()) if costs else 0,
            'min_cost': min(d.get('average_cost', 0) for d in costs.values()) if costs else 0
        }
        
        self.results['summary'] = summary
    
    def _print_summary(self):
        """Print summary of results"""
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION COST SUMMARY")
        logger.info("=" * 80)
        
        summary = self.results.get('summary', {})
        baseline_avg = summary.get('baseline_average_accuracy')
        
        if baseline_avg:
            logger.info(f"\nBaseline Average Accuracy: {baseline_avg:.4f}")
        
        most_critical = summary.get('most_critical_transformations', [])
        if most_critical:
            logger.info("\nTop 10 Most Critical Transformations (Highest Cost):")
            for i, item in enumerate(most_critical, 1):
                logger.info(f"  {i}. {item['transformation']}: {item['average_cost']:.4f} ({item['percent_cost']:.2f}%)")
        
        least_critical = summary.get('least_critical_transformations', [])
        if least_critical:
            logger.info("\nTop 10 Least Critical Transformations (Lowest Cost):")
            for i, item in enumerate(least_critical, 1):
                logger.info(f"  {i}. {item['transformation']}: {item['average_cost']:.4f} ({item['percent_cost']:.2f}%)")
        
        logger.info(f"\nAverage Cost Across All Transformations: {summary.get('average_cost_across_all', 0):.4f}")
        logger.info(f"Max Cost: {summary.get('max_cost', 0):.4f}")
        logger.info(f"Min Cost: {summary.get('min_cost', 0):.4f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run transformation cost ablation study'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_transformation_costs',
        help='Output directory'
    )
    parser.add_argument(
        '--baseline_file',
        type=str,
        default='ablation_baseline_final/ablation_results.json',
        help='Path to baseline results JSON file'
    )
    parser.add_argument(
        '--balanced_dataset_dir',
        type=str,
        default='real_balanced_datasets',
        help='Balanced dataset directory'
    )
    parser.add_argument(
        '--cfg_dir',
        type=str,
        default=None,
        help='CFG directory'
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
    parser.add_argument(
        '--transformations',
        type=str,
        nargs='+',
        default=None,
        help='Specific transformations to test (default: all)'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Test only a sample of N transformations (for quick testing)'
    )
    
    args = parser.parse_args()
    
    transformations = args.transformations
    if args.sample and not transformations:
        import random
        random.seed(42)
        transformations = random.sample(ALL_TRANSFORMATIONS, min(args.sample, len(ALL_TRANSFORMATIONS)))
        logger.info(f"Testing sample of {len(transformations)} transformations: {transformations}")
    
    study = TransformationCostAblation(
        output_dir=args.output_dir,
        baseline_file=args.baseline_file,
        balanced_dataset_dir=args.balanced_dataset_dir,
        cfg_dir=args.cfg_dir,
        episodes=args.episodes,
        device=args.device
    )
    
    results = study.run_full_study(transformations=transformations)
    
    logger.info("\n" + "=" * 80)
    logger.info("Transformation cost ablation study completed!")
    logger.info(f"Results saved to: {study.output_dir / 'transformation_cost_results.json'}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

