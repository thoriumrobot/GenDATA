#!/usr/bin/env python3
"""
Run transformation ablation study: Disable each semantic transformation one at a time.
Tracks training and validation accuracy for all models.

Note: For true transformation ablation, this would require regenerating the entire pipeline
(slicing → augmentation with disabled transform → CFG generation → dataset generation) for each transformation.
This is computationally expensive, so this script provides a framework and uses existing datasets.
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
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# All models
ALL_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

# All semantic transformations - get from JDT transformer if available
try:
    from jdt_semantic_transformer import JdtSemanticTransformer
    transformer = JdtSemanticTransformer()
    ENHANCED_TRANSFORMATIONS = transformer.get_available_transformations('enhanced')
    SIMPLE_TRANSFORMATIONS = transformer.get_available_transformations('simple')
    ALL_TRANSFORMATIONS = ENHANCED_TRANSFORMATIONS + SIMPLE_TRANSFORMATIONS
    logger.info(f"Loaded {len(ENHANCED_TRANSFORMATIONS)} enhanced + {len(SIMPLE_TRANSFORMATIONS)} simple = {len(ALL_TRANSFORMATIONS)} total transformations from JDT")
except Exception as e:
    logger.warning(f"Could not load transformations from JDT: {e}, using fallback list")
    # Fallback list (matches what JDT actually provides)
    ENHANCED_TRANSFORMATIONS = [
        'loop_conversion', 'guard_reversal', 'mathematical_expression', 'logical_expression',
        'ternary_operator', 'switch_statement', 'variable_operation', 'brace_normalization',
        'string_concatenation', 'numeric_literal'
    ]
    SIMPLE_TRANSFORMATIONS = [
        'simple_method_call', 'simple_assignment', 'simple_conditional',
        'simple_array_access', 'simple_return_statement', 'simple_variable_declaration',
        'simple_constructor_call', 'simple_field_access', 'simple_string_operation',
        'simple_numeric_operation'
    ]
    ALL_TRANSFORMATIONS = ENHANCED_TRANSFORMATIONS + SIMPLE_TRANSFORMATIONS

class TransformationAblationFinal:
    """Run transformation ablation study"""
    
    def __init__(self, output_dir: str = 'ablation_transformations_final',
                 balanced_dataset_dir: str = 'real_balanced_datasets',
                 cfg_dir: str = None,
                 cfg_dir_base_pattern: str = None,
                 episodes: int = 10, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.balanced_dataset_dir = Path(balanced_dataset_dir)
        self.episodes = episodes
        self.device = device
        
        if cfg_dir is None:
            cfg_dir = os.environ.get('CFG_OUTPUT_DIR', 'cfg_output_specimin')
        self.cfg_dir = cfg_dir
        
        # Pattern for CFG directories with transformations disabled
        # e.g., "cfg_output_ablate_{transform}" or "{base}/ablate_{transform}"
        self.cfg_dir_base_pattern = cfg_dir_base_pattern
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'episodes': episodes,
                'device': device,
                'total_transformations': len(ALL_TRANSFORMATIONS)
            },
            'baseline': {},
            'ablations': {}
        }
    
    def train_all_models_baseline(self) -> Dict[str, Any]:
        """Train all models with all transformations (baseline)"""
        logger.info("=" * 80)
        logger.info("BASELINE: Training all models with all transformations enabled")
        logger.info("=" * 80)
        
        # Use the unified ablation study script
        from run_unified_ablation_study import UnifiedAblationStudy
        
        study = UnifiedAblationStudy(
            output_dir=str(self.output_dir / 'baseline'),
            balanced_dataset_dir=str(self.balanced_dataset_dir),
            cfg_dir=self.cfg_dir,
            episodes=self.episodes,
            device=self.device
        )
        
        baseline_results = study.run_baseline_study()
        self.results['baseline'] = baseline_results
        
        return baseline_results
    
    def generate_dataset_for_transform_ablation(self, transform_name: str, cfg_dir: str, 
                                                output_dataset_dir: Path,
                                                examples_per_annotation: int = 2000) -> bool:
        """
        Generate dataset for a specific transformation ablation
        
        Args:
            transform_name: Name of the transformation being ablated
            cfg_dir: CFG directory with this transformation disabled
            output_dataset_dir: Output directory for the generated dataset
            examples_per_annotation: Number of examples per annotation type
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Generating dataset for {transform_name} ablation...")
        
        if not os.path.exists(cfg_dir):
            logger.error(f"CFG directory does not exist: {cfg_dir}")
            logger.error(f"Please ensure CFGs are generated with {transform_name} disabled")
            return False
        
        from ablation_dataset_generator import AblationDatasetGenerator
        
        generator = AblationDatasetGenerator(random_seed=42)
        success, error = generator.generate_dataset(
            cfg_dir=cfg_dir,
            output_dir=str(output_dataset_dir),
            examples_per_annotation=examples_per_annotation,
            target_balance=0.5,
            timeout=3600
        )
        
        if not success:
            logger.error(f"Failed to generate dataset for {transform_name} ablation: {error}")
            return False
        
        logger.info(f"Successfully generated dataset for {transform_name} ablation")
        return True
    
    def train_with_disabled_transform(self, transform: str) -> Dict[str, Any]:
        """Train all models with a specific transformation disabled"""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ABLATION: Training with '{transform}' disabled")
        logger.info(f"{'=' * 80}")
        
        transform_dir = self.output_dir / f'ablate_{transform}'
        transform_dir.mkdir(exist_ok=True)
        
        # Determine CFG directory for this transformation
        if self.cfg_dir_base_pattern:
            # Use pattern to construct CFG directory path
            try:
                cfg_dir_for_transform = self.cfg_dir_base_pattern.format(transform=transform)
            except KeyError:
                # Pattern doesn't use {transform}, try direct substitution
                cfg_dir_for_transform = self.cfg_dir_base_pattern.replace('{transform}', transform)
        else:
            # Try default pattern
            cfg_dir_for_transform = f'cfg_output_ablate_{transform}'
            if not os.path.exists(cfg_dir_for_transform):
                # Try alternative pattern
                cfg_dir_for_transform = os.path.join('ablation_studies', f'ablate_{transform}', 'cfg_output')
        
        # Generate dataset for this transformation
        dataset_dir = transform_dir / 'datasets'
        dataset_dir.mkdir(exist_ok=True)
        
        logger.info(f"Generating dataset from CFG directory: {cfg_dir_for_transform}")
        dataset_generated = self.generate_dataset_for_transform_ablation(
            transform_name=transform,
            cfg_dir=cfg_dir_for_transform,
            output_dataset_dir=dataset_dir,
            examples_per_annotation=2000
        )
        
        if not dataset_generated:
            logger.warning(f"Failed to generate dataset for {transform}.")
            logger.warning(f"CFG directory {cfg_dir_for_transform} does not exist.")
            logger.warning(f"Skipping {transform} ablation. This transformation requires CFGs generated with it disabled.")
            return {
                'transform': transform,
                'success': False,
                'error': f'CFG directory not found: {cfg_dir_for_transform}',
                'note': 'CFG directory with this transformation disabled must be generated first'
            }
        
        results = {}
        total = len(ALL_MODELS) * len(ANNOTATION_TYPES)
        completed = 0
        
        # Use unified study to train with the generated dataset
        from run_unified_ablation_study import UnifiedAblationStudy
        
        study = UnifiedAblationStudy(
            output_dir=str(transform_dir),
            balanced_dataset_dir=str(dataset_dir),
            cfg_dir=cfg_dir_for_transform,
            episodes=self.episodes,
            device=self.device
        )
        
        # Train all models (using existing dataset - limitation noted)
        for base_model in ALL_MODELS:
            for annotation_type in ANNOTATION_TYPES:
                key = f"{annotation_type}_{base_model}"
                completed += 1
                
                logger.info(f"[{completed}/{total}] {transform} ablation: {key}...")
                
                if base_model in ['gcn', 'hgt', 'gcsn']:
                    result = study.train_graph_based_model(annotation_type, base_model)
                else:
                    result = study.train_feature_based_model(annotation_type, base_model)
                
                results[key] = result
                
                if result.get('success'):
                    stats = result.get('training_stats', {})
                    val_acc = stats.get('val_accuracy', 'N/A')
                    logger.info(f"✅ {key}: Val Acc={val_acc}")
        
        return results
    
    def run_full_transformation_ablation(self, transformations: List[str] = None) -> Dict[str, Any]:
        """Run full transformation ablation study"""
        if transformations is None:
            transformations = ALL_TRANSFORMATIONS
        
        logger.info("=" * 80)
        logger.info(f"TRANSFORMATION ABLATION STUDY")
        logger.info(f"Testing {len(transformations)} transformations")
        logger.info("=" * 80)
        
        # Run baseline
        self.train_all_models_baseline()
        
        # Run ablations for each transformation
        for i, transform in enumerate(transformations, 1):
            logger.info(f"\n[{i}/{len(transformations)}] Ablating: {transform}")
            ablation_results = self.train_with_disabled_transform(transform)
            self.results['ablations'][transform] = ablation_results
        
        # Calculate comparison
        self._calculate_comparison()
        
        # Save results
        results_file = self.output_dir / 'transformation_ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        self._print_summary()
        
        return self.results
    
    def _calculate_comparison(self):
        """Calculate impact of each transformation"""
        baseline = self.results.get('baseline', {})
        ablations = self.results.get('ablations', {})
        
        # Calculate baseline average
        baseline_successful = [r for r in baseline.values() if r.get('success')]
        baseline_val_accs = [r.get('training_stats', {}).get('val_accuracy')
                             for r in baseline_successful
                             if r.get('training_stats', {}).get('val_accuracy') is not None]
        baseline_avg = sum(baseline_val_accs) / len(baseline_val_accs) if baseline_val_accs else None
        
        comparison = {
            'baseline_average_val_accuracy': baseline_avg,
            'transformation_impact': {}
        }
        
        if baseline_avg:
            for transform, results in ablations.items():
                # Check if results is a dict and has model results (not just an error dict)
                if not isinstance(results, dict):
                    logger.warning(f"Skipping {transform}: results is not a dict")
                    continue
                
                # Check if this is an error dict (has 'error' or 'transform' key but not model results)
                if 'error' in results and 'transform' in results:
                    # This is a skipped transformation, not model results
                    comparison['transformation_impact'][transform] = {
                        'skipped': True,
                        'reason': results.get('error', 'Unknown error'),
                        'note': results.get('note', 'CFG directory with this transformation disabled must be generated first')
                    }
                    continue
                
                # Process model results
                successful = [r for r in results.values() if isinstance(r, dict) and r.get('success')]
                val_accs = [r.get('training_stats', {}).get('val_accuracy')
                           for r in successful
                           if isinstance(r, dict) and r.get('training_stats', {}).get('val_accuracy') is not None]
                
                if val_accs:
                    avg_val_acc = sum(val_accs) / len(val_accs)
                    impact = baseline_avg - avg_val_acc  # Positive = performance loss
                    
                    comparison['transformation_impact'][transform] = {
                        'average_val_accuracy': avg_val_acc,
                        'impact': impact,
                        'percent_change': (impact / baseline_avg * 100) if baseline_avg > 0 else 0,
                        'successful_trainings': len(successful)
                    }
                else:
                    # No valid accuracy values found
                    comparison['transformation_impact'][transform] = {
                        'skipped': False,
                        'no_results': True,
                        'note': 'No valid accuracy values found in results'
                    }
        
        self.results['comparison'] = comparison
        return comparison
    
    def _print_summary(self):
        """Print summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION ABLATION SUMMARY")
        logger.info("=" * 80)
        
        comparison = self.results.get('comparison', {})
        baseline_avg = comparison.get('baseline_average_val_accuracy')
        
        if baseline_avg:
            logger.info(f"\nBaseline Average Val Accuracy: {baseline_avg:.4f}")
            
            impact = comparison.get('transformation_impact', {})
            if impact:
                logger.info("\nTop 5 Most Impactful Transformations (by performance loss):")
                sorted_impact = sorted(impact.items(),
                                     key=lambda x: x[1].get('impact', 0),
                                     reverse=True)[:5]
                for transform, data in sorted_impact:
                    logger.info(f"  {transform}: {data.get('impact', 0):.4f} ({data.get('percent_change', 0):.2f}%)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run transformation ablation study'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_transformations_final',
        help='Output directory'
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
        help='CFG directory (for baseline, used by graph models)'
    )
    parser.add_argument(
        '--cfg_dir_base_pattern',
        type=str,
        default=None,
        help='Pattern for CFG directories with transformations disabled. Use {transform} placeholder, e.g., "cfg_output_ablate_{transform}"'
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
        '--baseline_only',
        action='store_true',
        help='Run only baseline'
    )
    
    args = parser.parse_args()
    
    study = TransformationAblationFinal(
        output_dir=args.output_dir,
        balanced_dataset_dir=args.balanced_dataset_dir,
        cfg_dir=args.cfg_dir,
        cfg_dir_base_pattern=args.cfg_dir_base_pattern,
        episodes=args.episodes,
        device=args.device
    )
    
    if args.baseline_only:
        study.train_all_models_baseline()
        results_file = study.output_dir / 'baseline_results.json'
        with open(results_file, 'w') as f:
            json.dump(study.results, f, indent=2)
    else:
        study.run_full_transformation_ablation(transformations=args.transformations)
    
    return 0


if __name__ == '__main__':
    exit(main())

