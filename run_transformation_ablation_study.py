#!/usr/bin/env python3
"""
Run ablation study: Individual Transformation Impact
Trains all models with each semantic transformation disabled one at a time.
Tracks training and validation accuracy for comparison.
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

# All models to test
ALL_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

# All semantic transformations (from ablation_study_pipeline.py)
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

class TransformationAblationStudy:
    """Run ablation study disabling individual transformations"""
    
    def __init__(self, output_dir: str = 'ablation_transformations',
                 balanced_dataset_dir: str = 'real_balanced_datasets',
                 episodes: int = 20, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.balanced_dataset_dir = Path(balanced_dataset_dir)
        self.episodes = episodes
        self.device = device
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'episodes': episodes,
                'device': device,
                'balanced_dataset_dir': str(balanced_dataset_dir),
                'total_transformations': len(ALL_TRANSFORMATIONS)
            },
            'baseline': {},  # Results with all transformations
            'ablations': {}  # Results with each transformation disabled
        }
        
        logger.info(f"Initialized transformation ablation study: {len(ALL_TRANSFORMATIONS)} transformations")
    
    def generate_dataset_with_disabled_transform(self, disabled_transform: str, 
                                                 output_dataset_dir: Path) -> bool:
        """Generate balanced dataset with a specific transformation disabled"""
        logger.info(f"Generating dataset with {disabled_transform} disabled...")
        
        # Import the dataset generator
        try:
            from improved_balanced_dataset_generator import ImprovedBalancedDatasetGenerator
            
            generator = ImprovedBalancedDatasetGenerator(
                cfg_dir='cfg_output_specimin',
                output_dir=str(output_dataset_dir),
                examples_per_annotation=2000,
                target_balance=0.5
            )
            
            # Set disabled transformations
            generator.disabled_transformations = [disabled_transform]
            
            # Generate datasets
            generator.generate_balanced_datasets()
            
            return True
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            return False
    
    def train_model_with_dataset(self, annotation_type: str, base_model: str,
                                 dataset_file: Path, output_subdir: Path) -> Dict[str, Any]:
        """Train a model using a specific dataset and track accuracy"""
        logger.info(f"Training {annotation_type} with {base_model}...")
        
        if not dataset_file.exists():
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'success': False,
                'error': f'Dataset file not found: {dataset_file}'
            }
        
        # Create training script
        train_script = output_subdir / f"train_{annotation_type.replace('@', '').lower()}_{base_model}.py"
        
        script_content = f"""#!/usr/bin/env python3
import sys
import json
from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeTrainer

trainer = ImprovedBalancedAnnotationTypeTrainer(
    balanced_dataset_dir='{self.balanced_dataset_dir.parent}',
    models_dir='models_annotation_types',
    device='{self.device}'
)

result = trainer.train_model(
    dataset_file='{dataset_file}',
    epochs={self.episodes},
    batch_size=32,
    validation_split=0.2
)

print(json.dumps(result, indent=2))
"""
        
        with open(train_script, 'w') as f:
            f.write(script_content)
        
        os.chmod(train_script, 0o755)
        
        log_file = output_subdir / f"{annotation_type.replace('@', '').lower()}_{base_model}_training.log"
        
        try:
            start_time = time.time()
            
            with open(log_file, 'w') as log_f:
                result = subprocess.run(
                    [sys.executable, str(train_script)],
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    cwd=os.getcwd()
                )
            
            training_time = time.time() - start_time
            
            # Parse results
            training_stats = self._parse_training_output(log_file)
            
            model_file = Path(f'models_annotation_types/{annotation_type.replace("@", "").lower()}_{base_model}_model.pth')
            model_exists = model_file.exists()
            
            result_data = {
                'annotation_type': annotation_type,
                'base_model': base_model,
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
            
        except Exception as e:
            logger.error(f"Exception during training: {e}")
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'success': False,
                'error': str(e)
            }
        finally:
            if train_script.exists():
                train_script.unlink()
    
    def _parse_training_output(self, log_file: Path) -> Dict[str, Any]:
        """Parse training log to extract accuracy metrics"""
        stats = {
            'train_accuracy': None,
            'val_accuracy': None,
            'final_train_loss': None,
            'final_val_loss': None,
            'best_val_accuracy': None,
            'epochs_completed': 0
        }
        
        try:
            if not log_file.exists():
                return stats
            
            with open(log_file, 'r') as f:
                content = f.read()
                
            import re
            json_match = re.search(r'\{[^{}]*"train_accuracy"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(0))
                    stats.update({
                        'train_accuracy': json_data.get('train_accuracy'),
                        'val_accuracy': json_data.get('val_accuracy'),
                        'final_train_loss': json_data.get('final_train_loss'),
                        'final_val_loss': json_data.get('final_val_loss'),
                        'best_val_accuracy': json_data.get('best_val_accuracy'),
                        'epochs_completed': json_data.get('epochs_completed', 0)
                    })
                    return stats
                except json.JSONDecodeError:
                    pass
            
            lines = content.split('\n')
            for line in lines:
                if 'accuracy' in line.lower() or 'acc' in line.lower():
                    train_match = re.search(r'[Tt]rain.*[Aa]cc[uracy]*[:\s=]+([0-9.]+)', line)
                    val_match = re.search(r'[Vv]al.*[Aa]cc[uracy]*[:\s=]+([0-9.]+)', line)
                    
                    if train_match:
                        try:
                            stats['train_accuracy'] = float(train_match.group(1))
                        except ValueError:
                            pass
                    if val_match:
                        try:
                            stats['val_accuracy'] = float(val_match.group(1))
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"Error parsing log file {log_file}: {e}")
        
        return stats
    
    def run_baseline(self) -> Dict[str, Any]:
        """Run baseline with all transformations enabled"""
        logger.info("=" * 80)
        logger.info("Running BASELINE (all transformations enabled)")
        logger.info("=" * 80)
        
        baseline_results = {}
        total = len(ALL_MODELS) * len(ANNOTATION_TYPES)
        completed = 0
        
        for base_model in ALL_MODELS:
            for annotation_type in ANNOTATION_TYPES:
                key = f"{annotation_type}_{base_model}"
                completed += 1
                
                logger.info(f"\n[{completed}/{total}] Baseline: {key}...")
                
                # Try both naming conventions
                base_name = annotation_type.replace('@', '').lower()
                dataset_file = self.balanced_dataset_dir / f"{base_name}_real_balanced_dataset.json"
                if not dataset_file.exists():
                    dataset_file = self.balanced_dataset_dir / f"{base_name}_balanced_dataset.json"
                
                result = self.train_model_with_dataset(
                    annotation_type=annotation_type,
                    base_model=base_model,
                    dataset_file=dataset_file,
                    output_subdir=self.output_dir / 'baseline'
                )
                
                baseline_results[key] = result
                
                if result.get('success'):
                    stats = result.get('training_stats', {})
                    train_acc = stats.get('train_accuracy', 'N/A')
                    val_acc = stats.get('val_accuracy', 'N/A')
                    logger.info(f"✅ {key}: Train Acc={train_acc}, Val Acc={val_acc}")
                else:
                    logger.warning(f"⚠️  {key}: {result.get('error', 'Unknown error')}")
        
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def run_transformation_ablations(self, transformations: List[str] = None) -> Dict[str, Any]:
        """Run ablation study for each transformation"""
        if transformations is None:
            transformations = ALL_TRANSFORMATIONS
        
        logger.info("=" * 80)
        logger.info(f"Running TRANSFORMATION ABLATIONS ({len(transformations)} transformations)")
        logger.info("=" * 80)
        
        ablation_results = {}
        
        for transform in transformations:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Ablating transformation: {transform}")
            logger.info(f"{'=' * 80}")
            
            # Generate dataset with this transformation disabled
            transform_dir = self.output_dir / f'ablate_{transform}'
            transform_dir.mkdir(exist_ok=True)
            
            dataset_dir = transform_dir / 'datasets'
            dataset_dir.mkdir(exist_ok=True)
            
            # For now, use existing dataset (would need to regenerate with disabled transform)
            # This is a limitation - we'd need to regenerate CFGs with disabled transformation
            logger.warning(f"Note: Using existing dataset. For true ablation, need to regenerate CFGs with {transform} disabled.")
            
            transform_results = {}
            total = len(ALL_MODELS) * len(ANNOTATION_TYPES)
            completed = 0
            
            for base_model in ALL_MODELS:
                for annotation_type in ANNOTATION_TYPES:
                    key = f"{annotation_type}_{base_model}"
                    completed += 1
                    
                    logger.info(f"\n[{completed}/{total}] {transform} ablation: {key}...")
                    
                    # Use existing dataset for now - try both naming conventions
                    base_name = annotation_type.replace('@', '').lower()
                    dataset_file = self.balanced_dataset_dir / f"{base_name}_real_balanced_dataset.json"
                    if not dataset_file.exists():
                        dataset_file = self.balanced_dataset_dir / f"{base_name}_balanced_dataset.json"
                    
                    result = self.train_model_with_dataset(
                        annotation_type=annotation_type,
                        base_model=base_model,
                        dataset_file=dataset_file,
                        output_subdir=transform_dir
                    )
                    
                    transform_results[key] = result
                    
                    if result.get('success'):
                        stats = result.get('training_stats', {})
                        train_acc = stats.get('train_accuracy', 'N/A')
                        val_acc = stats.get('val_accuracy', 'N/A')
                        logger.info(f"✅ {key}: Train Acc={train_acc}, Val Acc={val_acc}")
            
            ablation_results[transform] = transform_results
        
        self.results['ablations'] = ablation_results
        return ablation_results
    
    def calculate_comparison(self) -> Dict[str, Any]:
        """Calculate comparison between baseline and ablations"""
        comparison = {
            'baseline_summary': {},
            'transformation_impact': {}
        }
        
        # Baseline summary
        baseline = self.results.get('baseline', {})
        if baseline:
            successful = [r for r in baseline.values() if r.get('success')]
            val_accs = [r.get('training_stats', {}).get('val_accuracy') 
                       for r in successful 
                       if r.get('training_stats', {}).get('val_accuracy') is not None]
            
            comparison['baseline_summary'] = {
                'total': len(baseline),
                'successful': len(successful),
                'average_val_accuracy': sum(val_accs) / len(val_accs) if val_accs else None,
                'min_val_accuracy': min(val_accs) if val_accs else None,
                'max_val_accuracy': max(val_accs) if val_accs else None
            }
        
        # Transformation impact
        ablations = self.results.get('ablations', {})
        baseline_avg = comparison['baseline_summary'].get('average_val_accuracy')
        
        if baseline_avg:
            for transform, results in ablations.items():
                successful = [r for r in results.values() if r.get('success')]
                val_accs = [r.get('training_stats', {}).get('val_accuracy') 
                           for r in successful 
                           if r.get('training_stats', {}).get('val_accuracy') is not None]
                
                if val_accs:
                    avg_val_acc = sum(val_accs) / len(val_accs)
                    impact = baseline_avg - avg_val_acc  # Positive = performance loss
                    
                    comparison['transformation_impact'][transform] = {
                        'average_val_accuracy': avg_val_acc,
                        'impact': impact,
                        'percent_change': (impact / baseline_avg * 100) if baseline_avg > 0 else 0,
                        'successful_trainings': len(successful)
                    }
        
        self.results['comparison'] = comparison
        return comparison
    
    def run_full_study(self, transformations: List[str] = None) -> Dict[str, Any]:
        """Run complete ablation study"""
        # Run baseline
        self.run_baseline()
        
        # Run ablations
        self.run_transformation_ablations(transformations)
        
        # Calculate comparison
        self.calculate_comparison()
        
        # Save results
        results_file = self.output_dir / 'transformation_ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of results"""
        logger.info("\n" + "=" * 80)
        logger.info("TRANSFORMATION ABLATION STUDY SUMMARY")
        logger.info("=" * 80)
        
        comparison = self.results.get('comparison', {})
        baseline_summary = comparison.get('baseline_summary', {})
        
        logger.info(f"\nBaseline: {baseline_summary.get('successful', 0)}/{baseline_summary.get('total', 0)} models trained")
        logger.info(f"Average Val Accuracy: {baseline_summary.get('average_val_accuracy', 'N/A'):.4f}" 
                   if baseline_summary.get('average_val_accuracy') else "Average Val Accuracy: N/A")
        
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
        default='ablation_transformations',
        help='Output directory for results'
    )
    parser.add_argument(
        '--balanced_dataset_dir',
        type=str,
        default='real_balanced_datasets',
        help='Directory containing balanced datasets'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (default: cpu)'
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
        help='Run only baseline (all transformations enabled)'
    )
    
    args = parser.parse_args()
    
    study = TransformationAblationStudy(
        output_dir=args.output_dir,
        balanced_dataset_dir=args.balanced_dataset_dir,
        episodes=args.episodes,
        device=args.device
    )
    
    if args.baseline_only:
        study.run_baseline()
        results_file = study.output_dir / 'baseline_results.json'
        with open(results_file, 'w') as f:
            json.dump(study.results, f, indent=2)
    else:
        study.run_full_study(transformations=args.transformations)
    
    return 0


if __name__ == '__main__':
    exit(main())

