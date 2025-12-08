#!/usr/bin/env python3
"""
Unified Ablation Study: Tracks training and validation accuracy for all models.

This script:
1. Trains all models using appropriate trainers
2. Extracts training/validation accuracy from training logs
3. Compares augmentation vs no augmentation
4. Compares individual transformation ablations
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

# All models to test
ALL_MODELS = ['gcn', 'hgt', 'gbt', 'causal', 'gcsn', 'dg2n', 'dgcrf']
ANNOTATION_TYPES = ['@Positive', '@NonNegative', '@GTENegativeOne']

# Feature-based models (use balanced dataset trainer)
FEATURE_BASED_MODELS = ['gbt', 'causal', 'enhanced_causal', 'dg2n', 'dgcrf']

# Graph-based models (use annotation_type_rl scripts)
GRAPH_BASED_MODELS = ['gcn', 'hgt', 'gcsn']

class UnifiedAblationStudy:
    """Unified ablation study that works with all model types"""
    
    def __init__(self, output_dir: str = 'ablation_unified',
                 balanced_dataset_dir: str = 'real_balanced_datasets',
                 cfg_dir: str = None,
                 cfg_dir_for_dataset: str = None,
                 episodes: int = 20, device: str = 'cpu'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.balanced_dataset_dir = Path(balanced_dataset_dir)
        self.episodes = episodes
        self.device = device
        
        if cfg_dir is None:
            cfg_dir = os.environ.get('CFG_OUTPUT_DIR', 'cfg_output_specimin')
        self.cfg_dir = cfg_dir
        
        # CFG directory for dataset generation (if different from cfg_dir)
        self.cfg_dir_for_dataset = cfg_dir_for_dataset if cfg_dir_for_dataset else cfg_dir
        
        # Results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'episodes': episodes,
                'device': device,
                'cfg_dir': str(cfg_dir),
                'balanced_dataset_dir': str(balanced_dataset_dir)
            },
            'baseline': {},
            'ablations': {}
        }
        
        logger.info(f"Initialized unified ablation study")
    
    def generate_dataset_if_needed(self, cfg_dir: str, output_dataset_dir: Path,
                                  examples_per_annotation: int = 2000) -> bool:
        """
        Generate dataset if it doesn't exist or is incomplete
        
        Args:
            cfg_dir: CFG directory to generate dataset from
            output_dataset_dir: Output directory for the dataset
            examples_per_annotation: Number of examples per annotation type
            
        Returns:
            True if dataset exists or was successfully generated, False otherwise
        """
        from ablation_dataset_generator import AblationDatasetGenerator
        
        generator = AblationDatasetGenerator(random_seed=42)
        
        # Check if dataset already exists and is complete
        if generator.verify_dataset_exists(str(output_dataset_dir)):
            logger.info(f"Dataset already exists in {output_dataset_dir}")
            return True
        
        # Generate dataset
        logger.info(f"Generating dataset from CFG directory: {cfg_dir}")
        logger.info(f"Output directory: {output_dataset_dir}")
        
        success, error = generator.generate_dataset(
            cfg_dir=cfg_dir,
            output_dir=str(output_dataset_dir),
            examples_per_annotation=examples_per_annotation,
            target_balance=0.5,
            timeout=3600
        )
        
        if not success:
            logger.error(f"Failed to generate dataset: {error}")
            return False
        
        return True
    
    def train_feature_based_model(self, annotation_type: str, base_model: str) -> Dict[str, Any]:
        """Train a feature-based model using improved_balanced_annotation_type_trainer"""
        logger.info(f"Training feature-based model: {annotation_type} with {base_model}")
        
        # Ensure dataset exists, generate if needed
        base_name = annotation_type.replace('@', '').lower()
        dataset_file = self.balanced_dataset_dir / f"{base_name}_real_balanced_dataset.json"
        
        if not dataset_file.exists():
            # Try alternative naming
            dataset_file = self.balanced_dataset_dir / f"{base_name}_balanced_dataset.json"
        
        if not dataset_file.exists():
            # Dataset doesn't exist, try to generate it
            logger.info(f"Dataset file not found: {dataset_file}")
            logger.info("Attempting to generate dataset from CFG directory...")
            
            dataset_generated = self.generate_dataset_if_needed(
                cfg_dir=self.cfg_dir_for_dataset,
                output_dataset_dir=self.balanced_dataset_dir,
                examples_per_annotation=2000
            )
            
            if not dataset_generated:
                return {
                    'annotation_type': annotation_type,
                    'base_model': base_model,
                    'success': False,
                    'error': f'Dataset file not found and generation failed: {self.balanced_dataset_dir}'
                }
            
            # Try again after generation
            dataset_file = self.balanced_dataset_dir / f"{base_name}_real_balanced_dataset.json"
            if not dataset_file.exists():
                dataset_file = self.balanced_dataset_dir / f"{base_name}_balanced_dataset.json"
            
            if not dataset_file.exists():
                return {
                    'annotation_type': annotation_type,
                    'base_model': base_model,
                    'success': False,
                    'error': f'Dataset file still not found after generation: {dataset_file}'
                }
        
        # Create training script
        train_script = self.output_dir / f"train_{base_name}_{base_model}.py"
        log_file = self.output_dir / f"{base_name}_{base_model}_training.log"
        
        script_content = f"""#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
sys.path.insert(0, os.getcwd())

from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeTrainer

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

trainer = ImprovedBalancedAnnotationTypeTrainer(device='{self.device}')

result = trainer.train_model(
    dataset_file='{dataset_file}',
    epochs={self.episodes},
    batch_size=32,
    validation_split=0.2
)

# Convert numpy types to native Python types
result_clean = json.loads(json.dumps(result, cls=NumpyEncoder, default=str))
print(json.dumps(result_clean, indent=2))
"""
        
        with open(train_script, 'w') as f:
            f.write(script_content)
        os.chmod(train_script, 0o755)
        
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
            training_stats = self._parse_training_output(log_file)
            
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'training_time': training_time,
                'success': result.returncode == 0,
                'log_file': str(log_file),
                'training_stats': training_stats
            }
        except Exception as e:
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'success': False,
                'error': str(e)
            }
        finally:
            if train_script.exists():
                train_script.unlink()
    
    def train_graph_based_model(self, annotation_type: str, base_model: str) -> Dict[str, Any]:
        """Train a graph-based model using annotation_type_rl scripts"""
        logger.info(f"Training graph-based model: {annotation_type} with {base_model}")
        
        # Determine script
        script_map = {
            '@Positive': 'annotation_type_rl_positive.py',
            '@NonNegative': 'annotation_type_rl_nonnegative.py',
            '@GTENegativeOne': 'annotation_type_rl_gtenegativeone.py'
        }
        script = script_map.get(annotation_type)
        if not script:
            return {'success': False, 'error': f'Unknown annotation type: {annotation_type}'}
        
        log_file = self.output_dir / f"{annotation_type.replace('@', '').lower()}_{base_model}_training.log"
        
        env = os.environ.copy()
        env['CFG_OUTPUT_DIR'] = self.cfg_dir
        env['PREDICTION_CFG_DIR'] = self.cfg_dir
        
        cmd = [
            sys.executable, script,
            '--mode', 'train',
            '--base_model', base_model,
            '--episodes', str(self.episodes),
            '--device', self.device,
            '--cfg_dir', self.cfg_dir
        ]
        
        try:
            start_time = time.time()
            with open(log_file, 'w') as log_f:
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    cwd=os.getcwd()
                )
            
            training_time = time.time() - start_time
            training_stats = self._parse_training_output(log_file)
            
            # Check if model was saved
            model_file = Path(f'models_annotation_types/{annotation_type.replace("@", "").lower()}_{base_model}_model.pth')
            model_exists = model_file.exists()
            
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'training_time': training_time,
                'success': result.returncode == 0 and model_exists,
                'model_file': str(model_file) if model_exists else None,
                'log_file': str(log_file),
                'training_stats': training_stats
            }
        except Exception as e:
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'success': False,
                'error': str(e)
            }
    
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
            
            # Try to parse JSON output first
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
            
            # Parse text output
            lines = content.split('\n')
            for line in lines:
                # Look for accuracy patterns - improved matching
                if 'accuracy' in line.lower() or 'acc' in line.lower():
                    # Match "Best validation accuracy: 90.25 percent"
                    best_val_match = re.search(r'[Bb]est\s+validation\s+accuracy[:\s]+([0-9.]+)\s+percent', line, re.IGNORECASE)
                    # Match "Training completed - Train Acc: 0.9850, Val Acc: 0.9850, Best Val Acc: 0.9850"
                    train_val_match = re.search(r'[Tt]raining\s+completed.*[Tt]rain\s+[Aa]cc[:\s=]+([0-9.]+).*[Vv]al\s+[Aa]cc[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                    # Match "Train Acc: X.XX percent" or "Train Acc=X.XX" or "Train Acc: X.XX"
                    train_match = re.search(r'[Tt]rain.*[Aa]cc[uracy]*[:\s=]+([0-9.]+)', line)
                    # Match "Val Acc: X.XX percent" or "Val Acc=X.XX" or "Val Acc: X.XX"
                    val_match = re.search(r'[Vv]al.*[Aa]cc[uracy]*[:\s=]+([0-9.]+)', line)
                    
                    # Try to extract from "Training completed" line
                    if train_val_match:
                        try:
                            train_acc_val = float(train_val_match.group(1))
                            val_acc_val = float(train_val_match.group(2))
                            if train_acc_val <= 1.0:  # Already normalized
                                stats['train_accuracy'] = train_acc_val
                            else:  # Percentage
                                stats['train_accuracy'] = train_acc_val / 100.0
                            if val_acc_val <= 1.0:  # Already normalized
                                stats['val_accuracy'] = val_acc_val
                            else:  # Percentage
                                stats['val_accuracy'] = val_acc_val / 100.0
                        except (ValueError, IndexError):
                            pass
                    
                    if best_val_match:
                        try:
                            acc_val = float(best_val_match.group(1))
                            stats['val_accuracy'] = acc_val / 100.0
                            stats['best_val_accuracy'] = acc_val / 100.0
                        except ValueError:
                            pass
                    
                    if train_match:
                        try:
                            acc_val = float(train_match.group(1))
                            if acc_val > 1.0:  # Percentage
                                stats['train_accuracy'] = acc_val / 100.0
                            else:  # Normalized
                                stats['train_accuracy'] = acc_val
                        except ValueError:
                            pass
                    if val_match and stats.get('val_accuracy') is None:  # Only if we didn't already set it
                        try:
                            acc_val = float(val_match.group(1))
                            if acc_val > 1.0:  # Percentage
                                stats['val_accuracy'] = acc_val / 100.0
                            else:  # Normalized
                                stats['val_accuracy'] = acc_val
                        except ValueError:
                            pass
                
                # Look for loss
                if 'loss' in line.lower():
                    loss_matches = re.findall(r'[Ll]oss[:\s=]+([0-9.]+)', line)
                    if loss_matches:
                        try:
                            loss_val = float(loss_matches[-1])
                            if 'train' in line.lower():
                                stats['final_train_loss'] = loss_val
                            elif 'val' in line.lower() or 'validation' in line.lower():
                                stats['final_val_loss'] = loss_val
                        except ValueError:
                            pass
                
                # Count epochs
                if 'epoch' in line.lower():
                    epoch_matches = re.findall(r'[Ee]poch\s+(\d+)', line)
                    if epoch_matches:
                        try:
                            stats['epochs_completed'] = max(stats['epochs_completed'], int(epoch_matches[-1]))
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"Error parsing log file {log_file}: {e}")
        
        return stats
    
    def run_baseline_study(self) -> Dict[str, Any]:
        """Run baseline study with all transformations enabled"""
        logger.info("=" * 80)
        logger.info("Running BASELINE Study (All Transformations Enabled)")
        logger.info("=" * 80)
        
        baseline_results = {}
        total = len(ALL_MODELS) * len(ANNOTATION_TYPES)
        completed = 0
        
        for base_model in ALL_MODELS:
            for annotation_type in ANNOTATION_TYPES:
                key = f"{annotation_type}_{base_model}"
                completed += 1
                
                logger.info(f"\n[{completed}/{total}] Baseline: {key}...")
                
                if base_model in GRAPH_BASED_MODELS:
                    result = self.train_graph_based_model(annotation_type, base_model)
                else:
                    result = self.train_feature_based_model(annotation_type, base_model)
                
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
        """Run ablation study for each transformation (simplified - uses existing dataset)"""
        # Note: For true transformation ablation, we'd need to regenerate datasets
        # with each transformation disabled. For now, we'll note this limitation.
        logger.info("=" * 80)
        logger.info("TRANSFORMATION ABLATION STUDY")
        logger.info("=" * 80)
        logger.info("Note: This uses existing dataset. For true ablation, datasets need")
        logger.info("to be regenerated with each transformation disabled during CFG generation.")
        logger.info("=" * 80)
        
        # For now, return empty results with note
        return {
            'note': 'True transformation ablation requires regenerating datasets with each transformation disabled. This would require re-running the full pipeline (slicing, augmentation with disabled transform, CFG generation, dataset generation) for each transformation.',
            'transformations_tested': transformations or [],
            'results': {}
        }
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        baseline = self.results.get('baseline', {})
        successful = [r for r in baseline.values() if r.get('success')]
        
        if not successful:
            return {'error': 'No successful training runs'}
        
        train_accs = [r.get('training_stats', {}).get('train_accuracy') 
                     for r in successful 
                     if r.get('training_stats', {}).get('train_accuracy') is not None]
        val_accs = [r.get('training_stats', {}).get('val_accuracy') 
                   for r in successful 
                   if r.get('training_stats', {}).get('val_accuracy') is not None]
        
        summary = {
            'total_configurations': len(baseline),
            'successful_trainings': len(successful),
            'failed_trainings': len(baseline) - len(successful),
            'average_train_accuracy': sum(train_accs) / len(train_accs) if train_accs else None,
            'average_val_accuracy': sum(val_accs) / len(val_accs) if val_accs else None,
            'min_val_accuracy': min(val_accs) if val_accs else None,
            'max_val_accuracy': max(val_accs) if val_accs else None,
            'per_model_breakdown': {}
        }
        
        # Per-model breakdown
        for base_model in ALL_MODELS:
            model_results = [r for r in successful 
                           if r.get('base_model') == base_model]
            if model_results:
                model_val_accs = [r.get('training_stats', {}).get('val_accuracy')
                                for r in model_results
                                if r.get('training_stats', {}).get('val_accuracy') is not None]
                summary['per_model_breakdown'][base_model] = {
                    'successful': len(model_results),
                    'average_val_accuracy': sum(model_val_accs) / len(model_val_accs) if model_val_accs else None
                }
        
        self.results['summary'] = summary
        return summary
    
    def run_full_study(self) -> Dict[str, Any]:
        """Run complete ablation study"""
        # Run baseline
        self.run_baseline_study()
        
        # Calculate summary
        self.calculate_summary()
        
        # Save results
        results_file = self.output_dir / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of results"""
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("=" * 80)
        
        summary = self.results.get('summary', {})
        logger.info(f"\nSuccessful trainings: {summary.get('successful_trainings', 0)}/{summary.get('total_configurations', 0)}")
        
        if summary.get('average_train_accuracy'):
            logger.info(f"Average Train Accuracy: {summary['average_train_accuracy']:.4f}")
        if summary.get('average_val_accuracy'):
            logger.info(f"Average Val Accuracy: {summary['average_val_accuracy']:.4f}")
        
        breakdown = summary.get('per_model_breakdown', {})
        if breakdown:
            logger.info("\nPer-Model Average Val Accuracy:")
            for model, data in sorted(breakdown.items()):
                avg_acc = data.get('average_val_accuracy')
                if avg_acc:
                    logger.info(f"  {model}: {avg_acc:.4f}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run unified ablation study tracking training/validation accuracy'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_unified',
        help='Output directory for results'
    )
    parser.add_argument(
        '--balanced_dataset_dir',
        type=str,
        default='real_balanced_datasets',
        help='Directory containing balanced datasets'
    )
    parser.add_argument(
        '--cfg_dir',
        type=str,
        default=None,
        help='CFG directory (default: from env or cfg_output_specimin)'
    )
    parser.add_argument(
        '--cfg_dir_for_dataset',
        type=str,
        default=None,
        help='CFG directory for dataset generation (if different from cfg_dir)'
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
    
    args = parser.parse_args()
    
    study = UnifiedAblationStudy(
        output_dir=args.output_dir,
        balanced_dataset_dir=args.balanced_dataset_dir,
        cfg_dir=args.cfg_dir,
        cfg_dir_for_dataset=args.cfg_dir_for_dataset,
        episodes=args.episodes,
        device=args.device
    )
    
    results = study.run_full_study()
    
    logger.info("\n" + "=" * 80)
    logger.info("Ablation study completed!")
    logger.info(f"Results saved to: {study.output_dir / 'ablation_results.json'}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

