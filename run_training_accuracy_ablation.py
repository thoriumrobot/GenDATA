#!/usr/bin/env python3
"""
Run ablation study comparing training and validation accuracy with and without augmentation.

This script trains all models and tracks training/validation accuracy metrics.
For the "no augmentation" case, it uses CFGs from the original (non-augmented) dataset.
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

class TrainingAccuracyAblation:
    """Run ablation study tracking training and validation accuracy"""
    
    def __init__(self, output_dir: str = 'ablation_training_accuracy', 
                 balanced_dataset_dir: str = 'real_balanced_datasets',
                 episodes: int = 50, device: str = 'cpu'):
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
                'balanced_dataset_dir': str(balanced_dataset_dir)
            },
            'models': {}
        }
        
        logger.info(f"Initialized ablation study: output_dir={output_dir}, episodes={episodes}")
    
    def train_model_and_track_accuracy(self, annotation_type: str, base_model: str) -> Dict[str, Any]:
        """Train a model and extract training/validation accuracy"""
        logger.info(f"Training {annotation_type} with {base_model}...")
        
        # Use improved balanced trainer which tracks accuracy
        # Try both naming conventions
        base_name = annotation_type.replace('@', '').lower()
        dataset_file = self.balanced_dataset_dir / f"{base_name}_real_balanced_dataset.json"
        if not dataset_file.exists():
            dataset_file = self.balanced_dataset_dir / f"{base_name}_balanced_dataset.json"
        
        if not dataset_file.exists():
            logger.warning(f"Dataset file not found: {dataset_file}")
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'success': False,
                'error': f'Dataset file not found: {dataset_file}'
            }
        
        # Create a temporary training script
        train_script = self.output_dir / f"train_{annotation_type.replace('@', '').lower()}_{base_model}.py"
        
        script_content = f"""#!/usr/bin/env python3
import sys
import json
from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeTrainer

# Train model
trainer = ImprovedBalancedAnnotationTypeTrainer(
    balanced_dataset_dir='{self.balanced_dataset_dir}',
    models_dir='models_annotation_types',
    device='{self.device}'
)

# Train the model
result = trainer.train_model(
    dataset_file='{dataset_file}',
    epochs={self.episodes},
    batch_size=32,
    validation_split=0.2
)

# Output results as JSON
print(json.dumps(result, indent=2))
"""
        
        with open(train_script, 'w') as f:
            f.write(script_content)
        
        os.chmod(train_script, 0o755)
        
        # Run training
        log_file = self.output_dir / f"{annotation_type.replace('@', '').lower()}_{base_model}_training.log"
        
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
            
            # Parse results from log
            training_stats = self._parse_training_output(log_file)
            
            # Check if model was saved
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
            
        except subprocess.TimeoutExpired:
            logger.error(f"Training timeout for {annotation_type} with {base_model}")
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'success': False,
                'error': 'Training timeout'
            }
        except Exception as e:
            logger.error(f"Exception during training: {e}")
            return {
                'annotation_type': annotation_type,
                'base_model': base_model,
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up temp script
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
                
            # Try to parse JSON output if present
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
            
            # Fallback: parse text output
            lines = content.split('\n')
            for line in lines:
                # Look for accuracy patterns
                if 'accuracy' in line.lower() or 'acc' in line.lower():
                    import re
                    # Try to extract train and val accuracy
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
                
                # Look for loss
                if 'loss' in line.lower():
                    loss_matches = re.findall(r'[Ll]oss[:\s=]+([0-9.]+)', line)
                    if loss_matches:
                        try:
                            if 'train' in line.lower():
                                stats['final_train_loss'] = float(loss_matches[-1])
                            elif 'val' in line.lower() or 'validation' in line.lower():
                                stats['final_val_loss'] = float(loss_matches[-1])
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
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """Run the complete ablation study"""
        logger.info("=" * 80)
        logger.info("Starting Training Accuracy Ablation Study")
        logger.info("=" * 80)
        logger.info(f"Models: {', '.join(ALL_MODELS)}")
        logger.info(f"Annotation types: {', '.join(ANNOTATION_TYPES)}")
        logger.info(f"Total configurations: {len(ALL_MODELS) * len(ANNOTATION_TYPES)}")
        logger.info("=" * 80)
        
        total = len(ALL_MODELS) * len(ANNOTATION_TYPES)
        completed = 0
        
        for base_model in ALL_MODELS:
            for annotation_type in ANNOTATION_TYPES:
                key = f"{annotation_type}_{base_model}"
                completed += 1
                
                logger.info(f"\n[{completed}/{total}] Training {key}...")
                
                result = self.train_model_and_track_accuracy(
                    annotation_type=annotation_type,
                    base_model=base_model
                )
                
                self.results['models'][key] = result
                
                if result.get('success'):
                    stats = result.get('training_stats', {})
                    train_acc = stats.get('train_accuracy', 'N/A')
                    val_acc = stats.get('val_accuracy', 'N/A')
                    logger.info(f"✅ {key}: Train Acc={train_acc}, Val Acc={val_acc}")
                else:
                    logger.warning(f"⚠️  {key}: {result.get('error', 'Unknown error')}")
        
        # Calculate summary statistics
        self._calculate_summary()
        
        # Save results
        results_file = self.output_dir / 'ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _calculate_summary(self):
        """Calculate summary statistics"""
        successful = [r for r in self.results['models'].values() if r.get('success')]
        
        if not successful:
            self.results['summary'] = {'error': 'No successful training runs'}
            return
        
        # Calculate average accuracies
        train_accs = [r.get('training_stats', {}).get('train_accuracy') 
                      for r in successful 
                      if r.get('training_stats', {}).get('train_accuracy') is not None]
        val_accs = [r.get('training_stats', {}).get('val_accuracy') 
                    for r in successful 
                    if r.get('training_stats', {}).get('val_accuracy') is not None]
        
        self.results['summary'] = {
            'total_configurations': len(self.results['models']),
            'successful_trainings': len(successful),
            'failed_trainings': len(self.results['models']) - len(successful),
            'average_train_accuracy': sum(train_accs) / len(train_accs) if train_accs else None,
            'average_val_accuracy': sum(val_accs) / len(val_accs) if val_accs else None,
            'min_val_accuracy': min(val_accs) if val_accs else None,
            'max_val_accuracy': max(val_accs) if val_accs else None
        }
    
    def _print_summary(self):
        """Print summary of results"""
        logger.info("\n" + "=" * 80)
        logger.info("ABLATION STUDY SUMMARY")
        logger.info("=" * 80)
        
        summary = self.results.get('summary', {})
        logger.info(f"\nSuccessful trainings: {summary.get('successful_trainings', 0)}/{summary.get('total_configurations', 0)}")
        logger.info(f"Average Train Accuracy: {summary.get('average_train_accuracy', 'N/A'):.4f}" if summary.get('average_train_accuracy') else "Average Train Accuracy: N/A")
        logger.info(f"Average Val Accuracy: {summary.get('average_val_accuracy', 'N/A'):.4f}" if summary.get('average_val_accuracy') else "Average Val Accuracy: N/A")
        
        logger.info("\nPer-Model Results:")
        for key, result in sorted(self.results['models'].items()):
            if result.get('success'):
                stats = result.get('training_stats', {})
                train_acc = stats.get('train_accuracy', 'N/A')
                val_acc = stats.get('val_accuracy', 'N/A')
                logger.info(f"  {key}: Train={train_acc}, Val={val_acc}")
            else:
                logger.info(f"  {key}: FAILED - {result.get('error', 'Unknown')}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run ablation study tracking training and validation accuracy'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='ablation_training_accuracy',
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
        default=50,
        help='Number of training epochs (default: 50)'
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
    study = TrainingAccuracyAblation(
        output_dir=args.output_dir,
        balanced_dataset_dir=args.balanced_dataset_dir,
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

