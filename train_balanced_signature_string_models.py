#!/usr/bin/env python3
"""
Train All Signature String Checker Models Using Balanced Datasets

This script trains all models for Signature String Checker annotations using balanced datasets:
- 7 base models × 3 annotation types = 21 models total
- @FullyQualifiedName
- @BinaryName
- @FieldDescriptor
"""

import os
import sys
import subprocess
import logging
import torch
from pathlib import Path

# Add GenDATA root to path
GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')
sys.path.insert(0, str(GEN_DATA_ROOT))

from improved_balanced_dataset_generator import ImprovedBalancedDatasetGenerator
from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancedSignatureStringModelsTrainer:
    def __init__(self, cfg_dir=None, balanced_dataset_dir=None, models_dir=None, 
                 examples_per_annotation=1000, epochs=100, batch_size=32, device='auto'):
        """
        Initialize balanced Signature String models trainer
        
        Args:
            cfg_dir: Directory containing CFG files (default: cfg_output_adaptive_specimin_signature_string/)
            balanced_dataset_dir: Directory to save balanced datasets
            models_dir: Directory to save trained models
            examples_per_annotation: Number of examples per annotation type
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to use ('auto', 'cuda', or 'cpu')
        """
        self.cfg_dir = cfg_dir or str(GEN_DATA_ROOT / 'cfg_output_adaptive_specimin_signature_string')
        self.balanced_dataset_dir = balanced_dataset_dir or str(GEN_DATA_ROOT / 'balanced_datasets_signature_string')
        self.models_dir = models_dir or str(GEN_DATA_ROOT / 'models_annotation_types_signature_string')
        self.examples_per_annotation = examples_per_annotation
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        
        # Base model types
        self.base_models = ['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']
        
        # Annotation types for Signature String
        self.annotation_types = ['@FullyQualifiedName', '@BinaryName', '@FieldDescriptor']
        
        # Create directories
        os.makedirs(self.balanced_dataset_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.trained_models = []
        self.failed_models = []

    def generate_balanced_datasets(self):
        """Generate balanced datasets for Signature String Checker"""
        logger.info("=" * 80)
        logger.info("Generating Balanced Datasets for Signature String Checker")
        logger.info("=" * 80)
        
        if not os.path.exists(self.cfg_dir):
            logger.error(f"CFG directory not found: {self.cfg_dir}")
            logger.error("Please run create_training_datasets.py --checker signature_string first")
            return False
        
        # Create generator
        generator = ImprovedBalancedDatasetGenerator(
            target_balance=0.5,
            random_seed=42,
            checker_name='signature_string'
        )
        
        # Load CFG files
        cfg_files = generator.load_cfg_files(self.cfg_dir)
        if not cfg_files:
            logger.error(f"No CFG files found in {self.cfg_dir}")
            return False
        
        # Generate balanced examples
        balanced_datasets = generator.generate_balanced_examples(
            cfg_files,
            examples_per_annotation=self.examples_per_annotation
        )
        
        # Save datasets
        generator.save_balanced_dataset(balanced_datasets, self.balanced_dataset_dir)
        generator.print_statistics()
        
        return True

    def train_all_models(self):
        """Train all Signature String models using balanced datasets"""
        logger.info("=" * 80)
        logger.info("Training All Signature String Models with Balanced Datasets")
        logger.info("=" * 80)
        
        # Check if balanced datasets exist
        dataset_files = {}
        for ann_type in self.annotation_types:
            ann_name = ann_type.replace('@', '').lower()
            dataset_file = os.path.join(self.balanced_dataset_dir, f"{ann_name}_real_balanced_dataset.json")
            if not os.path.exists(dataset_file):
                logger.error(f"Balanced dataset not found: {dataset_file}")
                logger.error("Please run generate_balanced_datasets() first")
                return False
            dataset_files[ann_type] = dataset_file
        
        total_models = len(self.base_models) * len(self.annotation_types)
        success_count = 0
        
        for ann_type in self.annotation_types:
            dataset_file = dataset_files[ann_type]
            
            for base_model in self.base_models:
                model_name = f"{ann_type.replace('@', '').lower()}_{base_model}_balanced"
                logger.info(f"\n{'='*80}")
                logger.info(f"Training {model_name}")
                logger.info(f"{'='*80}")
                
                try:
                    # Create trainer
                    trainer = ImprovedBalancedAnnotationTypeTrainer(
                        model_type=f'improved_balanced_{base_model}',
                        device=self.device
                    )
                    
                    # Load dataset to get feature dimension and annotation type
                    examples, dataset_annotation_type = trainer.load_balanced_dataset(dataset_file)
                    if not examples:
                        logger.error(f"No examples in dataset file: {dataset_file}")
                        self.failed_models.append(model_name)
                        continue
                    
                    input_dim = len(examples[0]['features'])
                    
                    # Train model
                    result = trainer.train_model(
                        dataset_file=dataset_file,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_split=0.2
                    )
                    
                    if result.get('success', False):
                        # Save model
                        ann_name = ann_type.replace('@', '').lower()
                        model_file = os.path.join(self.models_dir, f"{ann_name}_{base_model}_balanced_model.pth")
                        
                        # Save individual model - use dataset_annotation_type to access the model
                        model = trainer.models.get(dataset_annotation_type)
                        if model:
                            torch.save({
                                'model_state_dict': model.state_dict(),
                                'model_type': trainer.model_type,
                                'annotation_type': dataset_annotation_type,
                                'input_dim': input_dim,
                                'training_stats': result
                            }, model_file)
                            logger.info(f"Saved model to {model_file}")
                        else:
                            logger.warning(f"Model not found in trainer.models for {dataset_annotation_type}")
                        
                        logger.info(f"✅ Successfully trained {model_name}")
                        self.trained_models.append(model_name)
                        success_count += 1
                    else:
                        logger.error(f"❌ Failed to train {model_name}: {result.get('error', 'Unknown error')}")
                        self.failed_models.append(model_name)
                        
                except Exception as e:
                    logger.error(f"💥 Error training {model_name}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    self.failed_models.append(model_name)
        
        logger.info("=" * 80)
        logger.info(f"Training Summary: {success_count}/{total_models} models trained successfully")
        if self.failed_models:
            logger.error(f"Failed models: {self.failed_models}")
        logger.info("=" * 80)
        
        return success_count == total_models

    def run_complete_pipeline(self):
        """Run complete pipeline: generate datasets and train models"""
        # Step 1: Generate balanced datasets
        if not self.generate_balanced_datasets():
            return False
        
        # Step 2: Train all models
        return self.train_all_models()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Signature String models using balanced datasets')
    parser.add_argument('--cfg_dir', type=str, default=None,
                       help='Directory containing CFG files')
    parser.add_argument('--balanced_dataset_dir', type=str, default=None,
                       help='Directory to save balanced datasets')
    parser.add_argument('--models_dir', type=str, default=None,
                       help='Directory to save trained models')
    parser.add_argument('--examples_per_annotation', type=int, default=1000,
                       help='Number of examples per annotation type')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--skip_dataset_generation', action='store_true',
                       help='Skip dataset generation and use existing datasets')
    
    args = parser.parse_args()
    
    trainer = BalancedSignatureStringModelsTrainer(
        cfg_dir=args.cfg_dir,
        balanced_dataset_dir=args.balanced_dataset_dir,
        models_dir=args.models_dir,
        examples_per_annotation=args.examples_per_annotation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if args.skip_dataset_generation:
        success = trainer.train_all_models()
    else:
        success = trainer.run_complete_pipeline()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

