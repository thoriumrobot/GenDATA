#!/usr/bin/env python3
"""
Balanced Training Pipeline

This script integrates balanced dataset generation and training into the existing
annotation type pipeline to ensure proper model convergence with balanced positive
and negative examples.
"""

import os
import json
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BalancedTrainingPipeline:
    """Pipeline for balanced training of annotation type models"""
    
    def __init__(self, project_root: str, warnings_file: str, cfwr_root: str, 
                 output_dir: str = None):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        
        # Set up directories
        if output_dir is None:
            output_dir = os.path.join(cfwr_root, 'balanced_training_output')
        
        self.output_dir = output_dir
        self.balanced_dataset_dir = os.path.join(output_dir, 'balanced_datasets')
        self.trained_models_dir = os.path.join(output_dir, 'trained_models')
        
        # Create directories
        os.makedirs(self.balanced_dataset_dir, exist_ok=True)
        os.makedirs(self.trained_models_dir, exist_ok=True)
        
        # Pipeline statistics
        self.pipeline_stats = {
            'cfg_files_generated': 0,
            'balanced_examples_generated': 0,
            'models_trained': 0,
            'training_success': False
        }
    
    def generate_cfg_data(self, use_existing: bool = True) -> str:
        """Generate CFG data for balanced dataset creation"""
        cfg_dir = os.path.join(self.cfwr_root, 'cfg_output_specimin')
        
        if use_existing and os.path.exists(cfg_dir):
            # Count existing CFG files
            cfg_files = []
            for root, dirs, files in os.walk(cfg_dir):
                for file in files:
                    if file.endswith('.json'):
                        cfg_files.append(os.path.join(root, file))
            
            if cfg_files:
                logger.info(f"Using existing CFG files: {len(cfg_files)} files found")
                self.pipeline_stats['cfg_files_generated'] = len(cfg_files)
                return cfg_dir
        
        # Generate new CFG files using the existing pipeline
        logger.info("Generating new CFG files...")
        
        try:
            # Use the existing simple annotation type pipeline to generate CFGs
            cmd = [
                'python', 'simple_annotation_type_pipeline.py',
                '--mode', 'train',
                '--project_root', self.project_root,
                '--warnings_file', self.warnings_file,
                '--cfwr_root', self.cfwr_root,
                '--device', 'auto'
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode != 0:
                logger.error(f"CFG generation failed: {result.stderr}")
                return None
            
            # Count generated CFG files
            cfg_files = []
            for root, dirs, files in os.walk(cfg_dir):
                for file in files:
                    if file.endswith('.json'):
                        cfg_files.append(os.path.join(root, file))
            
            logger.info(f"Generated {len(cfg_files)} CFG files")
            self.pipeline_stats['cfg_files_generated'] = len(cfg_files)
            return cfg_dir
            
        except subprocess.TimeoutExpired:
            logger.error("CFG generation timed out")
            return None
        except Exception as e:
            logger.error(f"Error generating CFG files: {e}")
            return None
    
    def generate_balanced_datasets(self, cfg_dir: str, examples_per_annotation: int = 1000,
                                 target_balance: float = 0.5) -> bool:
        """Generate balanced datasets from CFG files"""
        logger.info(f"Generating balanced datasets with {examples_per_annotation} examples per annotation type")
        logger.info(f"Target balance: {target_balance*100:.1f} percent positive, {(1-target_balance)*100:.1f} percent negative")
        
        try:
            cmd = [
                'python', 'balanced_dataset_generator.py',
                '--cfg_dir', cfg_dir,
                '--output_dir', self.balanced_dataset_dir,
                '--examples_per_annotation', str(examples_per_annotation),
                '--target_balance', str(target_balance),
                '--random_seed', '42'
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode != 0:
                logger.error(f"Balanced dataset generation failed: {result.stderr}")
                return False
            
            # Check if datasets were created
            annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
            created_datasets = 0
            
            for ann_type in annotation_types:
                dataset_file = os.path.join(self.balanced_dataset_dir, f"{ann_type}_balanced_dataset.json")
                if os.path.exists(dataset_file):
                    # Count examples in the dataset
                    with open(dataset_file, 'r') as f:
                        dataset_data = json.load(f)
                    examples_count = dataset_data.get('total_examples', 0)
                    created_datasets += 1
                    logger.info(f"Created balanced dataset for {ann_type}: {examples_count} examples")
            
            self.pipeline_stats['balanced_examples_generated'] = created_datasets
            
            if created_datasets == 0:
                logger.error("No balanced datasets were created")
                return False
            
            logger.info(f"Successfully created {created_datasets} balanced datasets")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Balanced dataset generation timed out")
            return False
        except Exception as e:
            logger.error(f"Error generating balanced datasets: {e}")
            return False
    
    def train_balanced_models(self, epochs: int = 100, batch_size: int = 32) -> bool:
        """Train models using balanced datasets"""
        logger.info(f"Training balanced models for {epochs} epochs with batch size {batch_size}")
        
        try:
            cmd = [
                'python', 'balanced_annotation_type_trainer.py',
                '--balanced_dataset_dir', self.balanced_dataset_dir,
                '--output_dir', self.trained_models_dir,
                '--model_type', 'balanced_enhanced_causal',
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--device', 'auto'
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode != 0:
                logger.error(f"Balanced model training failed: {result.stderr}")
                return False
            
            # Check if models were created
            annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
            trained_models = 0
            
            for ann_type in annotation_types:
                model_file = os.path.join(self.trained_models_dir, f"{ann_type}_balanced_model.pth")
                if os.path.exists(model_file):
                    trained_models += 1
                    logger.info(f"Trained model for {ann_type}")
            
            self.pipeline_stats['models_trained'] = trained_models
            
            if trained_models == 0:
                logger.error("No models were trained")
                return False
            
            logger.info(f"Successfully trained {trained_models} balanced models")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Balanced model training timed out")
            return False
        except Exception as e:
            logger.error(f"Error training balanced models: {e}")
            return False
    
    def run_complete_pipeline(self, examples_per_annotation: int = 1000, 
                            target_balance: float = 0.5, epochs: int = 100,
                            batch_size: int = 32, use_existing_cfg: bool = True) -> bool:
        """Run the complete balanced training pipeline"""
        logger.info("Starting balanced training pipeline...")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Warnings file: {self.warnings_file}")
        logger.info(f"CFWR root: {self.cfwr_root}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Generate CFG data
        logger.info("\n=== STEP 1: CFG Data Generation ===")
        cfg_dir = self.generate_cfg_data(use_existing=use_existing_cfg)
        if not cfg_dir:
            logger.error("Failed to generate CFG data")
            return False
        
        # Step 2: Generate balanced datasets
        logger.info("\n=== STEP 2: Balanced Dataset Generation ===")
        if not self.generate_balanced_datasets(cfg_dir, examples_per_annotation, target_balance):
            logger.error("Failed to generate balanced datasets")
            return False
        
        # Step 3: Train balanced models
        logger.info("\n=== STEP 3: Balanced Model Training ===")
        if not self.train_balanced_models(epochs, batch_size):
            logger.error("Failed to train balanced models")
            return False
        
        # Pipeline completed successfully
        self.pipeline_stats['training_success'] = True
        logger.info("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
        self.print_pipeline_summary()
        
        return True
    
    def print_pipeline_summary(self):
        """Print pipeline execution summary"""
        print("\n" + "="*60)
        print("BALANCED TRAINING PIPELINE SUMMARY")
        print("="*60)
        print(f"CFG files generated/used: {self.pipeline_stats['cfg_files_generated']}")
        print(f"Balanced datasets created: {self.pipeline_stats['balanced_examples_generated']}")
        print(f"Models trained: {self.pipeline_stats['models_trained']}")
        print(f"Training success: {self.pipeline_stats['training_success']}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
        # Print file locations
        print("\nGenerated files:")
        print(f"  Balanced datasets: {self.balanced_dataset_dir}")
        print(f"  Trained models: {self.trained_models_dir}")
        
        # List generated files
        if os.path.exists(self.balanced_dataset_dir):
            print("\nBalanced dataset files:")
            for file in os.listdir(self.balanced_dataset_dir):
                if file.endswith('.json'):
                    print(f"  - {file}")
        
        if os.path.exists(self.trained_models_dir):
            print("\nTrained model files:")
            for file in os.listdir(self.trained_models_dir):
                if file.endswith('.pth'):
                    print(f"  - {file}")
        
        print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Balanced Training Pipeline for Annotation Type Models')
    parser.add_argument('--project_root', required=True,
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', required=True,
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', required=True,
                       help='Root directory of CFWR project')
    parser.add_argument('--output_dir',
                       help='Output directory for balanced training results')
    parser.add_argument('--examples_per_annotation', type=int, default=1000,
                       help='Number of examples to generate per annotation type')
    parser.add_argument('--target_balance', type=float, default=0.5,
                       help='Target balance ratio for positive examples (0.5 = 50 percent positive)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--use_existing_cfg', action='store_true', default=True,
                       help='Use existing CFG files if available')
    parser.add_argument('--regenerate_cfg', action='store_true',
                       help='Force regeneration of CFG files')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = BalancedTrainingPipeline(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        output_dir=args.output_dir
    )
    
    # Run pipeline
    success = pipeline.run_complete_pipeline(
        examples_per_annotation=args.examples_per_annotation,
        target_balance=args.target_balance,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_existing_cfg=not args.regenerate_cfg
    )
    
    if success:
        logger.info("Balanced training pipeline completed successfully!")
        return 0
    else:
        logger.error("Balanced training pipeline failed!")
        return 1


if __name__ == '__main__':
    exit(main())
