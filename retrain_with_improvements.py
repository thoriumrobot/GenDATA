#!/usr/bin/env python3
"""
Retrain all annotation type models with improved label semantics and features.

This script:
1. Regenerates balanced datasets with improved labeling rules
2. Retrains feature-based models (GBT, Causal) with cost-sensitive loss
3. Backs up existing models before replacement
"""

import os
import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelRetrainer:
    """Retrains models with improved label semantics and features"""
    
    def __init__(self, cfwr_root: str = None):
        if cfwr_root is None:
            cfwr_root = os.path.dirname(os.path.abspath(__file__))
        self.cfwr_root = cfwr_root
        self.models_dir = os.path.join(cfwr_root, 'models_annotation_types')
        self.cfg_dir = os.path.join(cfwr_root, 'cfg_output_specimin')
        self.balanced_datasets_dir = os.path.join(cfwr_root, 'real_balanced_datasets')
        self.backup_dir = os.path.join(cfwr_root, f'models_annotation_types_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Create directories
        os.makedirs(self.balanced_datasets_dir, exist_ok=True)
        
        self.retraining_stats = {
            'start_time': datetime.now().isoformat(),
            'datasets_regenerated': False,
            'models_trained': [],
            'models_failed': [],
            'backup_created': False
        }
    
    def backup_existing_models(self) -> bool:
        """Backup existing models before retraining"""
        logger.info("Backing up existing models...")
        
        try:
            if os.path.exists(self.models_dir):
                shutil.copytree(self.models_dir, self.backup_dir)
                logger.info(f"Models backed up to {self.backup_dir}")
                self.retraining_stats['backup_created'] = True
                return True
            else:
                logger.warning(f"Models directory not found: {self.models_dir}")
                return True  # Not an error if models don't exist yet
        except Exception as e:
            logger.error(f"Error backing up models: {e}")
            return False
    
    def regenerate_balanced_datasets(self, examples_per_annotation: int = 2000) -> bool:
        """Regenerate balanced datasets with improved labeling"""
        logger.info(f"Regenerating balanced datasets with {examples_per_annotation} examples per annotation type...")
        
        try:
            cmd = [
                sys.executable, 'improved_balanced_dataset_generator.py',
                '--cfg_dir', self.cfg_dir,
                '--output_dir', self.balanced_datasets_dir,
                '--examples_per_annotation', str(examples_per_annotation),
                '--target_balance', '0.5',
                '--random_seed', '42'
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=self.cfwr_root)
            
            if result.returncode != 0:
                logger.error(f"Dataset generation failed: {result.stderr}")
                return False
            
            logger.info("Successfully regenerated balanced datasets")
            logger.info(f"Dataset generation output: {result.stdout[-500:]}")  # Last 500 chars
            self.retraining_stats['datasets_regenerated'] = True
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Dataset generation timed out")
            return False
        except Exception as e:
            logger.error(f"Error regenerating datasets: {e}")
            return False
    
    def train_balanced_models(self, epochs: int = 200, batch_size: int = 32) -> bool:
        """Train balanced models with improved features and cost-sensitive loss"""
        logger.info(f"Training balanced models for {epochs} epochs with batch size {batch_size}...")
        
        try:
            cmd = [
                sys.executable, 'improved_balanced_annotation_type_trainer.py',
                '--balanced_dataset_dir', self.balanced_datasets_dir,
                '--output_dir', self.models_dir,
                '--model_type', 'improved_balanced_causal',
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--device', 'auto'
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=self.cfwr_root)
            
            if result.returncode != 0:
                logger.error(f"Model training failed: {result.stderr}")
                logger.error(f"Training output: {result.stdout[-1000:]}")
                return False
            
            logger.info("Successfully trained balanced models")
            logger.info(f"Training output: {result.stdout[-1000:]}")
            
            # Check which models were trained
            if os.path.exists(self.models_dir):
                model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
                self.retraining_stats['models_trained'] = model_files
                logger.info(f"Trained models: {model_files}")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Model training timed out")
            return False
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def verify_trained_models(self) -> bool:
        """Verify that models were trained successfully"""
        logger.info("Verifying trained models...")
        
        if not os.path.exists(self.models_dir):
            logger.error(f"Models directory not found: {self.models_dir}")
            return False
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
        
        if len(model_files) == 0:
            logger.error("No model files found")
            return False
        
        logger.info(f"Found {len(model_files)} model files:")
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            size = os.path.getsize(model_path)
            logger.info(f"  - {model_file} ({size / 1024:.1f} KB)")
        
        return True
    
    def save_retraining_stats(self):
        """Save retraining statistics"""
        self.retraining_stats['end_time'] = datetime.now().isoformat()
        stats_path = os.path.join(self.cfwr_root, 'retraining_stats.json')
        
        with open(stats_path, 'w') as f:
            json.dump(self.retraining_stats, f, indent=2)
        
        logger.info(f"Retraining statistics saved to {stats_path}")
    
    def run_full_retraining(self, examples_per_annotation: int = 2000, epochs: int = 200, batch_size: int = 32) -> bool:
        """Run the full retraining pipeline"""
        logger.info("="*70)
        logger.info("STARTING FULL MODEL RETRAINING WITH IMPROVEMENTS")
        logger.info("="*70)
        
        # Step 1: Backup existing models
        if not self.backup_existing_models():
            logger.warning("Backup failed, but continuing...")
        
        # Step 2: Regenerate balanced datasets
        if not self.regenerate_balanced_datasets(examples_per_annotation):
            logger.error("Failed to regenerate datasets. Aborting.")
            return False
        
        # Step 3: Train models
        if not self.train_balanced_models(epochs, batch_size):
            logger.error("Failed to train models.")
            return False
        
        # Step 4: Verify models
        if not self.verify_trained_models():
            logger.error("Model verification failed.")
            return False
        
        # Step 5: Save statistics
        self.save_retraining_stats()
        
        logger.info("="*70)
        logger.info("RETRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain models with improved label semantics')
    parser.add_argument('--cfwr_root', default=None, help='CFWR root directory')
    parser.add_argument('--examples_per_annotation', type=int, default=2000,
                       help='Number of examples per annotation type')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    
    args = parser.parse_args()
    
    retrainer = ModelRetrainer(cfwr_root=args.cfwr_root)
    success = retrainer.run_full_retraining(
        examples_per_annotation=args.examples_per_annotation,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

