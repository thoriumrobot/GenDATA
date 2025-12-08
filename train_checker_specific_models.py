#!/usr/bin/env python3
"""
Training Script for Checker-Specific Models

Trains models for each checker separately with automatic value emphasis learning.
Monitors attention weights during training to see which values are being emphasized.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch
import numpy as np

from checker_config import CheckerType, get_all_checker_types, get_checker_config
from checker_specific_models import create_checker_specific_model
from improved_balanced_annotation_type_trainer import ImprovedBalancedAnnotationTypeTrainer
from value_pattern_detector import ValuePatternDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CheckerSpecificModelTrainer:
    """Train checker-specific models with automatic value emphasis learning"""
    
    def __init__(
        self,
        output_dir: str = 'checker_specific_models',
        balanced_dataset_dir: str = 'real_balanced_datasets',
        cfg_dir: str = None,
        device: str = 'auto',
        base_model_types: List[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.balanced_dataset_dir = Path(balanced_dataset_dir)
        
        if cfg_dir is None:
            cfg_dir = os.environ.get('CFG_OUTPUT_DIR', 'cfg_output_specimin')
        self.cfg_dir = cfg_dir
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        if base_model_types is None:
            # Default: train feature-based models (GBT, Causal, Enhanced Causal)
            self.base_model_types = ['gbt', 'causal', 'enhanced_causal']
        else:
            self.base_model_types = base_model_types
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'device': self.device,
                'base_model_types': self.base_model_types,
                'checkers': [ct.name for ct in get_all_checker_types()]
            },
            'training_results': {}
        }
    
    def train_checker_models(
        self,
        checker_type: CheckerType,
        annotation_types: List[str] = None,
        epochs: int = 200,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train models for a specific checker
        
        Args:
            checker_type: Type of checker to train for
            annotation_types: List of annotation types (default: from checker config)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training results
        """
        config = get_checker_config(checker_type)
        checker_name = config.get('name', 'Unknown')
        
        if annotation_types is None:
            annotation_types = config.get('annotation_types', ['@Positive', '@NonNegative', '@GTENegativeOne'])
        
        logger.info("=" * 80)
        logger.info(f"Training models for {checker_name} ({checker_type.name})")
        logger.info("=" * 80)
        logger.info(f"Annotation types: {annotation_types}")
        logger.info(f"Base model types: {self.base_model_types}")
        logger.info(f"Target values: {config.get('target_values', [])}")
        
        checker_results = {
            'checker_type': checker_type.name,
            'checker_name': checker_name,
            'target_values': config.get('target_values', []),
            'models': {}
        }
        
        # Train models for each base model type
        for base_model_type in self.base_model_types:
            logger.info(f"\nTraining {base_model_type} models for {checker_name}...")
            
            model_results = {}
            
            for annotation_type in annotation_types:
                logger.info(f"  Training {annotation_type} with {base_model_type}...")
                
                # Create trainer with checker type
                trainer = ImprovedBalancedAnnotationTypeTrainer(
                    model_type=f'improved_balanced_{base_model_type}',
                    device=self.device,
                    checker_type=checker_type
                )
                
                # Find dataset file
                dataset_file = self.balanced_dataset_dir / f"{annotation_type.lower().replace('@', '').replace('gtenegativeone', 'gtenegativeone')}_real_balanced_dataset.json"
                
                if not dataset_file.exists():
                    logger.warning(f"Dataset file not found: {dataset_file}")
                    model_results[annotation_type] = {
                        'success': False,
                        'error': f'Dataset file not found: {dataset_file}'
                    }
                    continue
                
                # Train model
                try:
                    result = trainer.train_model(
                        dataset_file=str(dataset_file),
                        epochs=epochs,
                        batch_size=batch_size
                    )
                    
                    # Extract attention weights if available
                    attention_summary = None
                    if hasattr(trainer.models.get(annotation_type), 'value_attention'):
                        attention_module = trainer.models[annotation_type].value_attention
                        attention_summary = attention_module.get_attention_summary(
                            torch.zeros(len(config.get('value_patterns', [])))
                        )
                    
                    model_results[annotation_type] = {
                        'success': result.get('success', False),
                        'training_stats': result.get('training_stats', {}),
                        'attention_summary': attention_summary,
                        'best_val_accuracy': result.get('training_stats', {}).get('val_accuracy')
                    }
                    
                    logger.info(f"    ✅ {annotation_type}: Val Acc={model_results[annotation_type].get('best_val_accuracy', 'N/A')}")
                    
                except Exception as e:
                    logger.error(f"    ❌ Error training {annotation_type}: {e}")
                    model_results[annotation_type] = {
                        'success': False,
                        'error': str(e)
                    }
            
            checker_results['models'][base_model_type] = model_results
        
        return checker_results
    
    def train_all_checkers(
        self,
        checkers: List[CheckerType] = None,
        epochs: int = 200,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train models for all checkers
        
        Args:
            checkers: List of checkers to train (default: all)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with all training results
        """
        if checkers is None:
            checkers = get_all_checker_types()
        
        logger.info("=" * 80)
        logger.info("TRAINING CHECKER-SPECIFIC MODELS")
        logger.info("=" * 80)
        logger.info(f"Checkers: {[ct.name for ct in checkers]}")
        logger.info(f"Base models: {self.base_model_types}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Device: {self.device}")
        
        for checker_type in checkers:
            checker_results = self.train_checker_models(
                checker_type=checker_type,
                epochs=epochs,
                batch_size=batch_size
            )
            
            self.results['training_results'][checker_type.name] = checker_results
        
        # Save results
        results_file = self.output_dir / 'checker_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to {results_file}")
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print summary of training results"""
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        
        for checker_name, checker_results in self.results['training_results'].items():
            logger.info(f"\n{checker_results.get('checker_name', checker_name)}:")
            logger.info(f"  Target values: {checker_results.get('target_values', [])}")
            
            for model_type, models in checker_results.get('models', {}).items():
                logger.info(f"  {model_type}:")
                for ann_type, result in models.items():
                    if result.get('success'):
                        val_acc = result.get('best_val_accuracy', 'N/A')
                        logger.info(f"    {ann_type}: Val Acc={val_acc}")
                        
                        # Print attention summary
                        attn_summary = result.get('attention_summary')
                        if attn_summary:
                            top_patterns = sorted(attn_summary.items(), key=lambda x: x[1], reverse=True)[:3]
                            logger.info(f"      Top emphasized patterns: {top_patterns}")
                    else:
                        logger.info(f"    {ann_type}: Failed - {result.get('error', 'Unknown error')}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train checker-specific models with automatic value emphasis learning'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checker_specific_models',
        help='Output directory for trained models'
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
        help='CFG directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--base_model_types',
        type=str,
        nargs='+',
        default=['gbt', 'causal', 'enhanced_causal'],
        help='Base model types to train'
    )
    parser.add_argument(
        '--checkers',
        type=str,
        nargs='+',
        default=None,
        help='Specific checkers to train (default: all)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    args = parser.parse_args()
    
    # Parse checker types
    checkers = None
    if args.checkers:
        from checker_config import get_checker_by_name
        checkers = [get_checker_by_name(name) for name in args.checkers]
    
    trainer = CheckerSpecificModelTrainer(
        output_dir=args.output_dir,
        balanced_dataset_dir=args.balanced_dataset_dir,
        cfg_dir=args.cfg_dir,
        device=args.device,
        base_model_types=args.base_model_types
    )
    
    results = trainer.train_all_checkers(
        checkers=checkers,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Checker-specific model training completed!")
    logger.info(f"Results saved to: {trainer.output_dir / 'checker_training_results.json'}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

