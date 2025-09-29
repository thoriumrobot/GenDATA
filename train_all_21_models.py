#!/usr/bin/env python3
"""
Train All 21 Models Script

This script trains all 21 annotation type models:
- 7 base model types: gcn, gbt, causal, enhanced_causal, hgt, gcsn, dg2n
- 3 annotation types: @Positive, @NonNegative, @GTENegativeOne
- Total: 7 × 3 = 21 models

Each model is trained using the full pipeline:
1. Specimin slice generation
2. Slice augmentation
3. CFG generation using Checker Framework's CFG Builder
4. Model training on real CFG data

Models are saved with the correct naming convention: {annotation_type}_{base_model}_model.pth
"""

import os
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AllModelsTrainer:
    def __init__(self, project_root='/home/ubuntu/checker-framework/checker/tests/index', episodes=10):
        self.project_root = project_root
        self.episodes = episodes
        
        # All base model types (including enhanced_causal)
        self.base_models = ['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']
        
        # All annotation types with their scripts
        self.annotation_configs = [
            ('@Positive', 'annotation_type_rl_positive.py'),
            ('@NonNegative', 'annotation_type_rl_nonnegative.py'),
            ('@GTENegativeOne', 'annotation_type_rl_gtenegativeone.py')
        ]
        
        self.trained_models = []
        self.failed_models = []

    def train_single_model(self, annotation_type, base_model, script_path):
        """Train a single model using the full pipeline"""
        model_name = f"{annotation_type.replace('@', '').lower()}_{base_model}"
        logger.info(f"🚀 Training {model_name} model using full pipeline...")
        
        # Build command with full pipeline parameters
        cmd = [
            'python3', script_path,
            '--project_root', self.project_root,
            '--warnings_file', '/home/ubuntu/GenDATA/index1.out',
            '--cfwr_root', '/home/ubuntu/GenDATA',
            '--slices_dir', '/home/ubuntu/GenDATA/slices_augmented',
            '--cfg_dir', '/home/ubuntu/GenDATA/cfg_output_specimin',
            '--episodes', str(self.episodes),
            '--base_model', base_model,
            '--device', 'cpu',
            '--use_real_cfg_data'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully trained {model_name}")
                self.trained_models.append(model_name)
                return True
            else:
                logger.error(f"❌ Failed to train {model_name}: {result.stderr}")
                self.failed_models.append(model_name)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Training timeout for {model_name}")
            self.failed_models.append(model_name)
            return False
        except Exception as e:
            logger.error(f"💥 Error training {model_name}: {e}")
            self.failed_models.append(model_name)
            return False

    def train_all_models(self):
        """Train all 21 models using the full pipeline"""
        logger.info("🎯 Starting training of all 21 annotation type models...")
        logger.info("Using full pipeline: Specimin → Augmentation → CFG Builder → Training")
        logger.info("=" * 80)
        
        total_models = len(self.base_models) * len(self.annotation_configs)
        current_model = 0
        
        for annotation_type, script_path in self.annotation_configs:
            logger.info(f"📋 Training models for {annotation_type}...")
            
            for base_model in self.base_models:
                current_model += 1
                logger.info(f"📊 Progress: {current_model}/{total_models}")
                
                success = self.train_single_model(annotation_type, base_model, script_path)
                
                if success:
                    logger.info(f"✅ {current_model}/{total_models} completed")
                else:
                    logger.error(f"❌ {current_model}/{total_models} failed")
                
                logger.info("-" * 40)
        
        # Summary
        logger.info("=" * 80)
        logger.info("🎯 TRAINING COMPLETE!")
        logger.info(f"✅ Successfully trained: {len(self.trained_models)}/18 models")
        logger.info(f"❌ Failed to train: {len(self.failed_models)}/18 models")
        
        if self.trained_models:
            logger.info("\n✅ Successfully trained models:")
            for model in self.trained_models:
                logger.info(f"  - {model}")
        
        if self.failed_models:
            logger.info("\n❌ Failed models:")
            for model in self.failed_models:
                logger.info(f"  - {model}")
        
        # Check what models are actually saved
        self.check_saved_models()
        
        return len(self.trained_models) == 21

    def check_saved_models(self):
        """Check what model files are actually saved"""
        logger.info("\n🔍 Checking saved model files...")
        
        models_dir = 'models_annotation_types'
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory {models_dir} does not exist")
            return
        
        model_files = []
        stats_files = []
        
        for file in os.listdir(models_dir):
            if file.endswith('_model.pth'):
                model_files.append(file)
            elif file.endswith('_stats.json'):
                stats_files.append(file)
        
        logger.info(f"📁 Found {len(model_files)} model files (.pth)")
        logger.info(f"📁 Found {len(stats_files)} stats files (.json)")
        
        if model_files:
            logger.info("\n📋 Model files found:")
            for model_file in sorted(model_files):
                logger.info(f"  - {model_file}")
        
        if stats_files:
            logger.info("\n📊 Stats files found:")
            for stats_file in sorted(stats_files):
                logger.info(f"  - {stats_file}")

def main():
    """Main function"""
    logger.info("🎯 GenDATA - Train All 21 Models")
    logger.info("=" * 80)
    logger.info("Training 21 annotation type models:")
    logger.info("- 7 base model types: gcn, gbt, causal, enhanced_causal, hgt, gcsn, dg2n")
    logger.info("- 3 annotation types: @Positive, @NonNegative, @GTENegativeOne")
    logger.info("- Total: 7 × 3 = 21 models")
    logger.info("- Using full pipeline: Specimin → Augmentation → CFG Builder → Training")
    logger.info("=" * 80)
    
    # Create trainer
    trainer = AllModelsTrainer(episodes=10)  # Reduced episodes for faster training
    
    # Train all models
    success = trainer.train_all_models()
    
    if success:
        logger.info("\n🎉 All 21 models trained successfully!")
        logger.info("You can now use the model_based_predictor.py to load and use these models.")
    else:
        logger.info(f"\n⚠️ Training completed with some failures.")
        logger.info("Check the logs above for details on failed models.")
    
    return success

if __name__ == '__main__':
    main()
