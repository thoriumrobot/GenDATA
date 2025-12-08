#!/usr/bin/env python3
"""
Train All Signature String Checker Models

This script trains all models for Signature String Checker annotations:
- 7 base models × 3 annotation types = 21 models total
- @FullyQualifiedName
- @BinaryName
- @FieldDescriptor
"""

import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SignatureStringModelsTrainer:
    def __init__(self, project_root=None, episodes=100):
        self.project_root = project_root or '/home/ubuntu/checker-framework/checker/tests/signature'
        self.episodes = episodes
        
        # Base model types
        self.base_models = ['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n']
        
        # Annotation types for Signature String
        self.annotation_configs = [
            ('@FullyQualifiedName', 'annotation_type_rl_signature_string_fullyqualified.py'),
            ('@BinaryName', 'annotation_type_rl_signature_string_binary.py'),
            ('@FieldDescriptor', 'annotation_type_rl_signature_string_fielddescriptor.py')
        ]
        
        self.trained_models = []
        self.failed_models = []

    def train_single_model(self, annotation_type, base_model, script_path):
        """Train a single model"""
        model_name = f"{annotation_type.replace('@', '').lower()}_{base_model}"
        logger.info(f"🚀 Training {model_name} model...")
        
        # Build command
        cmd = [
            'python3', script_path,
            '--project_root', self.project_root,
            '--warnings_file', f'/home/ubuntu/GenDATA/signature_string_warnings.out',
            '--cfwr_root', '/home/ubuntu/GenDATA',
            '--episodes', str(self.episodes),
            '--base_model', base_model,
            '--device', 'auto',
            '--checker_type', 'signature_string'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully trained {model_name}")
                self.trained_models.append(model_name)
                return True
            else:
                logger.error(f"❌ Failed to train {model_name}: {result.stderr}")
                self.failed_models.append(model_name)
                return False
        except Exception as e:
            logger.error(f"💥 Error training {model_name}: {e}")
            self.failed_models.append(model_name)
            return False

    def train_all_models(self):
        """Train all 21 Signature String models"""
        logger.info("🎯 Starting training of all 21 Signature String Checker models...")
        logger.info("=" * 80)
        
        total_models = len(self.base_models) * len(self.annotation_configs)
        success_count = 0
        
        for annotation_type, script_path in self.annotation_configs:
            for base_model in self.base_models:
                if self.train_single_model(annotation_type, base_model, script_path):
                    success_count += 1
        
        logger.info("=" * 80)
        logger.info(f"✅ Training complete: {success_count}/{total_models} models trained successfully")
        
        if self.failed_models:
            logger.warning(f"❌ Failed models: {', '.join(self.failed_models)}")
        
        return success_count == total_models

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train all Signature String Checker models')
    parser.add_argument('--project_root', help='Project root directory')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    
    args = parser.parse_args()
    
    trainer = SignatureStringModelsTrainer(project_root=args.project_root, episodes=args.episodes)
    success = trainer.train_all_models()
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())

