#!/usr/bin/env python3
"""
Augment-First Pipeline for GenDATA

This pipeline implements the new approach:
1. Augment the original target code first
2. Slice each augmented variant
3. Generate CFGs from all slices

This approach is more effective because:
- Slicers work on semantically equivalent code with different syntax
- Different syntactic structures may produce different slice patterns
- Models see how the same semantic intent can be expressed differently
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
import numpy as np
import torch
import logging
from pathlib import Path
import time
import glob
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentFirstPipeline:
    """Pipeline that augments code first, then slices each variant"""
    
    def __init__(self, project_root: str, warnings_file: str, cfwr_root: str, 
                 augmentation_factor: int = 50, slicer_type: str = 'specimin'):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        self.augmentation_factor = augmentation_factor
        self.slicer_type = slicer_type
        
        # Set up directories for adaptive semantic augmentation
        self.augmented_code_dir = os.path.join(cfwr_root, 'augmented_code_adaptive')
        self.slices_dir = os.path.join(cfwr_root, 'slices_adaptive_augmented_first')
        self.cfg_dir = os.path.join(cfwr_root, 'cfg_output_adaptive_augmented_first')
        self.models_dir = os.path.join(cfwr_root, 'models_annotation_types')
        self.predictions_dir = os.path.join(cfwr_root, 'predictions_annotation_types')
        
        # Create directories
        for dir_path in [self.augmented_code_dir, self.slices_dir, self.cfg_dir, 
                        self.models_dir, self.predictions_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_training_pipeline(self, episodes: int = 50, base_model: str = 'gcn') -> bool:
        """Run the complete training pipeline with augment-first approach"""
        logger.info("Starting augment-first training pipeline")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Augmentation factor: {self.augmentation_factor}")
        logger.info(f"Slicer type: {self.slicer_type}")
        
        # Step 1: Augment original code first
        logger.info("Step 1: Augmenting original code with semantic transformations")
        if not self._augment_original_code():
            logger.error("Failed to augment original code")
            return False
        
        # Step 2: Slice each augmented variant
        logger.info("Step 2: Slicing each augmented variant")
        if not self._slice_augmented_variants():
            logger.error("Failed to slice augmented variants")
            return False
        
        # Step 3: Generate CFGs from all slices
        logger.info("Step 3: Generating CFGs from all slices")
        if not self._generate_cfgs_from_slices():
            logger.error("Failed to generate CFGs")
            return False
        
        # Step 4: Train annotation type models
        logger.info("Step 4: Training annotation type models")
        if not self._train_annotation_type_models(episodes, base_model):
            logger.error("Failed to train annotation type models")
            return False
        
        logger.info("Augment-first training pipeline completed successfully")
        return True
    
    def _augment_original_code(self) -> bool:
        """Augment the original code with adaptive semantic transformations"""
        try:
            # Import both augmentation systems
            from enhanced_semantic_augment_slices import EnhancedSemanticTransformer, iter_java_files
            from simple_code_semantic_augment_slices import SimpleCodeSemanticTransformer
            
            logger.info(f"Augmenting original code with ADAPTIVE semantic transformations")
            logger.info(f"Using {self.augmentation_factor} variants per file")
            logger.info("Adaptive system: Enhanced (17 methods) for complex code, Simple (10 methods) for Checker Framework test cases")
            
            # Initialize both transformers
            enhanced_transformer = EnhancedSemanticTransformer(seed=42)
            simple_transformer = SimpleCodeSemanticTransformer(seed=42)
            augmented_count = 0
            enhanced_count = 0
            simple_count = 0
            
            # Process each Java file in the project
            for java_file in iter_java_files(self.project_root):
                # Analyze code complexity to select appropriate augmentation system
                complexity_score = self._analyze_code_complexity(java_file)
                
                # Select transformer based on complexity
                if complexity_score >= 3:
                    transformer = enhanced_transformer
                    system_type = "Enhanced"
                    enhanced_count += 1
                else:
                    transformer = simple_transformer
                    system_type = "Simple"
                    simple_count += 1
                
                logger.debug(f"File: {os.path.basename(java_file)}, Complexity: {complexity_score}, System: {system_type}")
                
                # Create output directory maintaining structure
                rel_path = os.path.relpath(java_file, self.project_root)
                base_name = os.path.splitext(rel_path)[0]
                
                # Generate variants
                for variant_idx in range(self.augmentation_factor):
                    variant_dir = os.path.join(self.augmented_code_dir, f"{base_name}__variant_{variant_idx}")
                    os.makedirs(variant_dir, exist_ok=True)
                    output_path = os.path.join(variant_dir, os.path.basename(rel_path))
                    
                    # Apply semantic transformations
                    augmented_content = transformer.transform_file(java_file, variant_idx)
                    with open(output_path, 'w') as f:
                        f.write(augmented_content)
                    augmented_count += 1
            
            # Verify augmentation
            original_files = len(glob.glob(os.path.join(self.project_root, '**/*.java'), recursive=True))
            augmented_files = len(glob.glob(os.path.join(self.augmented_code_dir, '**/*.java'), recursive=True))
            
            logger.info(f"Original files: {original_files}, Augmented files: {augmented_files}")
            logger.info(f"Generated {augmented_count} semantically augmented code variants")
            logger.info(f"Enhanced augmentation used for {enhanced_count} files, Simple augmentation used for {simple_count} files")
            
            if augmented_files >= original_files * self.augmentation_factor:
                logger.info("Adaptive semantic augmentation completed successfully")
                return True
            else:
                logger.warning(f"Augmentation may be incomplete: expected ~{original_files * self.augmentation_factor}, got {augmented_files}")
                return True  # Still proceed
                
        except Exception as e:
            logger.error(f"Error augmenting original code: {e}")
            return False
    
    def _analyze_code_complexity(self, java_file_path: str) -> int:
        """Analyze code complexity to determine appropriate augmentation system."""
        try:
            with open(java_file_path, 'r') as f:
                content = f.read()
            
            # Complexity indicators for enhanced augmentation
            complexity_indicators = [
                'for (', 'while (', 'stream()', 'lambda', '->', 
                'try {', 'catch', 'switch', 'interface', 'enum',
                'Collection<', 'List<', 'Map<', 'Set<', 'Optional<',
                'Stream<', 'Function<', 'Predicate<', 'Consumer<',
                'synchronized', 'volatile', 'transient', 'native'
            ]
            
            # Count complexity indicators
            complexity_score = 0
            for indicator in complexity_indicators:
                if indicator in content:
                    complexity_score += 1
            
            return complexity_score
            
        except Exception as e:
            logger.warning(f"Error analyzing complexity for {java_file_path}: {e}")
            return 0  # Default to simple augmentation if analysis fails
    
    def _slice_augmented_variants(self) -> bool:
        """Slice each augmented variant using the specified slicer"""
        try:
            logger.info(f"Slicing {self.augmentation_factor} variants per file using {self.slicer_type} slicer")
            
            total_slices = 0
            
            # Process each augmented variant
            for variant_dir in os.listdir(self.augmented_code_dir):
                variant_path = os.path.join(self.augmented_code_dir, variant_dir)
                if not os.path.isdir(variant_path):
                    continue
                
                # Create slice output directory for this variant
                variant_slice_dir = os.path.join(self.slices_dir, variant_dir)
                os.makedirs(variant_slice_dir, exist_ok=True)
                
                # Run slicing on this variant
                if self._slice_single_variant(variant_path, variant_slice_dir):
                    # Count slices generated for this variant
                    variant_slices = len(glob.glob(os.path.join(variant_slice_dir, '**/*.java'), recursive=True))
                    total_slices += variant_slices
                    logger.info(f"Generated {variant_slices} slices for variant {variant_dir}")
                else:
                    logger.warning(f"Failed to slice variant {variant_dir}")
            
            logger.info(f"Total slices generated: {total_slices}")
            
            if total_slices > 0:
                logger.info("Augmented variant slicing completed successfully")
                return True
            else:
                logger.error("No slices generated from any variant")
                return False
                
        except Exception as e:
            logger.error(f"Error slicing augmented variants: {e}")
            return False
    
    def _slice_single_variant(self, variant_path: str, output_dir: str) -> bool:
        """Slice a single augmented variant"""
        try:
            # Use the existing slicing infrastructure
            from pipeline import run_slicing
            
            # Create a temporary warnings file for this variant
            # We'll use the same warnings file for all variants since they're semantically equivalent
            temp_warnings = os.path.join(output_dir, "temp_warnings.out")
            shutil.copy2(self.warnings_file, temp_warnings)
            
            # Run slicing on this variant
            success = run_slicing(
                project_root=variant_path,
                warnings_file=temp_warnings,
                cfwr_root=self.cfwr_root,
                base_slices_dir=output_dir,
                slicer_type=self.slicer_type
            )
            
            # Clean up temporary warnings file
            if os.path.exists(temp_warnings):
                os.remove(temp_warnings)
            
            return success
            
        except Exception as e:
            logger.error(f"Error slicing variant {variant_path}: {e}")
            return False
    
    def _generate_cfgs_from_slices(self) -> bool:
        """Generate CFGs from all slices"""
        try:
            # Use the existing CFG generation system
            from cfg import generate_control_flow_graphs
            
            logger.info("Generating CFGs from all slices")
            generate_control_flow_graphs(self.slices_dir, self.cfg_dir)
            
            # Verify CFG generation
            cfg_files = len(glob.glob(os.path.join(self.cfg_dir, '**/*.json'), recursive=True))
            logger.info(f"Generated {cfg_files} CFG files")
            
            if cfg_files > 0:
                logger.info("CFG generation completed successfully")
                return True
            else:
                logger.error("No CFG files generated")
                return False
                
        except Exception as e:
            logger.error(f"Error generating CFGs: {e}")
            return False
    
    def _train_annotation_type_models(self, episodes: int, base_model: str) -> bool:
        """Train models for each annotation type"""
        success_count = 0
        annotation_types = ['@Positive', '@NonNegative', '@GTENegativeOne']
        script_mapping = {
            '@Positive': 'annotation_type_rl_positive.py',
            '@NonNegative': 'annotation_type_rl_nonnegative.py',
            '@GTENegativeOne': 'annotation_type_rl_gtenegativeone.py'
        }
        
        for annotation_type in annotation_types:
            script_name = script_mapping[annotation_type]
            script_path = os.path.join(self.cfwr_root, script_name)
            
            if not os.path.exists(script_path):
                logger.warning(f"Training script not found: {script_path}")
                continue
            
            logger.info(f"Training {annotation_type} model using {script_name}")
            
            try:
                cmd = [
                    'python', script_path,
                    '--episodes', str(episodes),
                    '--base_model', base_model,
                    '--project_root', self.project_root,
                    '--warnings_file', self.warnings_file,
                    '--cfwr_root', self.cfwr_root,
                    '--models_dir', self.models_dir,
                    '--cfg_dir', self.cfg_dir,
                    '--slices_dir', self.slices_dir
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    logger.info(f"Successfully trained {annotation_type} model")
                    success_count += 1
                else:
                    logger.error(f"Failed to train {annotation_type} model: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Training {annotation_type} model timed out")
            except Exception as e:
                logger.error(f"Error training {annotation_type} model: {e}")
        
        logger.info(f"Successfully trained {success_count}/{len(annotation_types)} annotation type models")
        return success_count > 0
    
    def run_prediction_pipeline(self, target_file: str) -> bool:
        """Run the prediction pipeline using the augment-first approach"""
        logger.info("Starting augment-first prediction pipeline")
        logger.info(f"Target file: {target_file}")
        
        # For prediction, we can use the existing prediction infrastructure
        # since we now have trained models and CFG generation capability
        
        try:
            # Use the existing prediction system
            from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
            
            # Create a simple pipeline instance for prediction
            pipeline = SimpleAnnotationTypePipeline(
                project_root=os.path.dirname(target_file),
                warnings_file=self.warnings_file,
                cfwr_root=self.cfwr_root,
                mode='predict'
            )
            
            # Override directories to use our augment-first results
            pipeline.cfg_dir = self.cfg_dir
            pipeline.models_dir = self.models_dir
            pipeline.predictions_dir = self.predictions_dir
            
            # Run prediction
            return pipeline.run_prediction_pipeline(target_file)
            
        except Exception as e:
            logger.error(f"Error running prediction pipeline: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Augment-First Pipeline for GenDATA')
    parser.add_argument('--project_root', required=True, help='Root directory of the Java project')
    parser.add_argument('--warnings_file', required=True, help='Path to warnings file')
    parser.add_argument('--cfwr_root', default=os.getcwd(), help='CFWR root directory')
    parser.add_argument('--augmentation_factor', type=int, default=50, help='Number of variants to generate per file')
    parser.add_argument('--slicer_type', default='specimin', choices=['cf', 'specimin', 'wala', 'soot'], 
                       help='Slicer type to use')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], help='Pipeline mode')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--base_model', default='gcn', help='Base model type')
    parser.add_argument('--target_file', help='Target file for prediction mode')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = AugmentFirstPipeline(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        augmentation_factor=args.augmentation_factor,
        slicer_type=args.slicer_type
    )
    
    # Run pipeline
    if args.mode == 'train':
        success = pipeline.run_training_pipeline(episodes=args.episodes, base_model=args.base_model)
    elif args.mode == 'predict':
        if not args.target_file:
            logger.error("Target file required for prediction mode")
            return False
        success = pipeline.run_prediction_pipeline(args.target_file)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return False
    
    if success:
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed")
    
    return success


if __name__ == '__main__':
    main()


