#!/usr/bin/env python3
"""
Improved Augment-First Pipeline for GenDATA

This pipeline ensures:
1. Original code is NEVER modified (read-only access)
2. Augmentation happens FIRST with full project context
3. Each augmented variant maintains complete project structure
4. Slicer gets full code context for each variant
5. All operations work on copies, preserving originals
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
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedAugmentFirstPipeline:
    """Improved pipeline that ensures original code preservation and proper augment-first flow"""
    
    def __init__(self, project_root: str, warnings_file: str, cfwr_root: str, 
                 augmentation_factor: int = 50, slicer_type: str = 'soot'):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        self.augmentation_factor = augmentation_factor
        self.slicer_type = slicer_type
        
        # Store original checksums to verify preservation
        self.original_checksums = {}
        
        # Set up directories
        self.augmented_code_dir = os.path.join(cfwr_root, 'augmented_code')
        self.slices_dir = os.path.join(cfwr_root, 'slices_augmented_first')
        self.cfg_dir = os.path.join(cfwr_root, 'cfg_output_augmented_first')
        self.models_dir = os.path.join(cfwr_root, 'models_annotation_types')
        self.predictions_dir = os.path.join(cfwr_root, 'predictions_annotation_types')
        
        # Create directories
        for dir_path in [self.augmented_code_dir, self.slices_dir, self.cfg_dir, 
                        self.models_dir, self.predictions_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def preserve_original_code(self) -> bool:
        """Ensure original code is preserved (read-only verification)"""
        logger.info("🔒 Preserving original code integrity...")
        
        # Calculate and store checksums of all original files
        for root, dirs, files in os.walk(self.project_root):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    self.original_checksums[rel_path] = self.calculate_file_checksum(file_path)
        
        logger.info(f"📋 Calculated checksums for {len(self.original_checksums)} original Java files")
        logger.info("✅ Original code preservation initialized")
        return True
    
    def verify_original_code_preservation(self) -> bool:
        """Verify that original code has not been modified"""
        logger.info("🔍 Verifying original code preservation...")
        
        preserved_count = 0
        modified_count = 0
        
        for rel_path, original_checksum in self.original_checksums.items():
            original_file = os.path.join(self.project_root, rel_path)
            
            if not os.path.exists(original_file):
                logger.error(f"❌ Original file missing: {rel_path}")
                modified_count += 1
                continue
            
            current_checksum = self.calculate_file_checksum(original_file)
            if current_checksum == original_checksum:
                preserved_count += 1
            else:
                logger.error(f"❌ Original file modified: {rel_path}")
                modified_count += 1
        
        logger.info(f"✅ Preserved: {preserved_count}/{len(self.original_checksums)} files")
        if modified_count > 0:
            logger.error(f"❌ Modified: {modified_count} files")
            return False
        
        logger.info("✅ Original code preservation verified")
        return True
    
    def run_training_pipeline(self, episodes: int = 50, base_model: str = 'gcn') -> bool:
        """Run the complete training pipeline with improved augment-first approach"""
        logger.info("🚀 Starting improved augment-first training pipeline")
        logger.info(f"📁 Project root: {self.project_root}")
        logger.info(f"🔢 Augmentation factor: {self.augmentation_factor}")
        logger.info(f"🔧 Slicer type: {self.slicer_type}")
        
        # Step 0: Preserve original code
        logger.info("Step 0: Preserving original code integrity")
        if not self.preserve_original_code():
            logger.error("Failed to preserve original code")
            return False
        
        # Step 1: Augment original code first (with full project context)
        logger.info("Step 1: Augmenting original code with semantic transformations")
        if not self._augment_original_code_with_full_context():
            logger.error("Failed to augment original code")
            return False
        
        # Step 2: Verify original code preservation
        logger.info("Step 2: Verifying original code preservation")
        if not self.verify_original_code_preservation():
            logger.error("Original code was modified during augmentation")
            return False
        
        # Step 3: Slice each augmented variant (with full context)
        logger.info("Step 3: Slicing each augmented variant with full context")
        if not self._slice_augmented_variants_with_full_context():
            logger.error("Failed to slice augmented variants")
            return False
        
        # Step 4: Generate CFGs from all slices
        logger.info("Step 4: Generating CFGs from all slices")
        if not self._generate_cfgs_from_slices():
            logger.error("Failed to generate CFGs")
            return False
        
        # Step 5: Train annotation type models
        logger.info("Step 5: Training annotation type models")
        if not self._train_annotation_type_models(episodes, base_model):
            logger.error("Failed to train annotation type models")
            return False
        
        # Step 6: Final verification of original code preservation
        logger.info("Step 6: Final verification of original code preservation")
        if not self.verify_original_code_preservation():
            logger.error("Original code was modified during training")
            return False
        
        logger.info("🎉 Improved augment-first training pipeline completed successfully")
        return True
    
    def _augment_original_code_with_full_context(self) -> bool:
        """Augment the original code with full project context preserved"""
        try:
            from enhanced_semantic_augment_slices import EnhancedSemanticTransformer as SemanticTransformer
            
            logger.info(f"🔄 Augmenting original code with {self.augmentation_factor} variants per file")
            logger.info("📋 Maintaining full project structure for each variant")
            
            transformer = SemanticTransformer(seed=42)
            augmented_count = 0
            
            # Get all Java files in the project
            java_files = []
            for root, dirs, files in os.walk(self.project_root):
                for file in files:
                    if file.endswith('.java'):
                        java_files.append(os.path.join(root, file))
            
            logger.info(f"📁 Found {len(java_files)} Java files to augment")
            
            # Generate variants
            for variant_idx in range(self.augmentation_factor):
                variant_dir = os.path.join(self.augmented_code_dir, f"variant_{variant_idx}")
                os.makedirs(variant_dir, exist_ok=True)
                
                # Copy entire project structure for this variant
                variant_project_dir = os.path.join(variant_dir, "project")
                shutil.copytree(self.project_root, variant_project_dir, dirs_exist_ok=True)
                
                # Apply semantic transformations to each Java file in this variant
                variant_augmented_count = 0
                for java_file in java_files:
                    rel_path = os.path.relpath(java_file, self.project_root)
                    variant_java_file = os.path.join(variant_project_dir, rel_path)
                    
                    if os.path.exists(variant_java_file):
                        # Apply semantic transformations
                        augmented_content = transformer.transform_file(variant_java_file, variant_idx)
                        with open(variant_java_file, 'w') as f:
                            f.write(augmented_content)
                        variant_augmented_count += 1
                        augmented_count += 1
                
                logger.info(f"✅ Variant {variant_idx}: augmented {variant_augmented_count} files")
            
            # Verify augmentation
            total_variants = len(os.listdir(self.augmented_code_dir))
            logger.info(f"📊 Generated {total_variants} variants with {augmented_count} total augmented files")
            
            if total_variants >= self.augmentation_factor:
                logger.info("✅ Original code augmentation with full context completed successfully")
                return True
            else:
                logger.warning(f"⚠️ Augmentation may be incomplete: expected {self.augmentation_factor}, got {total_variants}")
                return True  # Still proceed
                
        except Exception as e:
            logger.error(f"❌ Error augmenting original code: {e}")
            return False
    
    def _slice_augmented_variants_with_full_context(self) -> bool:
        """Slice each augmented variant with full project context"""
        try:
            logger.info(f"🔪 Slicing {self.augmentation_factor} variants with full context using {self.slicer_type} slicer")
            
            total_slices = 0
            
            # Process each augmented variant
            for variant_dir in os.listdir(self.augmented_code_dir):
                variant_path = os.path.join(self.augmented_code_dir, variant_dir)
                if not os.path.isdir(variant_path):
                    continue
                
                # Each variant has its own project directory with full context
                variant_project_path = os.path.join(variant_path, "project")
                if not os.path.exists(variant_project_path):
                    logger.warning(f"⚠️ Variant project directory not found: {variant_project_path}")
                    continue
                
                # Create slice output directory for this variant
                variant_slice_dir = os.path.join(self.slices_dir, variant_dir)
                os.makedirs(variant_slice_dir, exist_ok=True)
                
                # Run slicing on this variant with full project context
                if self._slice_single_variant_with_full_context(variant_project_path, variant_slice_dir):
                    # Count slices generated for this variant
                    variant_slices = len(glob.glob(os.path.join(variant_slice_dir, '**/*.java'), recursive=True))
                    total_slices += variant_slices
                    logger.info(f"✅ Generated {variant_slices} slices for variant {variant_dir}")
                else:
                    logger.warning(f"⚠️ Failed to slice variant {variant_dir}")
            
            logger.info(f"📊 Total slices generated: {total_slices}")
            
            if total_slices > 0:
                logger.info("✅ Augmented variant slicing with full context completed successfully")
                return True
            else:
                logger.error("❌ No slices generated from any variant")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error slicing augmented variants: {e}")
            return False
    
    def _slice_single_variant_with_full_context(self, variant_project_path: str, output_dir: str) -> bool:
        """Slice a single augmented variant with full project context"""
        try:
            # Use the existing slicing infrastructure
            from pipeline import run_slicing
            
            # Create a temporary warnings file for this variant
            temp_warnings = os.path.join(output_dir, "temp_warnings.out")
            shutil.copy2(self.warnings_file, temp_warnings)
            
            # Run slicing on this variant with full project context
            success = run_slicing(
                project_root=variant_project_path,  # Full project context
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
            logger.error(f"❌ Error slicing variant {variant_project_path}: {e}")
            return False
    
    def _generate_cfgs_from_slices(self) -> bool:
        """Generate CFGs from all slices"""
        try:
            logger.info("🔗 Generating CFGs from all slices")
            
            # Use existing CFG generation infrastructure
            from pipeline import run_cfg_generation
            
            success = run_cfg_generation(self.slices_dir, self.cfg_dir)
            
            if success:
                logger.info("✅ CFG generation completed successfully")
                return True
            else:
                logger.error("❌ CFG generation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error generating CFGs: {e}")
            return False
    
    def _train_annotation_type_models(self, episodes: int, base_model: str) -> bool:
        """Train annotation type models"""
        try:
            logger.info(f"🎯 Training annotation type models with {episodes} episodes")
            
            # Use existing training infrastructure
            from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
            
            # Create pipeline instance
            pipeline = SimpleAnnotationTypePipeline(
                project_root=self.project_root,
                warnings_file=self.warnings_file,
                cfwr_root=self.cfwr_root,
                mode='train',
                augment_first=True  # Use augment-first approach
            )
            
            # Run training
            success = pipeline.run_training_pipeline(episodes, base_model)
            
            if success:
                logger.info("✅ Annotation type model training completed successfully")
                return True
            else:
                logger.error("❌ Annotation type model training failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error training annotation type models: {e}")
            return False
    
    def run_prediction_pipeline(self, target_file=None):
        """Run the prediction pipeline"""
        logger.info("🔮 Starting improved augment-first prediction pipeline")
        
        # Verify original code preservation
        if not self.verify_original_code_preservation():
            logger.error("Original code was modified")
            return False
        
        # Use existing prediction infrastructure
        from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
        
        pipeline = SimpleAnnotationTypePipeline(
            project_root=self.project_root,
            warnings_file=self.warnings_file,
            cfwr_root=self.cfwr_root,
            mode='predict',
            augment_first=True
        )
        
        return pipeline.run_prediction_pipeline(target_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Improved Augment-First Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='train',
                       help='Pipeline mode: train, predict, or both')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index',
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/GenDATA/index1.out',
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/GenDATA',
                       help='Root directory of CFWR project')
    parser.add_argument('--augmentation_factor', type=int, default=50,
                       help='Number of variants to generate per file')
    parser.add_argument('--slicer_type', default='soot', choices=['soot', 'specimin', 'cf'],
                       help='Slicer type to use')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--base_model', default='enhanced_causal', 
                       choices=['gcn', 'gbt', 'causal', 'enhanced_causal', 'hgt', 'gcsn', 'dg2n'],
                       help='Base model type')
    parser.add_argument('--target_file', 
                       help='Specific Java file to process (for prediction mode)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ImprovedAugmentFirstPipeline(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        augmentation_factor=args.augmentation_factor,
        slicer_type=args.slicer_type
    )
    
    # Run pipeline
    success = True
    
    if args.mode in ['train', 'both']:
        logger.info("🚀 Starting training mode")
        success = pipeline.run_training_pipeline(args.episodes, args.base_model)
    
    if args.mode in ['predict', 'both'] and success:
        logger.info("🔮 Starting prediction mode")
        success = pipeline.run_prediction_pipeline(args.target_file)
    
    if success:
        logger.info("🎉 Improved augment-first pipeline completed successfully")
        return 0
    else:
        logger.error("❌ Improved augment-first pipeline failed")
        return 1

if __name__ == '__main__':
    exit(main())



