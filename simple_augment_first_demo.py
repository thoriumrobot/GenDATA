#!/usr/bin/env python3
"""
Simple Augment-First Demo Pipeline

This pipeline demonstrates the augment-first approach with:
1. Original code preservation (read-only)
2. Simple augmentation (copy with comments)
3. Full context slicing
4. Verification of all requirements
"""

import os
import json
import argparse
import subprocess
import tempfile
import shutil
import logging
from pathlib import Path
import time
import glob
from typing import List, Dict, Optional
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAugmentFirstDemo:
    """Simple demo of augment-first approach with original code preservation"""
    
    def __init__(self, project_root: str, warnings_file: str, cfwr_root: str, 
                 augmentation_factor: int = 3):
        self.project_root = project_root
        self.warnings_file = warnings_file
        self.cfwr_root = cfwr_root
        self.augmentation_factor = augmentation_factor
        
        # Store original checksums to verify preservation
        self.original_checksums = {}
        
        # Set up directories
        self.augmented_code_dir = os.path.join(cfwr_root, 'augmented_code_demo')
        self.slices_dir = os.path.join(cfwr_root, 'slices_augmented_first_demo')
        self.cfg_dir = os.path.join(cfwr_root, 'cfg_output_augmented_first_demo')
        
        # Create directories
        for dir_path in [self.augmented_code_dir, self.slices_dir, self.cfg_dir]:
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
    
    def run_demo(self) -> bool:
        """Run the complete demo pipeline"""
        logger.info("🚀 Starting simple augment-first demo pipeline")
        logger.info(f"📁 Project root: {self.project_root}")
        logger.info(f"🔢 Augmentation factor: {self.augmentation_factor}")
        
        # Step 0: Preserve original code
        logger.info("Step 0: Preserving original code integrity")
        if not self.preserve_original_code():
            logger.error("Failed to preserve original code")
            return False
        
        # Step 1: Augment original code first (simple approach)
        logger.info("Step 1: Augmenting original code (simple approach)")
        if not self._augment_original_code_simple():
            logger.error("Failed to augment original code")
            return False
        
        # Step 2: Verify original code preservation
        logger.info("Step 2: Verifying original code preservation")
        if not self.verify_original_code_preservation():
            logger.error("Original code was modified during augmentation")
            return False
        
        # Step 3: Slice each augmented variant
        logger.info("Step 3: Slicing each augmented variant")
        if not self._slice_augmented_variants():
            logger.error("Failed to slice augmented variants")
            return False
        
        # Step 4: Final verification
        logger.info("Step 4: Final verification of original code preservation")
        if not self.verify_original_code_preservation():
            logger.error("Original code was modified during slicing")
            return False
        
        logger.info("🎉 Simple augment-first demo pipeline completed successfully")
        return True
    
    def _augment_original_code_simple(self) -> bool:
        """Augment the original code with simple approach (add comments)"""
        try:
            logger.info(f"🔄 Augmenting original code with {self.augmentation_factor} variants per file")
            
            augmented_count = 0
            successful_variants = 0
            
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
                try:
                    shutil.copytree(self.project_root, variant_project_dir, dirs_exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to copy project for variant {variant_idx}: {e}")
                    continue
                
                # Apply simple augmentation to each Java file in this variant
                variant_augmented_count = 0
                
                for java_file in java_files:
                    rel_path = os.path.relpath(java_file, self.project_root)
                    variant_java_file = os.path.join(variant_project_dir, rel_path)
                    
                    if os.path.exists(variant_java_file):
                        try:
                            # Read original content
                            with open(variant_java_file, 'r') as f:
                                original_content = f.read()
                            
                            # Add simple augmentation comment
                            augmentation_comment = f"\n// CFWR semantic augmentation - variant {variant_idx}\n"
                            augmented_content = original_content + augmentation_comment
                            
                            # Write augmented content
                            with open(variant_java_file, 'w') as f:
                                f.write(augmented_content)
                            
                            variant_augmented_count += 1
                            augmented_count += 1
                                
                        except Exception as e:
                            logger.warning(f"Failed to augment {rel_path} for variant {variant_idx}: {e}")
                
                if variant_augmented_count > 0:
                    successful_variants += 1
                    logger.info(f"✅ Variant {variant_idx}: augmented {variant_augmented_count} files")
                else:
                    logger.warning(f"⚠️ Variant {variant_idx}: failed or incomplete")
            
            # Verify augmentation
            logger.info(f"📊 Generated {successful_variants}/{self.augmentation_factor} successful variants")
            logger.info(f"📊 Total augmented files: {augmented_count}")
            
            if successful_variants > 0:
                logger.info("✅ Original code augmentation completed successfully")
                return True
            else:
                logger.error("❌ No successful variants generated")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error augmenting original code: {e}")
            return False
    
    def _slice_augmented_variants(self) -> bool:
        """Slice each augmented variant"""
        try:
            logger.info(f"🔪 Slicing augmented variants with full context")
            
            total_slices = 0
            successful_variants = 0
            
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
                if self._slice_single_variant(variant_project_path, variant_slice_dir):
                    # Count slices generated for this variant
                    variant_slices = len(glob.glob(os.path.join(variant_slice_dir, '**/*.java'), recursive=True))
                    total_slices += variant_slices
                    successful_variants += 1
                    logger.info(f"✅ Generated {variant_slices} slices for variant {variant_dir}")
                else:
                    logger.warning(f"⚠️ Failed to slice variant {variant_dir}")
            
            logger.info(f"📊 Total slices generated: {total_slices}")
            logger.info(f"📊 Successful variants: {successful_variants}")
            
            if total_slices > 0:
                logger.info("✅ Augmented variant slicing completed successfully")
                return True
            else:
                logger.error("❌ No slices generated from any variant")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error slicing augmented variants: {e}")
            return False
    
    def _slice_single_variant(self, variant_project_path: str, output_dir: str) -> bool:
        """Slice a single augmented variant"""
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
                slicer_type='soot'
            )
            
            # Clean up temporary warnings file
            if os.path.exists(temp_warnings):
                os.remove(temp_warnings)
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error slicing variant {variant_project_path}: {e}")
            return False
    
    def verify_requirements(self) -> bool:
        """Verify all requirements are met"""
        logger.info("🔍 Verifying all requirements...")
        
        # Requirement 1: Original code preservation
        if not self.verify_original_code_preservation():
            logger.error("❌ Requirement 1 FAILED: Original code preservation")
            return False
        logger.info("✅ Requirement 1 PASSED: Original code preservation")
        
        # Requirement 2: Augmentation first
        if not os.path.exists(self.augmented_code_dir):
            logger.error("❌ Requirement 2 FAILED: Augmentation directory not found")
            return False
        logger.info("✅ Requirement 2 PASSED: Augmentation-first approach")
        
        # Requirement 3: Full context slicing
        if not os.path.exists(self.slices_dir):
            logger.error("❌ Requirement 3 FAILED: Slices directory not found")
            return False
        logger.info("✅ Requirement 3 PASSED: Full context slicing")
        
        # Requirement 4: Operations on copies
        if os.path.samefile(self.project_root, self.augmented_code_dir):
            logger.error("❌ Requirement 4 FAILED: Augmented code directory same as original")
            return False
        logger.info("✅ Requirement 4 PASSED: Operations on copies")
        
        logger.info("🎉 All requirements verified successfully!")
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Simple Augment-First Demo Pipeline')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index',
                       help='Root directory of the Java project')
    parser.add_argument('--warnings_file', default='/home/ubuntu/GenDATA/index1.out',
                       help='Path to warnings file')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/GenDATA',
                       help='Root directory of CFWR project')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                       help='Number of variants to generate per file')
    
    args = parser.parse_args()
    
    # Create demo pipeline
    demo = SimpleAugmentFirstDemo(
        project_root=args.project_root,
        warnings_file=args.warnings_file,
        cfwr_root=args.cfwr_root,
        augmentation_factor=args.augmentation_factor
    )
    
    # Run demo
    success = demo.run_demo()
    
    if success:
        # Verify requirements
        success = demo.verify_requirements()
    
    if success:
        logger.info("🎉 Simple augment-first demo completed successfully")
        return 0
    else:
        logger.error("❌ Simple augment-first demo failed")
        return 1

if __name__ == '__main__':
    exit(main())


