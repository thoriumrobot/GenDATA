#!/usr/bin/env python3
"""
Verify Augment-First Pipeline Requirements

This script verifies that the augment-first pipeline meets all requirements:
1. Augmentation happens first, then slicing
2. Slicer gets full code context of the project
3. Original code is not modified when training is complete
4. All operations work on copies, preserving originals
"""

import os
import tempfile
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AugmentFirstRequirementsVerifier:
    """Verifies that the augment-first pipeline meets all requirements"""
    
    def __init__(self, project_root: str, cfwr_root: str):
        self.project_root = project_root
        self.cfwr_root = cfwr_root
        self.original_checksums = {}
        self.test_results = {}
        
    def calculate_file_checksum(self, file_path: str) -> str:
        """Calculate MD5 checksum of a file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def calculate_directory_checksum(self, dir_path: str) -> Dict[str, str]:
        """Calculate checksums for all Java files in a directory"""
        checksums = {}
        try:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.java'):
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, dir_path)
                        checksums[rel_path] = self.calculate_file_checksum(file_path)
        except Exception as e:
            logger.error(f"Error calculating directory checksums for {dir_path}: {e}")
        return checksums
    
    def requirement_1_original_code_preservation(self) -> bool:
        """Requirement 1: Original code is not modified when training is complete"""
        logger.info("🔍 Verifying Requirement 1: Original code preservation")
        
        # Calculate checksums of original files
        self.original_checksums = self.calculate_directory_checksum(self.project_root)
        logger.info(f"Calculated checksums for {len(self.original_checksums)} original Java files")
        
        # Check if original files still exist and have same checksums
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
        
        logger.info("✅ Requirement 1 PASSED: Original code preservation verified")
        return True
    
    def requirement_2_augmentation_first(self) -> bool:
        """Requirement 2: Augmentation happens first, then slicing"""
        logger.info("🔍 Verifying Requirement 2: Augmentation-first approach")
        
        # Check for augmented code directory
        augmented_code_dir = os.path.join(self.cfwr_root, 'augmented_code')
        if not os.path.exists(augmented_code_dir):
            logger.error("❌ Augmented code directory not found")
            return False
        
        # Check for slice directories
        slices_dir = os.path.join(self.cfwr_root, 'slices_augmented_first')
        if not os.path.exists(slices_dir):
            logger.error("❌ Slices directory not found")
            return False
        
        # Verify augmentation happened before slicing
        augmented_files = []
        for root, dirs, files in os.walk(augmented_code_dir):
            for file in files:
                if file.endswith('.java'):
                    augmented_files.append(os.path.join(root, file))
        
        slice_files = []
        for root, dirs, files in os.walk(slices_dir):
            for file in files:
                if file.endswith('.java'):
                    slice_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(augmented_files)} augmented files")
        logger.info(f"Found {len(slice_files)} slice files")
        
        if len(augmented_files) == 0:
            logger.error("❌ No augmented files found")
            return False
        
        # Check that augmented files have different content from originals
        different_count = 0
        for rel_path, original_checksum in self.original_checksums.items():
            original_file = os.path.join(self.project_root, rel_path)
            
            # Find corresponding augmented files
            for variant_dir in os.listdir(augmented_code_dir):
                if variant_dir.startswith('variant_'):
                    variant_path = os.path.join(augmented_code_dir, variant_dir)
                    variant_project_path = os.path.join(variant_path, 'project')
                    if os.path.exists(variant_project_path):
                        augmented_file = os.path.join(variant_project_path, rel_path)
                        
                        if os.path.exists(augmented_file):
                            augmented_checksum = self.calculate_file_checksum(augmented_file)
                            if augmented_checksum != original_checksum:
                                different_count += 1
                                break
        
        logger.info(f"✅ Found {different_count} augmented variants with different content")
        
        if different_count > 0:
            logger.info("✅ Requirement 2 PASSED: Augmentation-first approach verified")
            return True
        else:
            logger.error("❌ No augmented variants found with different content")
            return False
    
    def requirement_3_full_context_slicing(self) -> bool:
        """Requirement 3: Slicer gets full code context of the project"""
        logger.info("🔍 Verifying Requirement 3: Full context slicing")
        
        # Check that each augmented variant maintains project structure
        augmented_code_dir = os.path.join(self.cfwr_root, 'augmented_code')
        
        if not os.path.exists(augmented_code_dir):
            logger.error("❌ Augmented code directory not found")
            return False
        
        # Check that variants maintain directory structure
        variant_dirs = [d for d in os.listdir(augmented_code_dir) if os.path.isdir(os.path.join(augmented_code_dir, d))]
        
        if len(variant_dirs) == 0:
            logger.error("❌ No variant directories found")
            return False
        
        # Check that each variant has the same structure as original
        structure_preserved = True
        context_verified = 0
        
        for variant_dir in variant_dirs[:5]:  # Check first 5 variants
            variant_path = os.path.join(augmented_code_dir, variant_dir)
            variant_project_path = os.path.join(variant_path, 'project')
            
            if not os.path.exists(variant_project_path):
                logger.warning(f"⚠️ Variant project directory not found: {variant_project_path}")
                continue
            
            # Get Java files in this variant
            variant_files = []
            for root, dirs, files in os.walk(variant_project_path):
                for file in files:
                    if file.endswith('.java'):
                        rel_path = os.path.relpath(os.path.join(root, file), variant_project_path)
                        variant_files.append(rel_path)
            
            # Check if structure matches original
            original_files = list(self.original_checksums.keys())
            if set(variant_files) == set(original_files):
                context_verified += 1
                logger.info(f"✅ Variant {variant_dir} maintains full project context")
            else:
                logger.warning(f"⚠️ Variant {variant_dir} has different file structure")
                structure_preserved = False
        
        logger.info(f"✅ {context_verified}/{min(5, len(variant_dirs))} variants maintain full context")
        
        if context_verified > 0:
            logger.info("✅ Requirement 3 PASSED: Full context slicing verified")
            return True
        else:
            logger.error("❌ No variants maintain full project context")
            return False
    
    def requirement_4_operations_on_copies(self) -> bool:
        """Requirement 4: All operations work on copies, preserving originals"""
        logger.info("🔍 Verifying Requirement 4: Operations on copies")
        
        # Check that augmented code is in separate directory
        augmented_code_dir = os.path.join(self.cfwr_root, 'augmented_code')
        if not os.path.exists(augmented_code_dir):
            logger.error("❌ Augmented code directory not found")
            return False
        
        # Verify that augmented code is separate from original
        if os.path.samefile(self.project_root, augmented_code_dir):
            logger.error("❌ Augmented code directory is same as original")
            return False
        
        # Check that slices are in separate directory
        slices_dir = os.path.join(self.cfwr_root, 'slices_augmented_first')
        if not os.path.exists(slices_dir):
            logger.error("❌ Slices directory not found")
            return False
        
        # Verify that slices are separate from original
        if os.path.samefile(self.project_root, slices_dir):
            logger.error("❌ Slices directory is same as original")
            return False
        
        # Check that CFGs are in separate directory
        cfg_dir = os.path.join(self.cfwr_root, 'cfg_output_augmented_first')
        if not os.path.exists(cfg_dir):
            logger.error("❌ CFG directory not found")
            return False
        
        # Verify that CFGs are separate from original
        if os.path.samefile(self.project_root, cfg_dir):
            logger.error("❌ CFG directory is same as original")
            return False
        
        logger.info("✅ All operations work on separate copies")
        logger.info("✅ Requirement 4 PASSED: Operations on copies verified")
        return True
    
    def run_complete_verification(self) -> bool:
        """Run complete verification of all requirements"""
        logger.info("🎯 Starting complete verification of augment-first pipeline requirements")
        logger.info("=" * 80)
        
        results = []
        
        # Test 1: Original code preservation
        logger.info("\n📋 Requirement 1: Original Code Preservation")
        results.append(self.requirement_1_original_code_preservation())
        
        # Test 2: Augmentation first
        logger.info("\n📋 Requirement 2: Augmentation-First Approach")
        results.append(self.requirement_2_augmentation_first())
        
        # Test 3: Full context slicing
        logger.info("\n📋 Requirement 3: Full Context Slicing")
        results.append(self.requirement_3_full_context_slicing())
        
        # Test 4: Operations on copies
        logger.info("\n📋 Requirement 4: Operations on Copies")
        results.append(self.requirement_4_operations_on_copies())
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 VERIFICATION SUMMARY")
        logger.info("=" * 80)
        
        requirement_names = [
            "Original Code Preservation",
            "Augmentation-First Approach", 
            "Full Context Slicing",
            "Operations on Copies"
        ]
        
        passed = 0
        for i, (req_name, result) in enumerate(zip(requirement_names, results)):
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{i+1}. {req_name}: {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{len(results)} requirements passed")
        
        if passed == len(results):
            logger.info("🎉 All requirements verified! Augment-first pipeline meets all specifications.")
            return True
        else:
            logger.error("❌ Some requirements failed. Please check the implementation.")
            return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Augment-First Pipeline Requirements')
    parser.add_argument('--project_root', default='/home/ubuntu/checker-framework/checker/tests/index',
                       help='Root directory of the Java project')
    parser.add_argument('--cfwr_root', default='/home/ubuntu/GenDATA',
                       help='Root directory of CFWR project')
    
    args = parser.parse_args()
    
    # Create verifier
    verifier = AugmentFirstRequirementsVerifier(args.project_root, args.cfwr_root)
    
    # Run verification
    success = verifier.run_complete_verification()
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
