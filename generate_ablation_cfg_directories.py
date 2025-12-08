#!/usr/bin/env python3
"""
Generate CFG directories for ablation studies:
1. Non-augmented CFGs for augmentation comparison
2. CFGs with each transformation disabled for transformation ablation
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get available transformations from JDT transformer
try:
    from jdt_semantic_transformer import JdtSemanticTransformer
    transformer = JdtSemanticTransformer()
    ENHANCED_TRANSFORMATIONS = transformer.get_available_transformations('enhanced')
    SIMPLE_TRANSFORMATIONS = transformer.get_available_transformations('simple')
    ALL_TRANSFORMATIONS = ENHANCED_TRANSFORMATIONS + SIMPLE_TRANSFORMATIONS
    logger.info(f"Found {len(ENHANCED_TRANSFORMATIONS)} enhanced + {len(SIMPLE_TRANSFORMATIONS)} simple = {len(ALL_TRANSFORMATIONS)} total transformations")
except Exception as e:
    logger.warning(f"Could not load transformations from JDT: {e}")
    # Fallback list
    ENHANCED_TRANSFORMATIONS = [
        'loop_conversion', 'guard_reversal', 'mathematical_expression', 'logical_expression',
        'ternary_operator', 'switch_statement', 'variable_operation', 'brace_normalization',
        'string_concatenation', 'numeric_literal'
    ]
    SIMPLE_TRANSFORMATIONS = [
        'simple_method_call', 'simple_assignment', 'simple_conditional',
        'simple_array_access', 'simple_return_statement', 'simple_variable_declaration',
        'simple_constructor_call', 'simple_field_access', 'simple_string_operation',
        'simple_numeric_operation'
    ]
    ALL_TRANSFORMATIONS = ENHANCED_TRANSFORMATIONS + SIMPLE_TRANSFORMATIONS

def find_original_slices_dir() -> str:
    """Find the original (non-augmented) slices directory"""
    potential_dirs = [
        'slices_specimin',
        'slices',
        'slices_cf',
        'slices_wala'
    ]
    
    for dir_name in potential_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            # Check if it has Java files
            java_files = list(Path(dir_name).rglob('*.java'))
            if java_files:
                logger.info(f"Found original slices directory: {dir_name} ({len(java_files)} Java files)")
                return dir_name
    
    raise FileNotFoundError(f"Could not find original slices directory. Checked: {potential_dirs}")

def generate_cfgs_from_slices(slices_dir: str, cfg_output_dir: str) -> bool:
    """Generate CFGs from slices using pipeline.py"""
    logger.info(f"Generating CFGs from {slices_dir} to {cfg_output_dir}")
    
    os.makedirs(cfg_output_dir, exist_ok=True)
    
    try:
        # Use pipeline's CFG generation
        from pipeline import run_cfg_generation
        run_cfg_generation(slices_dir, cfg_output_dir)
        
        # Verify CFGs were generated
        cfg_files = list(Path(cfg_output_dir).rglob('*.json'))
        if cfg_files:
            logger.info(f"✅ Generated {len(cfg_files)} CFG files in {cfg_output_dir}")
            return True
        else:
            logger.warning(f"⚠️  No CFG files found in {cfg_output_dir}")
            return False
    except Exception as e:
        logger.error(f"❌ Error generating CFGs: {e}")
        return False

def generate_no_augmentation_cfgs(slices_dir: str, output_dir: str = 'ablation_studies/no_augmentation') -> bool:
    """Generate CFGs from non-augmented slices"""
    logger.info("=" * 80)
    logger.info("Generating non-augmented CFG directory for augmentation comparison")
    logger.info("=" * 80)
    
    cfg_output_dir = os.path.join(output_dir, 'cfg_output')
    os.makedirs(cfg_output_dir, exist_ok=True)
    
    success = generate_cfgs_from_slices(slices_dir, cfg_output_dir)
    
    if success:
        # Verify CFG directory exists and has files
        cfg_files = list(Path(cfg_output_dir).rglob('*.json'))
        if cfg_files:
            logger.info(f"✅ Non-augmented CFG directory verified: {len(cfg_files)} CFG files")
            return True
        else:
            logger.warning(f"⚠️  CFG directory created but no files found: {cfg_output_dir}")
            return False
    else:
        logger.error(f"❌ Failed to generate non-augmented CFGs")
        return False

def generate_transformation_ablated_cfgs(slices_dir: str, transform_name: str, 
                                         output_base: str = 'ablation_studies') -> bool:
    """Generate CFGs with a specific transformation disabled"""
    logger.info(f"Generating CFGs with '{transform_name}' disabled")
    
    # Create slices directory for this transformation
    slices_output_dir = os.path.join(output_base, f'ablate_{transform_name}', 'slices')
    cfg_output_dir = os.path.join(output_base, f'ablate_{transform_name}', 'cfg_output')
    
    os.makedirs(slices_output_dir, exist_ok=True)
    os.makedirs(cfg_output_dir, exist_ok=True)
    
    # Augment slices with this transformation disabled
    logger.info(f"Augmenting slices with '{transform_name}' disabled...")
    try:
        cmd = [
            sys.executable, 'enhanced_semantic_augment_slices.py',
            slices_dir,  # positional input
            slices_output_dir,  # positional output
            '--variants', '10',
            '--disabled', transform_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            logger.error(f"Augmentation failed: {result.stderr}")
            return False
        
        # Check if slices were generated
        java_files = list(Path(slices_output_dir).rglob('*.java'))
        if not java_files:
            logger.warning(f"No augmented slices generated for {transform_name}")
            return False
        
        logger.info(f"Generated {len(java_files)} augmented slice files")
        
        # Generate CFGs from augmented slices
        return generate_cfgs_from_slices(slices_output_dir, cfg_output_dir)
        
    except Exception as e:
        logger.error(f"Error generating ablated CFGs for {transform_name}: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate CFG directories for ablation studies')
    parser.add_argument('--slices_dir', help='Original slices directory (auto-detected if not provided)')
    parser.add_argument('--generate_no_aug', action='store_true', help='Generate non-augmented CFG directory')
    parser.add_argument('--generate_transforms', action='store_true', help='Generate CFG directories for all transformations')
    parser.add_argument('--transform', help='Generate CFG directory for specific transformation only')
    parser.add_argument('--output_base', default='ablation_studies', help='Base directory for output')
    
    args = parser.parse_args()
    
    # Find original slices directory
    if args.slices_dir:
        slices_dir = args.slices_dir
    else:
        slices_dir = find_original_slices_dir()
    
    logger.info(f"Using slices directory: {slices_dir}")
    
    success_count = 0
    total_count = 0
    
    # Generate non-augmented CFGs
    if args.generate_no_aug:
        total_count += 1
        if generate_no_augmentation_cfgs(slices_dir, args.output_base):
            success_count += 1
            logger.info("✅ Non-augmented CFG directory generated successfully")
        else:
            logger.error("❌ Failed to generate non-augmented CFG directory")
    
    # Generate transformation-ablated CFGs
    if args.generate_transforms or args.transform:
        transforms_to_process = [args.transform] if args.transform else ALL_TRANSFORMATIONS
        
        logger.info(f"Processing {len(transforms_to_process)} transformations...")
        
        for i, transform in enumerate(transforms_to_process, 1):
            total_count += 1
            logger.info(f"\n[{i}/{len(transforms_to_process)}] Processing: {transform}")
            
            if generate_transformation_ablated_cfgs(slices_dir, transform, args.output_base):
                success_count += 1
                logger.info(f"✅ CFG directory for '{transform}' generated successfully")
            else:
                logger.error(f"❌ Failed to generate CFG directory for '{transform}'")
    
    logger.info("=" * 80)
    logger.info(f"Summary: {success_count}/{total_count} directories generated successfully")
    logger.info("=" * 80)
    
    return 0 if success_count == total_count else 1

if __name__ == '__main__':
    sys.exit(main())

