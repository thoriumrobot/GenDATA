#!/usr/bin/env python3
"""
Create Training Datasets for GenDATA Checkers

This script generates training datasets (slices, CFGs, augmented code) for SQL Quotes
and Signature String checkers using their warning files, similar to how Lower Bound
Checker datasets are created.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GEN_DATA_ROOT = Path('/home/ubuntu/GenDATA')

def create_dataset_for_checker(checker_name: str, warnings_file: Path, project_root: str) -> bool:
    """
    Create training dataset for a specific checker.
    
    Args:
        checker_name: Name of the checker (e.g., 'sql_quotes', 'signature_string')
        warnings_file: Path to the warnings file for this checker
        project_root: Root directory of the test suite project
        
    Returns:
        True if dataset creation was successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"Creating training dataset for {checker_name}")
    logger.info("=" * 80)
    
    if not warnings_file.exists():
        logger.error(f"Warnings file not found: {warnings_file}")
        logger.error(f"Please run generate_checker_warning_files.py first to generate {warnings_file.name}")
        return False
    
    if not os.path.exists(project_root):
        logger.error(f"Project root not found: {project_root}")
        return False
    
    # Import pipeline
    try:
        from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
    except ImportError as e:
        logger.error(f"Failed to import SimpleAnnotationTypePipeline: {e}")
        return False
    
    logger.info(f"Warnings file: {warnings_file}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"CFWR root: {GEN_DATA_ROOT}")
    
    # Create pipeline instance
    # This will automatically use checker-specific directories based on warnings_file name
    pipeline = SimpleAnnotationTypePipeline(
        project_root=project_root,
        warnings_file=str(warnings_file),
        cfwr_root=str(GEN_DATA_ROOT),
        mode='train',
        device='auto',
        augment_first=True,
        disable_random_walk=False,
        run_checker_on_target=False  # Don't run checker again, we already have warnings
    )
    
    logger.info(f"Pipeline initialized with checker-specific directories:")
    logger.info(f"  Slices: {pipeline.slices_dir}")
    logger.info(f"  CFGs: {pipeline.cfg_dir}")
    if hasattr(pipeline, 'augmented_code_dir'):
        logger.info(f"  Augmented code: {pipeline.augmented_code_dir}")
    
    # Step 1: Generate slices from warnings
    logger.info("\n" + "-" * 80)
    logger.info("Step 1: Generating slices from warnings")
    logger.info("-" * 80)
    
    if not pipeline._generate_slices_with_soot():
        logger.error("Failed to generate slices")
        return False
    
    # Step 2: Generate CFGs from slices
    logger.info("\n" + "-" * 80)
    logger.info("Step 2: Generating CFGs from slices")
    logger.info("-" * 80)
    
    if not pipeline._generate_cfgs_from_slices():
        logger.error("Failed to generate CFGs")
        return False
    
    # Step 3: Augment original code (optional but recommended for better training)
    logger.info("\n" + "-" * 80)
    logger.info("Step 3: Augmenting original code")
    logger.info("-" * 80)
    
    if not pipeline._augment_original_code():
        logger.warning("Failed to augment original code, but continuing...")
        # Don't fail if augmentation fails, it's optional
    
    # Verify dataset creation
    logger.info("\n" + "-" * 80)
    logger.info("Verifying dataset creation")
    logger.info("-" * 80)
    
    slices_count = count_files(pipeline.slices_dir, '*.java')
    cfgs_count = count_files(pipeline.cfg_dir, '*.json')
    augmented_count = 0
    if hasattr(pipeline, 'augmented_code_dir') and pipeline.augmented_code_dir:
        augmented_count = count_files(pipeline.augmented_code_dir, '*.java')
    
    logger.info(f"Slices generated: {slices_count}")
    logger.info(f"CFGs generated: {cfgs_count}")
    logger.info(f"Augmented files: {augmented_count}")
    
    if slices_count == 0:
        logger.warning("⚠️ No slices generated. This may indicate:")
        logger.warning("  - No warnings in the warnings file")
        logger.warning("  - Slicing failed for all warnings")
        logger.warning("  - Project structure issues")
        return False
    
    if cfgs_count == 0:
        logger.warning("⚠️ No CFGs generated. This may indicate:")
        logger.warning("  - CFG generation failed")
        logger.warning("  - No valid slices to convert")
        return False
    
    logger.info("✅ Training dataset created successfully!")
    return True


def count_files(directory: str, pattern: str) -> int:
    """Count files matching pattern in directory (recursive)."""
    import glob
    if not os.path.exists(directory):
        return 0
    return len(glob.glob(os.path.join(directory, '**', pattern), recursive=True))


def main():
    """Main function to create training datasets for all checkers."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create training datasets for GenDATA checkers')
    parser.add_argument('--checker', choices=['sql_quotes', 'signature_string', 'all'],
                       default='all', help='Checker to create dataset for')
    parser.add_argument('--generate-warnings', action='store_true',
                       help='Generate warning files before creating datasets')
    
    args = parser.parse_args()
    
    # Generate warnings if requested
    if args.generate_warnings:
        logger.info("Generating warning files...")
        try:
            from generate_checker_warning_files import generate_all_warning_files
            generate_all_warning_files()
        except Exception as e:
            logger.error(f"Failed to generate warning files: {e}")
            return 1
    
    # Checker configurations
    checker_configs = {
        'sql_quotes': {
            'warnings_file': GEN_DATA_ROOT / 'sql_quotes_warnings.out',
            'project_root': '/home/ubuntu/checker-framework/checker/tests/sqlquotes',
            'name': 'SQL Quotes Checker'
        },
        'signature_string': {
            'warnings_file': GEN_DATA_ROOT / 'signature_string_warnings.out',
            'project_root': '/home/ubuntu/checker-framework/checker/tests/signature',
            'name': 'Signature String Checker'
        }
    }
    
    results = {}
    
    checkers_to_process = []
    if args.checker == 'all':
        checkers_to_process = list(checker_configs.keys())
    else:
        checkers_to_process = [args.checker]
    
    for checker_name in checkers_to_process:
        if checker_name not in checker_configs:
            logger.error(f"Unknown checker: {checker_name}")
            results[checker_name] = False
            continue
        
        config = checker_configs[checker_name]
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {config['name']}")
        logger.info(f"{'=' * 80}\n")
        
        # Check if test suite exists
        if not os.path.exists(config['project_root']):
            logger.warning(f"⚠️ Test suite not found: {config['project_root']}")
            logger.warning(f"   Skipping dataset creation for {checker_name}")
            results[checker_name] = False
            continue
        
        # Check if warnings file exists
        if not config['warnings_file'].exists():
            logger.warning(f"⚠️ Warnings file not found: {config['warnings_file']}")
            logger.warning(f"   Run: python3 generate_checker_warning_files.py")
            logger.warning(f"   Or use --generate-warnings flag")
            results[checker_name] = False
            continue
        
        # Create dataset
        success = create_dataset_for_checker(
            checker_name=checker_name,
            warnings_file=config['warnings_file'],
            project_root=config['project_root']
        )
        results[checker_name] = success
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Dataset Creation Summary")
    logger.info("=" * 80)
    
    for checker_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{checker_name}: {status}")
    
    all_success = all(results.values())
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

