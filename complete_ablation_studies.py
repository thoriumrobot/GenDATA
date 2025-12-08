#!/usr/bin/env python3
"""
Complete both ablation studies with real results:
1. Augmentation comparison (with vs without augmentation)
2. Transformation ablation (all transformations individually)
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Complete ablation studies')
    parser.add_argument('--slices_dir', help='Original slices directory', default='slices_specimin')
    parser.add_argument('--cfg_dir', help='Augmented CFG directory', default='cfg_output_specimin')
    parser.add_argument('--episodes', type=int, default=10, help='Training episodes')
    parser.add_argument('--device', default='cpu', help='Device for training')
    parser.add_argument('--skip_cfg_generation', action='store_true', help='Skip CFG generation (use existing)')
    parser.add_argument('--skip_augmentation_comparison', action='store_true', help='Skip augmentation comparison')
    parser.add_argument('--skip_transformation_ablation', action='store_true', help='Skip transformation ablation')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("COMPLETE ABLATION STUDIES")
    logger.info("=" * 80)
    
    # Step 1: Generate CFG directories if needed
    if not args.skip_cfg_generation:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Generating CFG directories")
        logger.info("=" * 80)
        
        # Generate non-augmented CFGs
        logger.info("Generating non-augmented CFG directory...")
        cmd = [
            sys.executable, 'generate_ablation_cfg_directories.py',
            '--slices_dir', args.slices_dir,
            '--generate_no_aug',
            '--output_base', 'ablation_studies'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to generate non-augmented CFGs: {result.stderr}")
        else:
            logger.info("✅ Non-augmented CFG directory generated")
        
        # Generate transformation-ablated CFGs (sequentially)
        logger.info("Generating transformation-ablated CFG directories (this will take a while)...")
        cmd = [
            sys.executable, 'generate_ablation_cfg_directories.py',
            '--slices_dir', args.slices_dir,
            '--generate_transforms',
            '--output_base', 'ablation_studies'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to generate transformation-ablated CFGs: {result.stderr}")
        else:
            logger.info("✅ Transformation-ablated CFG directories generated")
    else:
        logger.info("Skipping CFG generation (using existing directories)")
    
    # Step 2: Run augmentation comparison study
    if not args.skip_augmentation_comparison:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Augmentation Comparison Study")
        logger.info("=" * 80)
        
        from run_augmentation_comparison_study import AugmentationComparisonStudy
        
        study = AugmentationComparisonStudy(
            output_dir='ablation_augmentation_comparison_final',
            balanced_dataset_dir='real_balanced_datasets',
            cfg_dir=args.cfg_dir,
            cfg_dir_no_aug='ablation_studies/no_augmentation/cfg_output',
            episodes=args.episodes,
            device=args.device
        )
        
        results = study.run_comparison_study()
        
        # Save results
        results_file = Path('ablation_augmentation_comparison_final') / 'augmentation_comparison_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Augmentation comparison study completed. Results saved to {results_file}")
    else:
        logger.info("Skipping augmentation comparison study")
    
    # Step 3: Run transformation ablation study
    if not args.skip_transformation_ablation:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Transformation Ablation Study")
        logger.info("=" * 80)
        
        from run_transformation_ablation_final import TransformationAblationFinal
        
        # Get all transformations
        try:
            from jdt_semantic_transformer import JdtSemanticTransformer
            transformer = JdtSemanticTransformer()
            all_transforms = transformer.get_available_transformations('enhanced') + transformer.get_available_transformations('simple')
        except:
            # Fallback
            all_transforms = [
                'loop_conversion', 'guard_reversal', 'mathematical_expression', 'logical_expression',
                'ternary_operator', 'switch_statement', 'variable_operation', 'brace_normalization',
                'string_concatenation', 'numeric_literal',
                'simple_method_call', 'simple_assignment', 'simple_conditional',
                'simple_array_access', 'simple_return_statement', 'simple_variable_declaration',
                'simple_constructor_call', 'simple_field_access', 'simple_string_operation',
                'simple_numeric_operation'
            ]
        
        study = TransformationAblationFinal(
            output_dir='ablation_transformations_final',
            balanced_dataset_dir='real_balanced_datasets',
            cfg_dir=args.cfg_dir,
            cfg_dir_base_pattern='ablation_studies/ablate_{transform}/cfg_output',
            episodes=args.episodes,
            device=args.device
        )
        
        results = study.run_full_transformation_ablation(transformations=all_transforms)
        
        # Save results
        results_file = Path('ablation_transformations_final') / 'transformation_ablation_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Transformation ablation study completed. Results saved to {results_file}")
    else:
        logger.info("Skipping transformation ablation study")
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL ABLATION STUDIES COMPLETED")
    logger.info("=" * 80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

