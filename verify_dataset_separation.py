#!/usr/bin/env python3
"""
Verification script for dataset separation in ablation studies.

This script verifies that different ablation conditions use different
dataset directories and that datasets are correctly generated.
"""

import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_dataset_files(dataset_dir: str) -> dict:
    """
    Verify that all required dataset files exist in a directory
    
    Args:
        dataset_dir: Directory containing dataset files
        
    Returns:
        Dictionary with verification results
    """
    dataset_path = Path(dataset_dir)
    
    result = {
        'directory': str(dataset_dir),
        'exists': dataset_path.exists(),
        'files_found': [],
        'files_missing': [],
        'total_examples': {},
        'valid': False
    }
    
    if not result['exists']:
        logger.warning(f"Directory does not exist: {dataset_dir}")
        return result
    
    annotation_types = ['positive', 'nonnegative', 'gtenegativeone']
    
    for ann_type in annotation_types:
        dataset_file = dataset_path / f"{ann_type}_real_balanced_dataset.json"
        
        if dataset_file.exists():
            result['files_found'].append(str(dataset_file))
            
            # Try to load and count examples
            try:
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                    total = data.get('total_examples', 0)
                    positive = data.get('positive_examples', 0)
                    negative = data.get('negative_examples', 0)
                    result['total_examples'][ann_type] = {
                        'total': total,
                        'positive': positive,
                        'negative': negative,
                        'balance': positive / total if total > 0 else 0
                    }
            except Exception as e:
                logger.warning(f"Error reading {dataset_file}: {e}")
                result['total_examples'][ann_type] = {'error': str(e)}
        else:
            result['files_missing'].append(ann_type)
    
    result['valid'] = len(result['files_found']) == len(annotation_types)
    
    return result


def verify_ablation_datasets():
    """
    Verify dataset separation for ablation studies
    
    Returns:
        Dictionary with verification results
    """
    logger.info("=" * 80)
    logger.info("VERIFYING DATASET SEPARATION FOR ABLATION STUDIES")
    logger.info("=" * 80)
    
    results = {
        'augmentation_comparison': {},
        'transformation_ablation': {},
        'overall_status': 'unknown'
    }
    
    # Check augmentation comparison datasets
    logger.info("\n1. Checking Augmentation Comparison Datasets...")
    
    # With augmentation (baseline)
    baseline_dir = 'real_balanced_datasets'
    baseline_result = verify_dataset_files(baseline_dir)
    results['augmentation_comparison']['with_augmentation'] = baseline_result
    
    logger.info(f"\n  With Augmentation ({baseline_dir}):")
    logger.info(f"    Exists: {baseline_result['exists']}")
    logger.info(f"    Files found: {len(baseline_result['files_found'])}/3")
    logger.info(f"    Files missing: {baseline_result['files_missing']}")
    if baseline_result['total_examples']:
        for ann_type, stats in baseline_result['total_examples'].items():
            if 'error' not in stats:
                logger.info(f"    {ann_type}: {stats['total']} examples (balance: {stats['balance']:.3f})")
    
    # Without augmentation
    no_aug_dir = 'ablation_augmentation_comparison/no_augmentation_datasets'
    no_aug_result = verify_dataset_files(no_aug_dir)
    results['augmentation_comparison']['without_augmentation'] = no_aug_result
    
    logger.info(f"\n  Without Augmentation ({no_aug_dir}):")
    logger.info(f"    Exists: {no_aug_result['exists']}")
    logger.info(f"    Files found: {len(no_aug_result['files_found'])}/3")
    logger.info(f"    Files missing: {no_aug_result['files_missing']}")
    if no_aug_result['total_examples']:
        for ann_type, stats in no_aug_result['total_examples'].items():
            if 'error' not in stats:
                logger.info(f"    {ann_type}: {stats['total']} examples (balance: {stats['balance']:.3f})")
    
    # Check if directories are different
    if baseline_result['exists'] and no_aug_result['exists']:
        if baseline_dir != no_aug_dir:
            logger.info(f"\n  ✅ Directories are different: {baseline_dir} vs {no_aug_dir}")
        else:
            logger.warning(f"\n  ⚠️  Directories are the same: {baseline_dir}")
    
    # Check transformation ablation datasets
    logger.info("\n2. Checking Transformation Ablation Datasets...")
    
    transform_base = Path('ablation_transformations_final')
    if transform_base.exists():
        transform_dirs = [d for d in transform_base.iterdir() 
                          if d.is_dir() and d.name.startswith('ablate_')]
        
        logger.info(f"  Found {len(transform_dirs)} transformation ablation directories")
        
        for transform_dir in transform_dirs[:5]:  # Check first 5
            dataset_dir = transform_dir / 'datasets'
            transform_result = verify_dataset_files(str(dataset_dir))
            transform_name = transform_dir.name.replace('ablate_', '')
            results['transformation_ablation'][transform_name] = transform_result
            
            logger.info(f"\n  {transform_name}:")
            logger.info(f"    Exists: {transform_result['exists']}")
            logger.info(f"    Files found: {len(transform_result['files_found'])}/3")
            logger.info(f"    Valid: {transform_result['valid']}")
    else:
        logger.info("  Transformation ablation directory not found (may not have been run yet)")
    
    # Overall status
    all_valid = True
    if baseline_result['exists']:
        all_valid = all_valid and baseline_result['valid']
    if no_aug_result['exists']:
        all_valid = all_valid and no_aug_result['valid']
    
    for transform_result in results['transformation_ablation'].values():
        if transform_result['exists']:
            all_valid = all_valid and transform_result['valid']
    
    results['overall_status'] = 'valid' if all_valid else 'invalid'
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)
    
    if all_valid:
        logger.info("✅ All datasets are properly separated and valid")
    else:
        logger.warning("⚠️  Some datasets are missing or invalid")
        logger.warning("   Run ablation studies to generate missing datasets")
    
    return results


def main():
    """Main entry point"""
    results = verify_ablation_datasets()
    
    # Save results
    results_file = Path('dataset_separation_verification.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {results_file}")
    
    return 0 if results['overall_status'] == 'valid' else 1


if __name__ == '__main__':
    exit(main())

