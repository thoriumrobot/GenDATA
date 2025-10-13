#!/usr/bin/env python3
"""
Test Ablation Pipeline

Simple test script to validate the ablation study pipeline with a small subset.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ablation_pipeline():
    """Test the ablation pipeline with a small subset"""
    logger.info("Testing ablation pipeline with small subset")
    
    try:
        # Import ablation pipeline
        from ablation_study_pipeline import AblationStudyPipeline
        
        # Test parameters
        project_root = '/home/ubuntu/checker-framework/checker/tests/index'
        warnings_file = '/home/ubuntu/GenDATA/index1.out'
        cfwr_root = '/home/ubuntu/GenDATA'
        output_dir = 'test_ablation_studies'
        
        # Check if required files exist
        if not os.path.exists(project_root):
            logger.warning(f"Project root not found: {project_root}")
            return False
        
        if not os.path.exists(warnings_file):
            logger.warning(f"Warnings file not found: {warnings_file}")
            return False
        
        # Initialize pipeline
        logger.info("Initializing AblationStudyPipeline...")
        pipeline = AblationStudyPipeline(
            project_root=project_root,
            warnings_file=warnings_file,
            cfwr_root=cfwr_root,
            output_dir=output_dir,
            device='cpu'  # Use CPU for testing
        )
        
        logger.info("✓ Pipeline initialized successfully")
        
        # Test directory creation
        logger.info("Testing directory creation...")
        baseline_dir = pipeline.ablation_dirs['baseline']
        no_aug_dir = pipeline.ablation_dirs['no_augmentation']
        no_rw_dir = pipeline.ablation_dirs['no_random_walk']
        
        if baseline_dir.exists() and no_aug_dir.exists() and no_rw_dir.exists():
            logger.info("✓ All ablation directories created successfully")
        else:
            logger.error("✗ Failed to create ablation directories")
            return False
        
        # Test transformation list
        logger.info("Testing transformation list...")
        transforms = pipeline.transformation_ablations
        if len(transforms) == 27:
            logger.info(f"✓ Found {len(transforms)} transformations")
        else:
            logger.warning(f"Expected 27 transformations, found {len(transforms)}")
        
        # Test a single transformation ablation (dry run)
        logger.info("Testing single transformation ablation (dry run)...")
        test_transform = transforms[0]  # Use first transformation
        ablation_dir = pipeline.ablation_dirs[f'ablate_{test_transform}']
        
        if ablation_dir.exists():
            logger.info(f"✓ Transformation ablation directory created for: {test_transform}")
        else:
            logger.error(f"✗ Failed to create transformation ablation directory for: {test_transform}")
            return False
        
        # Test evaluator
        logger.info("Testing ablation evaluator...")
        try:
            from ablation_study_evaluator import AblationStudyEvaluator
            evaluator = AblationStudyEvaluator(output_dir)
            logger.info("✓ AblationStudyEvaluator initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize AblationStudyEvaluator: {e}")
            return False
        
        # Test report generator
        logger.info("Testing report generator...")
        try:
            from ablation_study_report_generator import AblationStudyReportGenerator
            generator = AblationStudyReportGenerator(output_dir)
            logger.info("✓ AblationStudyReportGenerator initialized successfully")
        except Exception as e:
            logger.error(f"✗ Failed to initialize AblationStudyReportGenerator: {e}")
            return False
        
        # Test augmentation components
        logger.info("Testing augmentation components...")
        try:
            from enhanced_semantic_augment_slices import EnhancedSemanticTransformer
            from simple_code_semantic_augment_slices import SimpleCodeSemanticTransformer
            
            # Test with disabled transformations
            disabled_transforms = ['loop_conversion', 'guard_reversal']
            enhanced_transformer = EnhancedSemanticTransformer(disabled_transformations=disabled_transforms)
            simple_transformer = SimpleCodeSemanticTransformer(disabled_transformations=disabled_transforms)
            
            if len(enhanced_transformer.disabled_transformations) == 2:
                logger.info("✓ EnhancedSemanticTransformer with disabled transformations works")
            else:
                logger.error("✗ EnhancedSemanticTransformer disabled transformations failed")
                return False
            
            if len(simple_transformer.disabled_transformations) == 2:
                logger.info("✓ SimpleCodeSemanticTransformer with disabled transformations works")
            else:
                logger.error("✗ SimpleCodeSemanticTransformer disabled transformations failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Failed to test augmentation components: {e}")
            return False
        
        # Test random walk optimizer
        logger.info("Testing random walk optimizer...")
        try:
            from augmentation_policy_learner import RandomWalkOptimizer
            
            # Test with random walk enabled
            rw_enabled = RandomWalkOptimizer(enable_random_walk=True)
            logger.info("✓ RandomWalkOptimizer with random walk enabled works")
            
            # Test with random walk disabled
            rw_disabled = RandomWalkOptimizer(enable_random_walk=False)
            logger.info("✓ RandomWalkOptimizer with random walk disabled works")
            
        except Exception as e:
            logger.error(f"✗ Failed to test random walk optimizer: {e}")
            return False
        
        logger.info("✓ All tests passed! Ablation pipeline is ready for use.")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test files"""
    test_dir = Path('test_ablation_studies')
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        logger.info("Cleaned up test files")

def main():
    """Main test function"""
    logger.info("Starting ablation pipeline test")
    
    success = test_ablation_pipeline()
    
    if success:
        logger.info("✓ All tests passed successfully!")
        cleanup_test_files()
        return 0
    else:
        logger.error("✗ Tests failed!")
        cleanup_test_files()
        return 1

if __name__ == '__main__':
    exit(main())
