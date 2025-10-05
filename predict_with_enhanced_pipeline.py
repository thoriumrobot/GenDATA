#!/usr/bin/env python3
"""
Predict with Enhanced Pipeline (No Augmentation)

This script runs prediction using the enhanced pipeline defaults:
- Enhanced Soot slicer with forward/backward slicing
- No augmentation during prediction
- Direct slicing of target files
- Uses trained models for annotation placement
"""

import os
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_prediction_with_enhanced_pipeline(target_file=None, case_studies_dir=None, 
                                        models_dir=None, output_dir=None):
    """
    Run prediction using enhanced pipeline without augmentation
    
    Args:
        target_file: Specific Java file to predict on
        case_studies_dir: Directory containing case study projects
        models_dir: Directory containing trained models
        output_dir: Output directory for predictions
    """
    
    # Import the simple annotation type pipeline
    from simple_annotation_type_pipeline import SimpleAnnotationTypePipeline
    
    # Set default paths
    if not case_studies_dir:
        case_studies_dir = '/home/ubuntu/GenDATA/case_studies'
    if not models_dir:
        models_dir = '/home/ubuntu/GenDATA/models_annotation_types'
    if not output_dir:
        output_dir = '/home/ubuntu/GenDATA/predictions_annotation_types'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("🔮 Starting prediction with enhanced pipeline (no augmentation)")
    logger.info(f"📁 Case studies directory: {case_studies_dir}")
    logger.info(f"📁 Models directory: {models_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    if target_file:
        logger.info(f"🎯 Target file: {target_file}")
    else:
        logger.info("🎯 Target: All case study projects")
    
    # Create pipeline instance for prediction
    pipeline = SimpleAnnotationTypePipeline(
        project_root=case_studies_dir,  # Use case studies as project root for prediction
        warnings_file='/home/ubuntu/GenDATA/index1.out',  # Dummy warnings file
        cfwr_root='/home/ubuntu/GenDATA',
        mode='predict',
        augment_first=False  # No augmentation during prediction
    )
    
    # Run prediction pipeline
    logger.info("🚀 Running prediction pipeline...")
    success = pipeline.run_prediction_pipeline(target_file)
    
    if success:
        logger.info("✅ Prediction completed successfully")
        logger.info(f"📊 Results saved to: {output_dir}")
        
        # List prediction files
        prediction_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.json') or file.endswith('.java'):
                    prediction_files.append(os.path.join(root, file))
        
        logger.info(f"📋 Generated {len(prediction_files)} prediction files")
        for pred_file in prediction_files[:10]:  # Show first 10 files
            logger.info(f"  - {pred_file}")
        
        if len(prediction_files) > 10:
            logger.info(f"  ... and {len(prediction_files) - 10} more files")
        
        return True
    else:
        logger.error("❌ Prediction failed")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict with Enhanced Pipeline (No Augmentation)')
    parser.add_argument('--target_file', 
                       help='Specific Java file to predict on')
    parser.add_argument('--case_studies_dir', 
                       default='/home/ubuntu/GenDATA/case_studies',
                       help='Directory containing case study projects')
    parser.add_argument('--models_dir', 
                       default='/home/ubuntu/GenDATA/models_annotation_types',
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', 
                       default='/home/ubuntu/GenDATA/predictions_annotation_types',
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # Verify models directory exists
    if not os.path.exists(args.models_dir):
        logger.error(f"❌ Models directory not found: {args.models_dir}")
        logger.error("Please train models first using: python train_all_21_models.py")
        return 1
    
    # Count available models
    model_files = []
    for file in os.listdir(args.models_dir):
        if file.endswith('_model.pth'):
            model_files.append(file)
    
    logger.info(f"📊 Found {len(model_files)} trained models")
    
    if len(model_files) == 0:
        logger.error("❌ No trained models found")
        logger.error("Please train models first using: python train_all_21_models.py")
        return 1
    
    # Run prediction
    success = run_prediction_with_enhanced_pipeline(
        target_file=args.target_file,
        case_studies_dir=args.case_studies_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )
    
    if success:
        logger.info("🎉 Prediction with enhanced pipeline completed successfully")
        return 0
    else:
        logger.error("❌ Prediction with enhanced pipeline failed")
        return 1

if __name__ == '__main__':
    exit(main())
