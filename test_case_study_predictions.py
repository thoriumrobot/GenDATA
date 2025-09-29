#!/usr/bin/env python3
"""
Test case study predictions with the generated CFGs
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_case_study_predictions():
    """Test predictions on case study files with generated CFGs"""
    
    case_study_cfg_dir = '/home/ubuntu/GenDATA/test_case_study_cfg_output'
    models_dir = '/home/ubuntu/GenDATA/models_annotation_types'
    
    # Import the predictor
    from model_based_predictor import ModelBasedPredictor
    
    # Create predictor
    predictor = ModelBasedPredictor(models_dir=models_dir, auto_train=True)
    
    # Test with one base model
    base_model = 'enhanced_causal'
    logger.info(f"Testing with base model: {base_model}")
    
    # Load or train the model
    if not predictor.load_or_train_models(base_model_type=base_model, episodes=10, 
                                        project_root='/home/ubuntu/checker-framework/checker/tests/index'):
        logger.error(f"Failed to load/train {base_model} model")
        return False
    
    # Find a test Java file
    test_files = []
    for root, dirs, files in os.walk('/home/ubuntu/GenDATA/case_studies'):
        for file in files:
            if file.endswith('.java') and len(test_files) < 2:
                test_files.append(os.path.join(root, file))
    
    logger.info(f"Testing predictions on {len(test_files)} files")
    
    total_predictions = 0
    
    for java_file in test_files:
        try:
            logger.info(f"Predicting for: {os.path.basename(java_file)}")
            
            # Debug: Check CFG file exists
            java_basename = os.path.splitext(os.path.basename(java_file))[0]
            cfg_file = os.path.join(case_study_cfg_dir, java_basename, 'cfg.json')
            logger.info(f"Looking for CFG file: {cfg_file}")
            logger.info(f"CFG file exists: {os.path.exists(cfg_file)}")
            
            # Use the case study CFG directory with lower threshold for debugging
            preds = predictor.predict_annotations_for_file_with_cfg(java_file, case_study_cfg_dir, threshold=0.1)
            
            if preds:
                logger.info(f"Generated {len(preds)} predictions for {os.path.basename(java_file)}")
                total_predictions += len(preds)
                
                # Show first prediction as example
                if preds:
                    pred = preds[0]
                    logger.info(f"Example prediction: {pred.get('annotation_type', 'N/A')} at line {pred.get('line', 'N/A')} with confidence {pred.get('confidence', 'N/A')}")
            else:
                logger.warning(f"No predictions generated for {os.path.basename(java_file)}")
                
        except Exception as e:
            logger.error(f"Prediction failed for {java_file}: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to debug the CFG data
            try:
                java_basename = os.path.splitext(os.path.basename(java_file))[0]
                cfg_file = os.path.join(case_study_cfg_dir, java_basename, 'cfg.json')
                if os.path.exists(cfg_file):
                    with open(cfg_file, 'r') as f:
                        cfg_data = json.load(f)
                    logger.info(f"CFG data sample: {cfg_data.get('nodes', [])[:2]}")
            except Exception as debug_e:
                logger.error(f"Debug failed: {debug_e}")
    
    logger.info(f"Total predictions generated: {total_predictions}")
    
    if total_predictions > 0:
        logger.info("✅ Case study predictions working!")
        return True
    else:
        logger.error("❌ No predictions generated")
        return False

if __name__ == '__main__':
    success = test_case_study_predictions()
    sys.exit(0 if success else 1)
