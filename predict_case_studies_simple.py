#!/usr/bin/env python3
"""
Simple solution: Generate CFGs for case study files using the existing pipeline
and then run predictions using the same infrastructure that works for Checker Framework files.
"""

import os
import sys
import json
import subprocess
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CASE_STUDIES_ROOT = os.path.join(os.getcwd(), 'case_studies')
PRED_CFG_DIR = os.path.join(os.getcwd(), 'prediction_cfg_output')
PRED_OUT_DIR = os.path.join(os.getcwd(), 'predictions_annotation_types')
MODELS_DIR = os.path.join(os.getcwd(), 'models_annotation_types')

def run(cmd: List[str]):
    """Run a command and exit if it fails"""
    logger.info("$ " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        logger.error(f"Command failed with return code: {res.returncode}")
        sys.exit(res.returncode)

def generate_case_study_cfgs():
    """Generate CFGs for case study files using the existing simple pipeline"""
    logger.info("Generating CFGs for case study files using simple pipeline...")
    
    # Use the simple pipeline to generate CFGs for case study files
    # This will create CFGs compatible with the existing models
    cmd = [sys.executable, 'simple_annotation_type_pipeline.py', '--mode', 'predict', '--target_file']
    
    # Find all Java files in case studies
    java_files = []
    for root, dirs, files in os.walk(CASE_STUDIES_ROOT):
        for file in files:
            if file.endswith('.java'):
                java_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(java_files)} Java files in case studies")
    
    # Process each file individually to generate CFGs
    processed_count = 0
    for java_file in java_files[:10]:  # Limit to first 10 files for testing
        try:
            logger.info(f"Processing {os.path.basename(java_file)}...")
            file_cmd = cmd + [java_file]
            run(file_cmd)
            processed_count += 1
        except Exception as e:
            logger.warning(f"Failed to process {java_file}: {e}")
    
    logger.info(f"Successfully processed {processed_count} case study files")

def list_java_files(root: str) -> List[str]:
    """List all Java files in the given directory tree"""
    files: List[str] = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.endswith('.java'):
                files.append(os.path.join(r, f))
    return files

def predict_for_file(predictor, java_file: str, base_model_type: str) -> List[Dict]:
    """Predict annotations for a Java file using existing CFG infrastructure"""
    from model_based_predictor import ModelBasedPredictor
    
    try:
        # Use the existing prediction method that works with Checker Framework CFGs
        preds = predictor.predict_annotations_for_file_with_cfg(java_file, PRED_CFG_DIR, threshold=0.3)
        
        # Tag with model_type if not present
        for p in preds:
            p.setdefault('model_type', base_model_type)
        
        return preds
    except Exception as e:
        logger.error(f"Prediction failed for {java_file}: {e}")
        return []

def main():
    """Main function to run predictions on case studies"""
    os.makedirs(PRED_OUT_DIR, exist_ok=True)
    
    # 1) Generate CFGs for case study files using the simple pipeline
    generate_case_study_cfgs()
    
    # 2) Prepare predictor
    from model_based_predictor import ModelBasedPredictor
    predictor = ModelBasedPredictor(models_dir=MODELS_DIR, auto_train=True)
    
    base_models = ['enhanced_causal']  # Test with just one model first
    java_files = list_java_files(CASE_STUDIES_ROOT)
    logger.info(f"Found {len(java_files)} Java files under case_studies/")
    
    total_predictions = 0
    
    for base in base_models:
        logger.info(f"== Base model: {base} ==")
        
        if not predictor.load_or_train_models(base_model_type=base, episodes=10, 
                                            project_root='/home/ubuntu/checker-framework/checker/tests/index'):
            logger.warning(f"Skipping {base}: load/train failed")
            continue
        
        per_file_results: Dict[str, List[Dict]] = {}
        model_predictions = 0
        
        # Test with first few files
        for jf in java_files[:5]:
            try:
                preds = predict_for_file(predictor, jf, base)
                if preds:
                    per_file_results.setdefault(jf, []).extend(preds)
                    model_predictions += len(preds)
            except Exception as e:
                logger.warning(f"Prediction failed for {jf} ({base}): {e}")
        
        # Save grouped predictions for this base model
        out_path = os.path.join(PRED_OUT_DIR, f"case_studies_{base}.predictions.json")
        with open(out_path, 'w') as f:
            json.dump(per_file_results, f, indent=2)
        
        logger.info(f"Saved {model_predictions} predictions for {base} to: {out_path}")
        total_predictions += model_predictions
    
    logger.info(f"Total predictions generated: {total_predictions}")
    logger.info("✅ Case study predictions completed successfully!")

if __name__ == '__main__':
    main()
